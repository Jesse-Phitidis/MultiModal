import torchio as tio
import torch
import torch.nn.functional as F
import random
import numpy as np
import copy
from typing import Dict
import os

class CopyAnyAffine(tio.Transform):

    def __init__(self):
        super().__init__()

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        keys = list(subject.get_images_names())
        reference = subject[keys[0]]
        affine = copy.deepcopy(reference.affine)
        for key in keys[1:]:
            image = subject[key]
            image.load()
            if not np.allclose(affine, image.affine, atol=1e-5):
                raise RuntimeError(
                    f"Not all affines for the subject are close. Found: \n\n{reference.path}:\n{affine}\n{image.path}:\n{image.affine}\n"
                    )
            image.affine = affine
        return subject


def get_min_max(subject):
        min_max = {}
        for key, value in subject.get_images_dict().items():
            if isinstance(value, tio.ScalarImage):
                min = value.data.min()
                max = value.data.max()
                min_max[key] = (min, max)
        return min_max


class Contrast(tio.transforms.Transform):

    def __init__(self, rng=(0.65, 1.5), **kwargs):
        super().__init__(**kwargs)
        self.rng = rng

    def apply_transform(self, subject):

        x = random.uniform(*self.rng)

        min_max = get_min_max(subject)

        for key, value in subject.get_images_dict().items():
            if isinstance(value, tio.ScalarImage):
                min, max = min_max[key][0], min_max[key][1]
                normalised_data = (value.data - min) / (max - min)
                scaled_data = normalised_data * x
                clamped_data = torch.clamp(scaled_data, 0, 1)
                value.set_data(clamped_data * (max - min) + min)

        return subject
    
    
class SimulateLowResolution(tio.transforms.Transform):

    def __init__(self, factor=(1, 2), p_per_mod=0.5, **kwargs):
        super().__init__(**kwargs)
        self.factor = factor
        self.p_per_mod = p_per_mod

    def apply_transform(self, subject):

        for image in subject.get_images():
            T = tio.transforms.RandomAnisotropy(axes=(0,1,2), downsampling=self.factor, image_interpolation="bspline", p=self.p_per_mod)
            image.set_data(T(image.data))

        return subject
    
    
class Gamma(tio.transforms.Transform):

    def __init__(self, rng=(0.7, 1.5), **kwargs):
        super().__init__(**kwargs)
        self.rng = rng

    def apply_transform(self, subject):

        gamma = random.uniform(*self.rng)
        min_max = get_min_max(subject)

        for key, value in subject.get_images_dict().items():
            if isinstance(value, tio.ScalarImage):
                min, max = min_max[key][0], min_max[key][1]
                normalised_data = (value.data - min) / (max - min)
                if random.random() < 0.15:
                    augmented_data = 1 - (1 - normalised_data) ** gamma
                else:
                    augmented_data = normalised_data ** gamma
                rescaled_data = augmented_data * (max - min) + min
                value.set_data(rescaled_data)

        return subject
    
    
class AddMissingModalities(tio.Transform):
    
    def __init__(self, all_modalities: list[str], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.all_modalities = all_modalities
        
    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        subject_keys = subject.get_images_names()
        missing_keys = [k for k in self.all_modalities if k not in subject_keys]
        
        if len(missing_keys) == 0:
            return subject
            
        ref_image = subject.get_first_image()
        zero_tensor = torch.zeros_like(ref_image.data)
        
        for key in missing_keys:
            new_image = tio.ScalarImage(tensor=zero_tensor, affine=ref_image.affine)
            subject.add_image(new_image, key)
            
        return subject
    
    
class ResampleWithinRange(tio.Transform):
    
    def __init__(self, target, resample_kwargs={}, to_midpoint=False, **kwargs):
        super().__init__(**kwargs)
        if len(target) != 6:
            target = 3 * target
            assert len(target) == 6
        target = [(mini, maxi) for mini, maxi in zip(target[::2], target[1::2])]
        self.target = target
        self.resample_kwargs = resample_kwargs
        self.to_midpoint = to_midpoint
        
    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        target = copy.deepcopy(list(subject.spacing))
        in_rng_count = 0
        for i, (rng, s) in enumerate(zip(self.target, target)):
            if s >= rng[0] and s <= rng[1]:
                in_rng_count += 1
                continue
            if self.to_midpoint:
                target[i] = (rng[0] + rng[1]) / 2
            else:
                if abs(s - rng[0]) < abs(s - rng[1]):
                    target[i] = rng[0]
                else:
                    target[i] = rng[1]
        T = tio.Resample(target, **self.resample_kwargs)
        
        if in_rng_count == 3:
            return subject
        else:
            return T(subject)