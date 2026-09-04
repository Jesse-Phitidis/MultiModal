import monai.transforms as t


# Class for keeping the min and max intensity scale after transforms
class TransformsKeepScale(t.Transform):
    def __init__(self, transform, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transform = transform
        
    def __call__(self, data):
        min_data, max_data = data.min(), data.max()
        data = self.transform(data)
        data = t.ScaleIntensity(min_data, max_data)(data)
        return data
    

# Class for any transform happen in coarse patches
class RandCoarseGeneral(t.RandCoarseTransform):
    
    def __init__(self, transform, *args, **kwargs):
        super().__init__(*args, **kwargs)   
        self.transform = TransformsKeepScale(transform)
        
    def _transform_holes(self, img):
        for h in self.hole_coords:
            img[h] = self.transform(img[h])
        return img


# Individvidual transforms preconfigured to make images almost unusable
coarse_config = {"prob": 1.0, "holes": 3, "spatial_size": 40, "max_holes": 10, "max_spatial_size": 80}
rand_gaussian_noise = t.RandGaussianNoise(prob=1.0,mean=0,std=3)
rand_gaussian_noise_coarse = RandCoarseGeneral(transform=rand_gaussian_noise,**coarse_config)
rand_bias_field = t.RandBiasField(prob=1.0,degree=3,coeff_range=(0.4,0.9))
rand_adjust_contrast_low = t.RandAdjustContrast(prob=1.0,gamma=(0.0,0.1))
rand_adjust_contrast_low_coarse = RandCoarseGeneral(transform=rand_adjust_contrast_low,**coarse_config)
rand_adjust_contrast_high = t.RandAdjustContrast(prob=1.0,gamma=(10.0,30.0))
rand_adjust_contrast_high_coarse = RandCoarseGeneral(transform=rand_adjust_contrast_high,**coarse_config)
rand_gaussian_smooth = t.RandGaussianSmooth(prob=1.0,sigma_x=(2,6),sigma_y=(2,6),sigma_z=(2,6))
rand_gaussian_smooth_coarse = RandCoarseGeneral(transform=rand_gaussian_smooth,**coarse_config)
rand_gibbs_noise = t.RandGibbsNoise(prob=1.0, alpha=(0.75, 1.0))
rand_gibbs_noise_coarse = RandCoarseGeneral(transform=rand_gibbs_noise, **coarse_config)
rand_kspace_spike_noise = t.RandKSpaceSpikeNoise(prob=1.0, intensity_range=(15,18))
rand_kspace_spike_noise_coarse = RandCoarseGeneral(transform=rand_kspace_spike_noise, **coarse_config)
rand_rician_noise = t.RandRicianNoise(prob=1.0, mean=0, std=1)
rand_rician_noise_coarse = RandCoarseGeneral(transform=rand_rician_noise, **coarse_config)
rand_dropout_coarse = t.RandCoarseDropout(dropout_holes=True, fill_value=0, **coarse_config)
dropin_config = {"prob": 1.0, "holes": 5, "spatial_size": 5, "max_holes": 30, "max_spatial_size": 30}
rand_dropin_coarse = t.RandCoarseDropout(dropout_holes=False, fill_value=0, **dropin_config)


# One-of transform with weights
one_of_corruption_keep_scale = TransformsKeepScale(t.OneOf(
    transforms=[
        rand_gaussian_noise,
        rand_gaussian_noise_coarse,
        rand_bias_field,
        rand_adjust_contrast_low,
        rand_adjust_contrast_low_coarse,
        rand_adjust_contrast_high,
        rand_adjust_contrast_high_coarse,
        rand_gaussian_smooth,
        rand_gaussian_smooth_coarse,
        rand_gibbs_noise,
        rand_gibbs_noise_coarse,
        rand_kspace_spike_noise,
        rand_kspace_spike_noise_coarse,
        rand_rician_noise,
        rand_rician_noise_coarse,
        rand_dropout_coarse,
        rand_dropin_coarse
    ]
))


# Final class to use in lightning CLI to avoid unreadable config file
class CorruptionTransform(t.Transform):
    def __call__(self, data):
        return one_of_corruption_keep_scale(data)
    
    
# Test time corruption wrapper
class TestTimeCorruption:
    
    def __init__(self, transform):
        self.transform = TransformsKeepScale(transform)
        
    def __call__(self, images, idxs):
        for b in range(images.shape[0]):
            for idx in idxs:
                self.transform.transform.set_random_state(seed=0) # deterministic 
                images[b,idx:idx+1] = self.transform(images[b,idx:idx+1])
        return images
    
    
# False present transform
class FalsePresent(t.Transform):
        
    def __call__(self, data):
        data[...] = 0.0
        data[...,0] = 1.0
        return data
    
    def set_random_state(self,*args, **kwargs):
        pass