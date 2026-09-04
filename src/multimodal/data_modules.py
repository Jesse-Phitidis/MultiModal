import pytorch_lightning as pl
import torchio as tio
import pandas as pd
from typing import Sequence
from pathlib import Path
import torch
import nibabel as nib


class DataModule(pl.LightningDataModule):
    
    def __init__(
        self, 
        data_dir: str, 
        train_csv: str | None, 
        val_csv: str | None,
        test_csv: str | None,
        predict_csv: str | None,
        modalities: Sequence[str],
        labels: Sequence[str],
        train_transforms: tio.Transform,
        val_transforms: tio.Transform,
        inference_transforms: tio.Transform,
        sampler: tio.sampler.PatchSampler,
        queue_max_length: int,
        samples_per_volume: int,
        num_workers: int,
        batch_size: int,
        load_in_memory: bool = False
        ) -> None:
        super().__init__()
    
        self.data_dir = Path(data_dir)
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.test_csv = test_csv
        self.predict_csv = predict_csv
        self.modalities = modalities
        self.labels = labels
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.inference_transforms = inference_transforms
        self.sampler = sampler
        self.queue_max_length = queue_max_length
        self.samples_per_volume = samples_per_volume
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.load_in_memory = load_in_memory
        
    def _get_subjects(self, csv: pd.DataFrame, is_predict_step: bool = False) -> list[tio.Subject]:
        subject_list = []
        for _, row in csv.iterrows():
            subject_dict = {}
            at_least_one_modality = False
            for mod in self.modalities:
                file = self.data_dir / "images" / f"{row['idx']}_{mod}.nii.gz"
                if row[mod] == "allowed" and file.exists():
                    if self.load_in_memory:
                        nii = nib.load(file)
                        data = torch.from_numpy(nii.get_fdata()).float()[None]
                        p = Path(file).resolve()
                        subject_dict[mod] = tio.ScalarImage(tensor=data, affine=nii.affine, path=p)
                    else:
                        subject_dict[mod] = tio.ScalarImage(path=file)
                    at_least_one_modality = True
            if not at_least_one_modality: # skip subjects with no images
                continue
            for lab in self.labels:
                if is_predict_step:
                    continue
                file = self.data_dir / "labels" / f"{row['idx']}_{lab}.nii.gz"
                assert file.exists(), f"All subjects must have all labels but the file '{file}' does not exist"
                if self.load_in_memory:
                    nii = nib.load(file)
                    data = torch.from_numpy(nii.get_fdata()).float()[None]
                    p = Path(file).resolve()
                    subject_dict[lab] = tio.LabelMap(tensor=data, affine=nii.affine, path=p)
                else:
                    subject_dict[lab] = tio.LabelMap(path=file)
            subject_dict["idx"] = row["idx"]
            subject_list.append(tio.Subject(subject_dict))
        return subject_list

    def setup(self, stage: str) -> None:
        
        if stage=="fit":
            # train
            subjects = self._get_subjects(pd.read_csv(self.train_csv))
            self.train_dataset = tio.SubjectsDataset(subjects, self.train_transforms)
            # val
            subjects = self._get_subjects(pd.read_csv(self.val_csv))
            self.val_dataset = tio.SubjectsDataset(subjects, self.val_transforms)
        if stage=="test":
            # test
            subjects = self._get_subjects(pd.read_csv(self.test_csv))
            self.test_dataset = tio.SubjectsDataset(subjects, self.inference_transforms)
        if stage=="predict":
            # predict
            subjects = self._get_subjects(pd.read_csv(self.predict_csv), is_predict_step=True)
            self.predict_dataset = tio.SubjectsDataset(subjects, self.inference_transforms)
    
    def train_dataloader(self) -> tio.SubjectsLoader:
        queue =  tio.Queue(
            subjects_dataset=self.train_dataset,
            max_length=self.queue_max_length,
            samples_per_volume=self.samples_per_volume,
            sampler=self.sampler,
            num_workers=self.num_workers
        )
        return tio.SubjectsLoader(dataset=queue, batch_size=self.batch_size, num_workers=0)
    
    def val_dataloader(self) -> tio.SubjectsLoader:
        return tio.SubjectsLoader(dataset=self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def test_dataloader(self) -> tio.SubjectsLoader:
        return tio.SubjectsLoader(dataset=self.test_dataset, batch_size=1, num_workers=self.num_workers)
    
    def predict_dataloader(self) -> tio.SubjectsLoader:
        return tio.SubjectsLoader(dataset=self.predict_dataset, batch_size=1, num_workers=self.num_workers)