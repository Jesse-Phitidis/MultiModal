import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.inferers import sliding_window_inference
from torchmetrics.functional.segmentation import dice_score
from typing import Sequence, Any, Callable
from collections import defaultdict
from pathlib import Path
import nibabel as nib
import numpy as np
import torchio as tio
import json


class LightningModule(pl.LightningModule):
    
    def __init__(
        self,
        network: nn.Module,
        loss_fn: nn.Module,
        patch_size: Sequence[int],
        modality_augmentation_fn: Callable,
        inference_corruption_fn: Callable,
        modality_loss_weight: Sequence[float],
        num_epochs_skip_weight: int,
        num_epochs_all_modalities: int,
        inference_modalities: Sequence[Sequence[str]],
        inference_corruption: Sequence[Sequence[str]],
        pred_dir: str | None = None,
        metric_dir: str | None = None,
        ):
        super().__init__()
        
        assert len(inference_modalities) == len(inference_corruption)
        
        self.network = network
        self.loss_fn = loss_fn
        self.patch_size = patch_size
        self.modality_augmentation_function = modality_augmentation_fn
        self.inference_corruption_fn = inference_corruption_fn
        self.modality_loss_weight = torch.tensor([modality_loss_weight])
        self.num_epochs_skip_weight = num_epochs_skip_weight
        self.num_epochs_all_modalities = num_epochs_all_modalities
        self.inference_modalities = inference_modalities
        self.inference_corruption = inference_corruption
        self.pred_dir = Path(pred_dir) if pred_dir else None
        self.metric_dir = Path(metric_dir) if metric_dir else None
        self.save_hyperparameters(ignore=[
            "network", "loss_fn", "modality_augmentation_fn", "inference_corruption_fn", 
            "inference_modalities", "inference_corruption", "pred_dir", "metric_dir", 
            ])
        
    def forward(self, x: torch.Tensor, *args, **kwargs) -> list[torch.Tensor]:
        pred = self.network(x, *args, **kwargs)
        if isinstance(pred, list):
            return pred
        elif isinstance(pred, torch.Tensor):
            return [pred]
        else:
            raise Exception(f"network should return a tensor or a list of tensors but returned {type(pred)}")
        
    def sliding_window_predictor(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor: 
        return torch.softmax(self(x, *args, **kwargs)[-1], dim=1)
        
    def training_step(self, batch: dict) -> torch.Tensor:

        images, labels, modalities_missing = self._prepare_batch(batch) # (B,C,*), (B,K,*), (B,C)
        
        # drop batches with any missing modalities early in training
        if self.current_epoch < self.num_epochs_all_modalities:
            keep_batches = ~modalities_missing.any(dim=1)
            if not keep_batches.any():
                return None # skip batch entirely
            images = images[keep_batches]
            labels = labels[keep_batches]
            modalities_missing = modalities_missing[keep_batches]
        
        # randomly drop modalities by zeroing input channels 
        # (not in transforms since we want to drop datasets with missing modalities early on with the above code)
        images, modalities_corrupted, modalities_dropped = self.modality_augmentation_function(images, modalities_missing, self.current_epoch) # (B,C,*), (B,C)
        modalities_missing = modalities_missing | modalities_dropped
        
        # forward and log unweighted loss
        pred = self(images) # [(B,K,*)]
        loss = self.loss_fn(pred[::-1], labels) # deep supervision loss expects final output first, so reverse list of preds
        self.log("loss", loss.mean(), batch_size=images.shape[0], on_step=False, on_epoch=True)
        
        # apply weight
        if self.current_epoch >= self.num_epochs_skip_weight:
            weight = self.modality_loss_weight.to(device=loss.device).expand_as(modalities_missing).clone()
            weight[modalities_missing] = 0
            weight = weight.sum(dim=1)
            loss = (weight * loss)
        
        loss = loss.mean()
        
        return loss
        
    def validation_step(self, batch: dict):
        
        images, labels, modalities_missing = self._prepare_batch(batch) # (B,C,*), (B,K,*), (B,C)
        
        # do entire val step once per set of inference modalities
        for mod_lst, corrupt_lst in zip(self.inference_modalities, self.inference_corruption):
            
            # get idxs of input channels to zero out or keep for inference with mod_lst
            keep_idxs, drop_idxs = self._get_idxs_for_inference(mod_lst)
            
            # get idxs of input channels to apply corrution
            corrupt_idxs, good_idxs = self._get_idxs_for_inference(corrupt_lst)
                    
            assert not modalities_missing[:,keep_idxs].any(), f"Some of {mod_lst} are missing in a validation batch"
            assert keep_idxs != corrupt_idxs, f"We cannot corrupt all inference idxs"
            
            # zero out dropped modalities
            images_mods = images.clone()
            images_mods[:, drop_idxs] = torch.zeros_like(images_mods[:, drop_idxs])
            
            # corrupt required modalities
            images_mods = self.inference_corruption_fn(images_mods, corrupt_idxs)
            
            # forward on central patch
            pred = self(images_mods)[-1] # final unet layer output
            pred = F.one_hot(torch.argmax(pred, dim=1), num_classes=pred.shape[1]).movedim(-1,1)
            
            # metric calculation
            scores = dice_score(pred, labels, num_classes=pred.shape[1], include_background=False, average="none")
            mods_names = "_".join(mod_lst) + (("-X-" + "_".join(corrupt_lst)) if len(corrupt_idxs)!=0 else "")
            log_score = scores.nanmean()
            if not torch.isnan(log_score):
                self.log(f"{mods_names}/macro_avg", log_score, batch_size=scores.shape[0], on_step=False, on_epoch=True)
            for i, lab in enumerate(self.trainer.datamodule.labels):
                log_score = scores[:,i].nanmean()
                if not torch.isnan(log_score):
                    self.log(f"{mods_names}/{lab}", log_score, batch_size=scores.shape[0], on_step=False, on_epoch=True)
            
    def on_test_epoch_start(self):
        self.jsons = defaultdict(dict)
        assert self.metric_dir, "metric_dir must be set for test"
        self._make_output_dirs(preds=True, metrics=True)
            
    def test_step(self, batch):
        
        # WARNING - if all images in a batch are not the same size, this step will fail
        # because the dataloader won't be able to collate the batch. Validation and training steps
        # use patch sampling, but test and predict steps use sliding window inference.
        # it is safest to enforce batch_size=1 in the test loader
        
        images, labels, modalities_missing = self._prepare_batch(batch) # (B,C,*), (B,K,*), (B,C)
        idxs = batch["idx"]
        
        # do entire test step once per set of inference modalities
        for mod_lst, corrupt_lst in zip(self.inference_modalities, self.inference_corruption):
            
            # get idxs of input channels to zero out or keep for inference with mod_lst
            keep_idxs, drop_idxs = self._get_idxs_for_inference(mod_lst)
            
            # get idxs of input channels to apply corrution
            corrupt_idxs, good_idxs = self._get_idxs_for_inference(corrupt_lst)
            
            mods_names = "_".join(mod_lst) + (("-X-" + "_".join(corrupt_lst)) if len(corrupt_idxs)!=0 else "")
            
            # for each element in the batch, print a warning if any inference modalities not available
            for b, idx in enumerate(idxs):
                missing = []
                for keep_idx in keep_idxs:
                    if modalities_missing[b, keep_idx]:
                        missing.append(self.trainer.datamodule.modalities[keep_idx])
                if len(missing) != 0:
                    print(f"Subject {idx} is missing {missing} while attempting to use {mod_lst} for inference")
            
            # zero out dropped modalities
            images_mods = images.clone()
            images_mods[:, drop_idxs] = torch.zeros_like(images_mods[:, drop_idxs])
            
            # corrupt required modalities
            images_mods = self.inference_corruption_fn(images_mods, corrupt_idxs)
            
            # foward using sliding window inference
            pred = sliding_window_inference(
                inputs=images_mods,
                roi_size=self.patch_size,
                sw_batch_size=images_mods.shape[0],
                predictor=self.sliding_window_predictor,
            )
            pred = F.one_hot(torch.argmax(pred, dim=1), num_classes=pred.shape[1]).movedim(-1,1)
            
            # metric calculation in label resolution (can be different to pred resolution if scalar_only is set in tio.Resample)
            scores = []
            for b in range(len(idxs)):
                pred_b = F.interpolate(pred[b:b+1].float(), size=labels.shape[2:], mode="nearest-exact")
                score = dice_score(pred_b, labels[b:b+1], num_classes=pred.shape[1], include_background=False, average="none")
                scores.append(score.nan_to_num(nan=1.0))
            scores = torch.cat(scores, dim=0)
            
            # per subject metric logging
            for b in range(len((idxs))):
                subject_scores_dict = {}
                for i in range(len((self.trainer.datamodule.labels))):
                    lab_name = self.trainer.datamodule.labels[i]
                    subject_scores_dict[lab_name] = scores[b,i].item()
                    
                    # save predictions if required
                    if self.pred_dir:
                        self._save_predictions(mod_lst, missing, batch, pred, b, i, mods_names)
                        
                subject_scores_dict["macro_avg"] = scores[b].nanmean().item()
                self.jsons[mods_names][idxs[b]] = subject_scores_dict
                
    def on_test_epoch_end(self):
        
        for mods_names, json_data in self.jsons.items():
            agg = defaultdict(list)
            for sub, metrics in json_data.items():
                for metric_name, metric_value in metrics.items():
                    agg[metric_name].append(metric_value)
            json_data["mean"] = {}
            json_data["median"] = {}
            json_data["std"] = {}
            for metric_name, metric_lst in agg.items():
                json_data["mean"][metric_name] = float(np.nanmean(metric_lst))
                json_data["median"][metric_name] = float(np.nanmedian(metric_lst))
                json_data["std"][metric_name] = float(np.nanstd(metric_lst))
                
            save_path = self.metric_dir / f"{mods_names}.json"
            with open(save_path, "w") as f:
                json.dump(json_data, f, indent=4)
            
    def on_predict_epoch_start(self):
        assert self.pred_dir, "pred_dir must be set for predict"
        self._make_output_dirs(preds=True, metrics=False)
    
    def predict_step(self, batch: dict):
        
        # WARNING - if all images in a batch are not the same size, this step will fail
        # because the dataloader won't be able to collate the batch. Validation and training steps
        # use patch sampling, but test and predict steps use sliding window inference.
        # it is safest to enforce batch_size=1 in the test loader
        
        images, _, modalities_missing = self._prepare_batch(batch) # (B,C,*), (B,K,*), (B,C)
        idxs = batch["idx"]
        
        # do entire predict step once per set of inference modalities
        for mod_lst, corrupt_lst in zip(self.inference_modalities, self.inference_corruption):
            
            # get idxs of input channels to zero out or keep for inference with mod_lst
            keep_idxs, drop_idxs = self._get_idxs_for_inference(mod_lst)
            
            # get idxs of input channels to apply corrution
            corrupt_idxs, good_idxs = self._get_idxs_for_inference(corrupt_lst)
            
            mods_names = "_".join(mod_lst) + (("-X-" + "_".join(corrupt_lst)) if len(corrupt_idxs)!=0 else "")
            
            # for each element in the batch, print a warning if any inference modalities not available
            for b, idx in enumerate(idxs):
                missing = []
                for keep_idx in keep_idxs:
                    if modalities_missing[b, keep_idx]:
                        missing.append(self.trainer.datamodule.modalities[keep_idx])
                if len(missing) != 0:
                    print(f"Subject {idx} is missing {missing} while attempting to use {mod_lst} for inference")
            
            # zero out dropped modalities
            images_mods = images.clone()
            images_mods[:, drop_idxs] = torch.zeros_like(images_mods[:, drop_idxs])
            
            # corrupt required modalities
            images_mods = self.inference_corruption_fn(images_mods, corrupt_idxs)
            
            # foward using sliding window inference
            pred = sliding_window_inference(
                inputs=images_mods,
                roi_size=self.patch_size,
                sw_batch_size=images_mods.shape[0],
                predictor=self.sliding_window_predictor,
            )
            pred = F.one_hot(torch.argmax(pred, dim=1), num_classes=pred.shape[1]).movedim(-1,1)
            
            # per subject prediction
            for b in range(len(idxs)):
                for i in range(len(self.trainer.datamodule.labels)):
                    self._save_predictions(mod_lst, missing, batch, pred, b, i, mods_names)
    
    def _prepare_batch(self, batch: dict) -> tuple[torch.Tensor]:
        modalities_missing = []
        images_lst = []
        for mod in self.trainer.datamodule.modalities:
            data = batch[mod]["data"]
            mod_status = data.abs().sum(dim=[i for i in range(2, len(data.shape))]) == 0
            modalities_missing.append(mod_status)
            images_lst.append(data)
        images = torch.cat(images_lst, dim=1)
        labels_lst = []
        for lab in self.trainer.datamodule.labels:
            if lab in batch: # might not be for predict step
                labels_lst.append(batch[lab]["data"])
        if len(labels_lst) != 0:
            fg = torch.max(torch.cat(labels_lst, dim=1), dim=1, keepdim=True).values
            bg = 1 - fg
            labels_lst.insert(0, bg) 
            labels = torch.cat(labels_lst, dim=1)
        else:
            labels = None # predict step
        modalities_missing = torch.cat(modalities_missing, dim=1)
        return images, labels, modalities_missing
    
    def _get_idxs_for_inference(self, mod_lst: Sequence[str]):
        keep_idxs, drop_idxs = [], []
        for i, mod_name in enumerate(self.trainer.datamodule.modalities):
            if mod_name in mod_lst:
                keep_idxs.append(i)
            else:
                drop_idxs.append(i)
        return keep_idxs, drop_idxs
    
    def _make_output_dirs(self, preds: bool = True, metrics: bool = True):
        if preds:
            if self.pred_dir:
                self.pred_dir.mkdir(exist_ok=True, parents=True)
        if metrics:
            if self.metric_dir:
                self.metric_dir.mkdir(exist_ok=True, parents=True)
                
    def _save_predictions(
        self, 
        mod_lst: list[str], 
        missing: list[str],
        batch: dict, 
        pred: torch.Tensor, 
        b: int,
        i: int,
        mods_names: str
        ):
        # save predictions
        ref_mod = list(set(mod_lst)-set(missing))[0] # first modality name which is used and present
        ref = nib.load(batch[ref_mod]["path"][b]) # reference images
        pred_data = pred[b,i+1].cpu().numpy() # (H,W,D) binary prediction for label i
        pred_affine = batch[ref_mod]["affine"][b] # affine of the prediction
        # resample the prediction if required
        if not ((pred_data.shape == ref.shape) and np.allclose(pred_affine, ref.affine)):
            resampler = tio.Resample(target=(ref.shape, ref.affine), image_interpolation="nearest")
            pred_nii = nib.Nifti1Image(pred_data.astype("uint8"), pred_affine)
            pred_data = resampler(pred_nii).get_fdata()
        lab_name = self.trainer.datamodule.labels[i]
        save_name = f"{batch['idx'][b]}_{mods_names}_{lab_name}.nii.gz"
        nii = nib.Nifti1Image(pred_data.astype("uint8"), affine=ref.affine)
        nii.set_data_dtype("uint8")
        nib.save(nii, self.pred_dir / save_name)
        
    def on_test_end(self):
        Path("config.yaml").unlink()
        
    def on_predict_end(self):
        Path("config.yaml").unlink()
        
        
  
  
        
class DynUNetLightningModule(LightningModule):
    def forward(self, x: torch.Tensor, *args, **kwargs) -> list[torch.Tensor]:
        pred = self.network(x)
        if len(pred.shape) > len(x.shape): # DynUNet stacks outputs in dim 1 for deep supervision (final output first)
            return list(pred.unbind(dim=1))[::-1] # reverse to align with my models (final output last)
        else:
            return [pred] # DynUNet without deep supervision
        
        
        
        
        
class RouterLightningModule(LightningModule):
    
    def __init__(
        self, 
        bce_loss_scheduler: Callable | None, 
        do_test_with_router_logging: bool = False, 
        freeze_encoders_epochs: Sequence[int] = (),
        unfreeze_encoders_epochs: Sequence[int] = (),
        freeze_routers_epochs: Sequence[int] = (),
        unfreeze_routers_epochs: Sequence[int] = (),
        freeze_decoder_epochs: Sequence[int] = (),
        unfreeze_decoder_epochs: Sequence[int] = (),
        *args, 
        **kwargs
        ):
        super().__init__(*args, **kwargs)
        self.bce_loss_scheduler = bce_loss_scheduler
        self.do_test_with_router_logging = do_test_with_router_logging
        self.freeze_encoders_epochs = freeze_encoders_epochs
        self.unfreeze_encoders_epochs = unfreeze_encoders_epochs
        self.freeze_routers_epochs = freeze_routers_epochs
        self.unfreeze_routers_epochs = unfreeze_routers_epochs
        self.freeze_decoder_epochs = freeze_decoder_epochs
        self.unfreeze_decoder_epochs = unfreeze_decoder_epochs
        self.save_hyperparameters(ignore=["do_test_with_router_logging"])
    
    def forward(self, x: torch.Tensor, *args, **kwargs) -> list[torch.Tensor]:
        return self.network(x, *args, **kwargs)
    
    def on_train_epoch_start(self):
        self.train_router_logs = defaultdict(list)
        
        if self.current_epoch in self.freeze_encoders_epochs:
            print(f"Freezing encoders at epoch {self.current_epoch}")
            for param in self.network.encoders.parameters():
                param.requires_grad = False
                
        if self.current_epoch in self.unfreeze_encoders_epochs:
            print(f"Unfreezing encoders at epoch {self.current_epoch}")
            for param in self.network.encoders.parameters():
                param.requires_grad = True
                
        if self.current_epoch in self.freeze_routers_epochs:
            print(f"Freezing routers at epoch {self.current_epoch}")
            for param in self.network.routers.parameters():
                param.requires_grad = False
                
        if self.current_epoch in self.unfreeze_routers_epochs:
            print(f"Unfreezing routers at epoch {self.current_epoch}")
            for param in self.network.routers.parameters():
                param.requires_grad = True
                
        if self.current_epoch in self.freeze_decoder_epochs:
            print(f"Freezing decoder at epoch {self.current_epoch}")
            for param in self.network.decoder.parameters():
                param.requires_grad = False
                
        if self.current_epoch in self.unfreeze_decoder_epochs:
            print(f"Unfreezing decoder at epoch {self.current_epoch}")
            for param in self.network.decoder.parameters():
                param.requires_grad = True
        
    def training_step(self, batch: dict) -> torch.Tensor:

        images, labels, modalities_missing = self._prepare_batch(batch) # (B,C,*), (B,K,*), (B,C)
        
        # drop batches with any missing modalities early in training
        if self.current_epoch < self.num_epochs_all_modalities:
            keep_batches = ~modalities_missing.any(dim=1)
            if not keep_batches.any():
                return None # skip batch entirely
            images = images[keep_batches]
            labels = labels[keep_batches]
            modalities_missing = modalities_missing[keep_batches]
        
        # randomly drop modalities by zeroing input channels 
        images, modalities_corrupted, modalities_dropped = self.modality_augmentation_function(images, modalities_missing, self.current_epoch) # (B,C,*), (B,C)
        modalities_missing = modalities_missing | modalities_dropped
        
        # forward and log unweighted loss
        pred, router_weights = self(images, return_weights=True) # [(B,K,*)], [(B,num_encoders,1)]
        loss = self.loss_fn(pred[::-1], labels) # deep supervision loss expects final output first, so reverse list of preds
        self.log("loss", loss.mean(), batch_size=images.shape[0], on_step=False, on_epoch=True)
        
        # apply weight
        if self.current_epoch >= self.num_epochs_skip_weight:
            weight = self.modality_loss_weight.to(device=loss.device).expand_as(modalities_missing).clone()
            weight[modalities_missing] = 0
            weight = weight.sum(dim=1)
            loss = (weight * loss)
            
        loss = loss.mean()
            
        # log router weights 
        router_active_batches = (~modalities_missing).sum(dim=1) > 1 # where there are at least 2 modalities present
        for b in range(images.shape[0]):
            if not router_active_batches[b]:
                continue
            
            idx_mods_active = torch.where((~modalities_missing)[b]==True)[0].cpu()
            
            if len(idx_mods_active) > 1:
                names = "_".join(np.array(self.trainer.datamodule.modalities)[idx_mods_active])
            else:
                names = np.array(self.trainer.datamodule.modalities)[idx_mods_active]
                
            for idx_mod in idx_mods_active:
                for lvl, router_weight in enumerate(router_weights[::-1]): # bottleneck should be logged as lvl-0
                    mod_name = self.trainer.datamodule.modalities[idx_mod]
                    log_key = f"router_{names}/{mod_name}-lvl-{lvl}"
                    log_value = router_weight[b,idx_mod].mean()
                    self.train_router_logs[log_key].append(log_value.item())
                    
        # apply BCE loss to encourage low weight for corrupt modalities (Vectorized)
        if self.bce_loss_scheduler is not None:
            bce_loss_weight = self.bce_loss_scheduler(self.current_epoch)
            self.log("bce_weight", bce_loss_weight, batch_size=images.shape[0], on_step=False, on_epoch=True)
            
            # The boolean mask is simply the corrupted modalities tensor
            loss_mask = modalities_corrupted
            
            if loss_mask.any():
                bce_losses = []
                for router_weight in router_weights:
                    # Average over spatial dims to get (B, C)
                    probs = router_weight.mean(dim=list(range(2, len(router_weight.shape))))
                    
                    # Apply mask to grab all corrupted probabilities across the entire batch at once
                    corrupt_probs = probs[loss_mask]
                    
                    with torch.autocast(device_type=corrupt_probs.device.type, enabled=False):
                        bce = torch.nn.functional.binary_cross_entropy(
                            corrupt_probs.float(), 
                            torch.zeros_like(corrupt_probs).float()
                        )
                        
                    bce_losses.append(bce)
                    
                bce_loss = torch.stack(bce_losses).mean()
                loss = bce_loss_weight * bce_loss + (1 - bce_loss_weight) * loss
                
                # Log batch_size as the total number of corrupted modalities we calculated loss on
                self.log("bce_loss", bce_loss, batch_size=loss_mask.sum().item(), on_step=False, on_epoch=True)
            
        return loss
    
    def on_train_epoch_end(self):
        for log_key, log_value_lst in self.train_router_logs.items():
            self.log(log_key + "_mean", np.nanmean(log_value_lst))
            self.log(log_key + "_std", np.nanstd(log_value_lst))
        self.train_router_logs.clear()
        
    def on_validation_epoch_start(self):
        self.val_router_logs = defaultdict(list)
        
    def validation_step(self, batch: dict):
        
        images, labels, modalities_missing = self._prepare_batch(batch) # (B,C,*), (B,K,*), (B,C)
        
        # do entire val step once per set of inference modalities
        for mod_lst, corrupt_lst in zip(self.inference_modalities, self.inference_corruption):
            
            # get idxs of input channels to zero out or keep for inference with mod_lst
            keep_idxs, drop_idxs = self._get_idxs_for_inference(mod_lst)
            
            # get idxs of input channels to apply corrution
            corrupt_idxs, good_idxs = self._get_idxs_for_inference(corrupt_lst)
                    
            assert not modalities_missing[:,keep_idxs].any(), f"Some of {mod_lst} are missing in a validation batch"
            assert keep_idxs != corrupt_idxs, f"We cannot corrupt all inference idxs"
            
            # zero out dropped modalities
            images_mods = images.clone()
            images_mods[:, drop_idxs] = torch.zeros_like(images_mods[:, drop_idxs])
            
            # corrupt required modalities
            images_mods = self.inference_corruption_fn(images_mods, corrupt_idxs)
            
            # forward on central patch
            pred, router_weights = self(images_mods, return_weights=True) # [(B,K,*)], [(B,num_encoders,1)]
            pred = pred[-1] # final unet layer output
            pred = F.one_hot(torch.argmax(pred, dim=1), num_classes=pred.shape[1]).movedim(-1,1)
            
            # metric calculation
            scores = dice_score(pred, labels, num_classes=pred.shape[1], include_background=False, average="none")
            mods_names = "_".join(mod_lst) + (("-X-" + "_".join(corrupt_lst)) if len(corrupt_idxs)!=0 else "")
            log_score = scores.nanmean()
            if not torch.isnan(log_score):
                self.log(f"{mods_names}/macro_avg", log_score, batch_size=scores.shape[0], on_step=False, on_epoch=True)
            for i, lab in enumerate(self.trainer.datamodule.labels):
                log_score = scores[:,i].nanmean()
                if not torch.isnan(log_score):
                    self.log(f"{mods_names}/{lab}", log_score, batch_size=scores.shape[0], on_step=False, on_epoch=True)

            # log router weights
            router_active = True #len(keep_idxs) > 1 # where there are at least 2 modalities present
            if router_active:
                for b in range(images.shape[0]):
                    for idx_mod in keep_idxs:
                        for lvl, router_weight in enumerate(router_weights[::-1]): # bottleneck should be logged as lvl-0
                            mod_name = self.trainer.datamodule.modalities[idx_mod]
                            log_key = f"{mods_names}/router_{mod_name}-lvl-{lvl}"
                            log_value = router_weight[b,idx_mod].mean()
                            self.val_router_logs[log_key].append(log_value.item())
                            
    def on_validation_epoch_end(self):
        for log_key, log_value_lst in self.val_router_logs.items():
            self.log(log_key + "_mean", np.nanmean(log_value_lst))
            self.log(log_key + "_std", np.nanstd(log_value_lst))
        self.val_router_logs.clear()
                            
    def on_test_epoch_start(self):
        if self.do_test_with_router_logging:
            print("\n\nINITIALISING TEST WITH ROUTER LOGGING\n")
            print("ALL test images and labels must have the shape expected by the router-based model!!!\n")
            print("Predicted segmentations will be saved along with the cropped input images and labels in pred_dir\n\n")
        self.jsons = defaultdict(dict)
        assert self.metric_dir, "metric_dir must be set for test"
        self._make_output_dirs(preds=True, metrics=True)
            
    def test_step(self, batch):
        
        if self.do_test_with_router_logging:
            self._test_step_with_router_logging(batch)
            return None
        
        # WARNING - if all images in a batch are not the same size, this step will fail
        # because the dataloader won't be able to collate the batch. Validation and training steps
        # use patch sampling, but test and predict steps use sliding window inference.
        # it is safest to enforce batch_size=1 in the test loader
        
        images, labels, modalities_missing = self._prepare_batch(batch) # (B,C,*), (B,K,*), (B,C)
        idxs = batch["idx"]
        
        # do entire test step once per set of inference modalities
        for mod_lst, corrupt_lst in zip(self.inference_modalities, self.inference_corruption):
            
            # get idxs of input channels to zero out or keep for inference with mod_lst
            keep_idxs, drop_idxs = self._get_idxs_for_inference(mod_lst)
            
            # get idxs of input channels to apply corrution
            corrupt_idxs, good_idxs = self._get_idxs_for_inference(corrupt_lst)
            
            mods_names = "_".join(mod_lst) + (("-X-" + "_".join(corrupt_lst)) if len(corrupt_idxs)!=0 else "")
            
            # for each element in the batch, print a warning if any inference modalities not available
            for b, idx in enumerate(idxs):
                missing = []
                for keep_idx in keep_idxs:
                    if modalities_missing[b, keep_idx]:
                        missing.append(self.trainer.datamodule.modalities[keep_idx])
                if len(missing) != 0:
                    print(f"Subject {idx} is missing {missing} while attempting to use {mod_lst} for inference")
            
            # zero out dropped modalities
            images_mods = images.clone()
            images_mods[:, drop_idxs] = torch.zeros_like(images_mods[:, drop_idxs])
            
            # corrupt required modalities
            images_mods = self.inference_corruption_fn(images_mods, corrupt_idxs)
            
            # foward using sliding window inference
            pred = sliding_window_inference(
                inputs=images_mods,
                roi_size=self.patch_size,
                sw_batch_size=images_mods.shape[0],
                predictor=self.sliding_window_predictor,
            )
            pred = F.one_hot(torch.argmax(pred, dim=1), num_classes=pred.shape[1]).movedim(-1,1)
            
            # metric calculation in label resolution (can be different to pred resolution if scalar_only is set in tio.Resample)
            scores = []
            for b in range(len(idxs)):
                pred_b = F.interpolate(pred[b:b+1].float(), size=labels.shape[2:], mode="nearest-exact")
                score = dice_score(pred_b, labels[b:b+1], num_classes=pred.shape[1], include_background=False, average="none")
                scores.append(score.nan_to_num(nan=1.0))
            scores = torch.cat(scores, dim=0)
            
            # per subject metric logging
            for b in range(len((idxs))):
                subject_scores_dict = {}
                for i in range(len((self.trainer.datamodule.labels))):
                    lab_name = self.trainer.datamodule.labels[i]
                    subject_scores_dict[lab_name] = scores[b,i].item()
                    
                    # save predictions if required
                    if self.pred_dir:
                        self._save_predictions(mod_lst, missing, batch, pred, b, i, mods_names)
                        
                subject_scores_dict["macro_avg"] = scores[b].nanmean().item()
                self.jsons[mods_names][idxs[b]] = subject_scores_dict
            
    def _test_step_with_router_logging(self, batch):
        
        # WARNING - if all images in a batch are not the same size, this step will fail
        # because the dataloader won't be able to collate the batch. Validation and training steps
        # use patch sampling, but test and predict steps use sliding window inference.
        # it is safest to enforce batch_size=1 in the test loader
        
        images, labels, modalities_missing = self._prepare_batch(batch) # (B,C,*), (B,K,*), (B,C)
        idxs = batch["idx"]
        
        # do entire test step once per set of inference modalities
        for mod_lst, corrupt_lst in zip(self.inference_modalities, self.inference_corruption):
            
            # get idxs of input channels to zero out or keep for inference with mod_lst
            keep_idxs, drop_idxs = self._get_idxs_for_inference(mod_lst)
            
            # get idxs of input channels to apply corrution
            corrupt_idxs, good_idxs = self._get_idxs_for_inference(corrupt_lst)
            
            mods_names = "_".join(mod_lst) + (("-X-" + "_".join(corrupt_lst)) if len(corrupt_idxs)!=0 else "")
            
            # for each element in the batch, print a warning if any inference modalities not available
            for b, idx in enumerate(idxs):
                missing = []
                for keep_idx in keep_idxs:
                    if modalities_missing[b, keep_idx]:
                        missing.append(self.trainer.datamodule.modalities[keep_idx])
                if len(missing) != 0:
                    print(f"Subject {idx} is missing {missing} while attempting to use {mod_lst} for inference")
            
            # zero out dropped modalities
            images_mods = images.clone()
            images_mods[:, drop_idxs] = torch.zeros_like(images_mods[:, drop_idxs])
            
            # corrupt required modalities
            images_mods = self.inference_corruption_fn(images_mods, corrupt_idxs)
            
            # forward on central patch
            pred, router_weights = self(images_mods, return_weights=True) # [(B,K,*)], [(B,num_encoders,1)]
            pred = pred[-1] # final unet layer output
            pred = F.one_hot(torch.argmax(pred, dim=1), num_classes=pred.shape[1]).movedim(-1,1)
            
            # metric calculation
            scores = dice_score(pred, labels, num_classes=pred.shape[1], include_background=False, average="none").nan_to_num(nan=1.0)
            
            # per subject metric logging and router weights
            router_active = True #len(keep_idxs) > 1 # where there are at least 2 modalities present
            for b in range(len((idxs))):
                do_save_images = True
                subject_scores_dict = {}
                for i in range(len((self.trainer.datamodule.labels))):
                    lab_name = self.trainer.datamodule.labels[i]
                    subject_scores_dict[lab_name] = scores[b,i].item()
                    
                    # save predictions, labels (cropped) and images (cropped) if required
                    if self.pred_dir:
                        self._save_predictions_with_router_logging(
                            mod_lst, missing, keep_idxs, batch, pred, labels, images_mods, b, i, mods_names, save_images=do_save_images
                            )
                        do_save_images = False
                        
                subject_scores_dict["macro_avg"] = scores[b].nanmean().item()
                
                # router weights
                if router_active:
                    for idx_mod in keep_idxs:
                        for lvl, router_weight in enumerate(router_weights[::-1]): # bottleneck should be logged as lvl-0
                            mod_name = self.trainer.datamodule.modalities[idx_mod]
                            log_key = f"router_{mod_name}-lvl-{lvl}"
                            log_value = router_weight[b,idx_mod].mean()
                            subject_scores_dict[log_key] = log_value.item()
                            
                self.jsons[mods_names][idxs[b]] = subject_scores_dict
                
    def _save_predictions_with_router_logging(
        self, 
        mod_lst: list[str], 
        missing: list[str],
        keep_idxs: list[int],
        batch: dict, 
        pred: torch.Tensor, 
        label: torch.Tensor,
        image: torch.Tensor,
        b: int,
        i: int,
        mods_names: str,
        save_images: bool
        ):
        ref_mod = list(set(mod_lst)-set(missing))[0] # first modality name which is used and present
        pred_data = pred[b,i+1].cpu().numpy() # (H,W,D) binary prediction for label i
        label_data = label[b,i+1].cpu().numpy() # (H,W,D) binary gt for label i
        affine = batch[ref_mod]["affine"][b] # affine 
        lab_name = self.trainer.datamodule.labels[i]
        # save prediction
        save_name = f"{batch['idx'][b]}_{mods_names}_{lab_name}.nii.gz"
        nii = nib.Nifti1Image(pred_data.astype("uint8"), affine=affine)
        nii.set_data_dtype("uint8")
        nib.save(nii, self.pred_dir / save_name)
        # save gt
        save_name = f"{batch['idx'][b]}_{mods_names}_{lab_name}_gt_cropped.nii.gz"
        nii = nib.Nifti1Image(label_data.astype("uint8"), affine=affine)
        nii.set_data_dtype("uint8")
        nib.save(nii, self.pred_dir / save_name)
        # save input images
        if save_images:
            for keep_idx in keep_idxs:
                mod_name = self.trainer.datamodule.modalities[keep_idx]
                image_data = image[b,keep_idx].cpu().numpy() # (H,W,D) image
                save_name = f"{batch['idx'][b]}_{mods_names}_{mod_name}_cropped.nii.gz"
                nii = nib.Nifti1Image(image_data, affine=affine)
                nib.save(nii, self.pred_dir / save_name)