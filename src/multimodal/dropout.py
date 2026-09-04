import torch
from multimodal.corruption import CorruptionTransform
    
    
class NoDropout:
    
    def __call__(self, images: torch.Tensor, modalities_missing: torch.Tensor, epoch: int) -> tuple[torch.Tensor]:
        dropped = torch.zeros_like(modalities_missing).to(torch.bool)
        return images, dropped
    

class RandomUniformMultiDropCorrupt:
    
    def __init__(
        self, 
        num_modalities: int, 
        num_skip_epochs_dropout: int = 0,
        num_skip_epochs_corruption: int = 0, 
        corruption_prob: float = 0.5):
        
        self.num_modalities = num_modalities
        self.num_skip_epochs_dropout = num_skip_epochs_dropout
        self.num_skip_epochs_corruption = num_skip_epochs_corruption
        self.corruption_prob = corruption_prob
        self.corruption_transform = CorruptionTransform()
    
    def __call__(self, images: torch.Tensor, modalities_missing: torch.Tensor, epoch: int, apply_dropout: bool = True) -> tuple[torch.Tensor]:
        
        select_count = torch.randint(0, self.num_modalities, (images.shape[0],1), device=modalities_missing.device)
        select_count = (select_count - modalities_missing.sum(dim=1, keepdim=True)).clip(0, torch.inf)
        selected = torch.rand_like(modalities_missing, dtype=float)
        selected[modalities_missing] = -1.0
        selected = selected.argsort(dim=-1, descending=True).argsort(dim=-1)
        selected = selected < select_count
        corrupted = torch.zeros_like(selected).to(torch.bool)
        dropped = torch.zeros_like(selected).to(torch.bool)
        
        # not batchable
        for b in range(selected.shape[0]):
            for i in range(selected.shape[1]):
                if selected[b,i]:
                    if torch.rand(1) <= self.corruption_prob and epoch >= self.num_skip_epochs_corruption:
                        images[b,i:i+1] = self.corruption_transform(images[b,i:i+1])
                        corrupted[b,i] = True
                    elif epoch >= self.num_skip_epochs_dropout:
                        if apply_dropout:
                            images[b,i] = 0.0
                        dropped[b,i] = True
                        
        return images, corrupted, dropped
    
    
