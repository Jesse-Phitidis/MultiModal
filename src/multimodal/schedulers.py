from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR


class WarmupCosineLR(SequentialLR):
    def __init__(self, optimizer, warmup_epochs: int, max_epochs: int, start_factor: float = 0.1, eta_min: float = 1e-6, last_epoch: int = -1):
        
        warmup = LinearLR(
            optimizer, 
            start_factor=start_factor, 
            total_iters=warmup_epochs
        )
        
        cosine = CosineAnnealingLR(
            optimizer, 
            T_max=(max_epochs - warmup_epochs), 
            eta_min=eta_min
        )
        
        super().__init__(
            optimizer, 
            schedulers=[warmup, cosine], 
            milestones=[warmup_epochs], 
            last_epoch=last_epoch
        )
        
        
class TruncatedPolynomialScheduler:
    
    def __init__(
        self, 
        start_epoch: int, 
        end_epoch: int, 
        weight_init: float, 
        weight_end: float, 
        weight_before: float, 
        weight_after: float,
        power: float
        ):
        
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.weight_init = weight_init
        self.weight_end = weight_end
        self.weight_before = weight_before
        self.weight_after = weight_after
        self.power = power
        
    def __call__(self, current_epoch: int) -> float:
        
        if current_epoch < self.start_epoch:
            return self.weight_before
        
        if current_epoch > self.end_epoch:
            return self.weight_after
        
        t1 = (self.weight_init - self.weight_end)
        t2 = (1.0 - (current_epoch - self.start_epoch)/(self.end_epoch - self.start_epoch))**self.power
        t3 = self.weight_end
        weight = t1*t2 + t3
        
        return weight
    
    
class TruncatedSectionalPolynomialScheduler(TruncatedPolynomialScheduler):
    
    def __init__(
        self, 
        schedule_start_epoch: float,
        schedule_end_epoch: float,
        *args,
        **kwargs
        ):
        
        super().__init__(*args, **kwargs)
        self.schedule_start_epoch = schedule_start_epoch
        self.schedule_end_epoch = schedule_end_epoch

    def __call__(self, current_epoch: int) -> float:
        
        if current_epoch < self.schedule_start_epoch:
            return 0.0
        
        if current_epoch > self.schedule_end_epoch:
            return 0.0
        
        return super().__call__(current_epoch)
    
    
class ConstantScheduler:
    
    def __init__(self, weight: float, epoch_start: int, epoch_end: int):
        self.weight = weight
        self.epoch_start = epoch_start
        self.epoch_end = epoch_end

    def __call__(self, current_epoch: int) -> float:
        if self.epoch_start <= current_epoch <= self.epoch_end:
            return self.weight
        else:
            return 0.0   