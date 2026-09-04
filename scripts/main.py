import sys
from pathlib import Path
from pytorch_lightning.cli import LightningCLI
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
import warnings


# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# OmegaConf resolvers
def length(lst: list) -> int:
    return len(lst)
OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("length", length)

# Set torch precision options
torch.set_float32_matmul_precision('medium')

# Ignore specific pytorch scheduler deprecation warning cluttering console
warnings.filterwarnings("ignore", message="The epoch parameter in `scheduler.step()` was not necessary")
    

if __name__ == "__main__":
    cli = LightningCLI(
        pl.LightningModule,
        pl.LightningDataModule,
        subclass_mode_model=True,
        subclass_mode_data=True,
        parser_kwargs={"parser_mode": "omegaconf"},
    )