import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import time


class TimeIteration(Callback):

    """Average time to process a batch"""

    def on_train_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        pl_module.train_dataloader_len = len(trainer.train_dataloader)

    def on_train_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        pl_module.epoch_start_time = time.perf_counter()

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        epoch_time = time.perf_counter() - pl_module.epoch_start_time
        mean_iter_time = epoch_time / pl_module.train_dataloader_len
        pl_module.log("mean_iter_time", mean_iter_time, on_step=False, on_epoch=True)