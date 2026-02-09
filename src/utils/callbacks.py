import pytorch_lightning as pl
import pandas as pd
from typing import Any
from pathlib import Path


class ValidationLossLogger(pl.Callback):
    def __init__(self, output_path: str = "val_batch_losses.csv"):
        super().__init__()
        self.output_path = Path(output_path)
        self.batch_losses = []

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        # Assuming loss is returned as outputs['loss'] or just outputs if a tensor
        loss = outputs["loss"] if isinstance(outputs, dict) and "loss" in outputs else outputs
        loss_value = loss.detach().cpu().item()
        self.batch_losses.append({
            "epoch": trainer.current_epoch,
            "batch_idx": batch_idx,
            "val_loss": loss_value,
        })

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        df = pd.DataFrame(self.batch_losses)
        df.to_csv(self.output_path, index=False)
        # Clear for next epoch
        self.batch_losses = []
