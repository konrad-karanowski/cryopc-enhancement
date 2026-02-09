import glob
import os
import shutil
from typing import Any, Optional

import torch
from torch import nn
from torch.nn import functional as F
from lightning.pytorch import LightningModule
import torchmetrics
from lightning.pytorch.cli import OptimizerCallable, LRSchedulerCallable
import glob
import hydra


class CryoPCLitModule(LightningModule):

    def __init__(
        self,
        net: nn.Module = None,
        train_loss: nn.Module = None,
        condition_on_point_clouds: bool = False,
        use_mask_loss: bool = False,
        *args,
        **kwargs
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = hydra.utils.instantiate(self.hparams.net)

        # loss function
        self.train_loss = hydra.utils.instantiate(self.hparams.train_loss)
        self.val_loss = nn.MSELoss()

        self.example_input_array = torch.zeros(1, 1, 48, 48, 48)

        # All conditioning mechanisms
        self.condition_on_point_clouds = condition_on_point_clouds

        self.use_mask_loss = use_mask_loss
    
    def forward(self, x: torch.Tensor, struc_feat: torch.Tensor = None):
        return self.net(x, struc_feat=struc_feat)

    def _prepare_inputs(self, x: torch.Tensor, batch: Any) -> torch.Tensor:
        """
        Function that allows taking any kwargs we want.
        """
        inputs = {'x': x}
        if self.condition_on_point_clouds and 'c' in batch.keys():
            inputs['struc_feat'] = batch['c']
        return inputs


    def training_step(self, batch: Any, batch_idx: int):
        x, y, filename = batch['X'], batch['y'], batch['filename']

        inputs = self._prepare_inputs(x, batch)
        preds = self.forward(**inputs) 


        if self.hparams.use_mask_loss:
            mask = (preds == 0) & (y == 0)
            masked_preds = preds[~mask]
            masked_y = y[~mask]

            loss = self.train_loss(masked_preds, masked_y)
            #loss = self.train_loss(torch.flatten(masked_preds), torch.flatten(masked_y))
        else:
            loss = self.train_loss(preds, y)
            #loss = self.train_loss(torch.flatten(preds), torch.flatten(y))

        # Calculate and log other metrics if needed
        self.log(
            "train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(batch), sync_dist=True
        )

        return {"loss": loss}

    def validation_step(self, batch: Any, batch_idx: int):
        x, y, filename = batch['X'], batch['y'], batch['filename']

        inputs = self._prepare_inputs(x, batch)
        preds = self.forward(**inputs) 

        mask = (x == 0) & (y == 0)

        loss = self.val_loss(torch.flatten(preds), torch.flatten(y))
        loss_x1 = self.val_loss(torch.flatten(x), torch.flatten(y))

        loss_masked = self.val_loss(torch.flatten(preds[~mask]), torch.flatten(y[~mask]))
        loss_x1_masked = self.val_loss(torch.flatten(x[~mask]), torch.flatten(y[~mask]))


        # Calculate and log other metrics if needed
        self.log_dict({
            'val/loss': loss,
            'val/loss_masked': loss_masked,
            'val/loss_x1': loss_x1,
            'val/loss_x1_masked': loss_x1_masked
        }, on_step=False, on_epoch=True, prog_bar=True, batch_size=len(batch), sync_dist=True)
        return {"loss": loss}

    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, files = batch['X'], batch['file']

        inputs = self._prepare_inputs(x, batch)
        preds = self.forward(**inputs) 
        
        return preds, files

    def configure_optimizers(self):
        """Configure optimizer and lr scheduler.

        Returns:
            Union[Sequence[Optimizer], Tuple[Sequence[Optimizer], Sequence[Any]]]:
                Optimizer or optimizer and lr scheduler.
        """
        params = self.net.parameters()
        optimizer = hydra.utils.instantiate(
            self.hparams.optimizer, params=params, _convert_="partial"
        )
        if "lr_scheduler" not in self.hparams:
            return [optimizer]
        scheduler = hydra.utils.instantiate(
            self.hparams.lr_scheduler, optimizer=optimizer, _convert_="partial"
        )
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": self.hparams.monitor_metric,
            }
        return [optimizer], [scheduler]
