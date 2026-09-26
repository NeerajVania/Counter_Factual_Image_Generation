import numpy as np
import torch
import lightning.pytorch as pl
import torchvision.utils as vutils

from .models import get_model
from .optimizers import get_optimizer, get_scheduler
from .metrics import get_metrics
from .criterion import get_criterion

class EncoderWrapper(pl.LightningModule):
    def __init__(self, cfg):
        super(EncoderWrapper, self).__init__()
        self.save_hyperparameters()

        self.config = cfg

        self.model = get_model(cfg)
        self.criterion = get_criterion(cfg)

        self.image_resolution = cfg.data.image_resolution

        self.optimizer_name = cfg.optimizer.name
        self.optimizer_kwargs = cfg.optimizer.kwargs

        self.scheduler_name = cfg.scheduler.name
        self.scheduler_kwargs = cfg.scheduler.kwargs

        self.modes = ["train"]
        self.loss = {mode: [] for mode in self.modes}

        self.metrics = torch.nn.ModuleList([get_metrics(cfg) for mode in self.modes])
        self.mode_to_metrics = {
            mode: metric for mode, metric in zip(self.modes, self.metrics)
        }

    def configure_optimizers(self):
        optimizer = get_optimizer(self.optimizer_name)(self.parameters(), **self.optimizer_kwargs)
        
        if self.scheduler_name is None:
            return optimizer
        else:
            scheduler = get_scheduler(self.scheduler_name)(optimizer, **self.scheduler_kwargs)
            return [optimizer], [scheduler]

    def forward(self, x):
        return self.model(x)

    def on_train_epoch_start(self) -> None:
        self.__reset_metrics("train")

    def __reset_metrics(self, mode: str) -> None:
        self.loss[mode] = []
        metrics = self.mode_to_metrics[mode]
        metrics.reset()

    def training_step(self, batch, batch_idx):
        return self.__step(batch, batch_idx, "train")

    def __step(self, batch, batch_idx, mode):

        imgs1, imgs2, labels = batch 

        z_i = self.model(imgs1)
        z_j = self.model(imgs2)

        loss = self.criterion(z_i, z_j)

        if torch.isnan(loss.cpu()):
          raise Exception("Loss is NaN.")
        else:
          self.loss[mode].append(loss.detach().cpu().item())
        return loss

    def on_train_epoch_end(self) -> None:
        self.__log_metrics(mode="train")

    def __log_metrics(self, mode):
        metrics = self.mode_to_metrics[mode]
        metrics_out = metrics.get_out_dict()
        logs = {f"{mode} {key}": val for key, val in metrics_out.items()}
        
        logs[f"{mode} total_loss"] = np.mean(self.loss[mode])
        logs["step"] = float(self.current_epoch)
        
        self.log_dict(logs, on_step=False, on_epoch=True, sync_dist=True)
