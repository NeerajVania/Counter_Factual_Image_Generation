import numpy as np
import torch
import torch.nn.functional as F
import lightning.pytorch as pl
from .models import get_model
from .optimizers import get_optimizer, get_scheduler
from .metrics import get_metrics
from .criterion import get_criterion


class LightningWrapper(pl.LightningModule):
    def __init__(self, cfg):
        super(LightningWrapper, self).__init__()
        self.save_hyperparameters()

        self.cfg = cfg

        self.max_epochs = cfg.max_epochs

        self.model = get_model(cfg)
        self.criterion = get_criterion(cfg)

        self.image_resolution = cfg.data.image_resolution

        self.optimizer_name = cfg.optimizer.name
        self.optimizer_kwargs = cfg.optimizer.kwargs

        self.scheduler_name = cfg.scheduler.name
        self.scheduler_kwargs = cfg.scheduler.kwargs

        self.modes = ["train", "val"]

        self.losses = {mode: [] for mode in self.modes}
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
        y = self.model(x)
        return y

    def on_train_epoch_start(self) -> None:
        self.__reset_metrics("train")

    def on_validation_epoch_start(self) -> None:
        self.__reset_metrics("val")

    def __reset_metrics(self, mode: str) -> None:
        self.losses[mode] = []
        metrics = self.mode_to_metrics[mode]
        metrics.reset()

    def training_step(self, batch, batch_idx):
        return self.__step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.__step(batch, batch_idx, "val")

    def __step(self, batch, batch_idx, mode):  # mode is one of ['train','test','val']

        imgs, labels, _ = batch

        labels = labels.float()
        
        n, c, h, w = imgs.shape
        assert h == w == self.image_resolution

        logits = self.model(imgs)
        if self.cfg.training.criterion == "coral":
            num_classes = logits.shape[1] + 1
            levels = (labels.unsqueeze(1) > torch.arange(num_classes - 1).to(labels.device)).float()
            loss = self.criterion(logits.squeeze(), levels, reduction=None)
            pred_class = torch.sum(torch.sigmoid(logits) > 0.5, dim=1)
        else:
            loss = self.criterion(logits.squeeze(), labels, reduction="none")
            pred_class = torch.round(logits.squeeze())
        
        loss = torch.mean(loss)

        with torch.no_grad():
            metrics = self.mode_to_metrics[mode]
            metrics.update(imgs, labels.long(), pred_class)

        if torch.isnan(loss.cpu()):
            raise Exception("loss is Nan.")
        else:
            self.losses[mode].append(loss.detach().cpu().item())

        return loss

    def on_train_epoch_end(self) -> None:
        self.__log_metrics(mode="train")

    def on_validation_epoch_end(self) -> None:
        self.__log_metrics(mode="val")

    def __log_metrics(self, mode):
        metrics = self.mode_to_metrics[mode]
        metrics_out = metrics.get_out_dict()
        logs = {f"{mode} {key}": val for key, val in metrics_out.items()}
        logs[f"{mode} loss"] = np.mean(self.losses[mode])
        logs["step"] = float(self.global_step)
        
        self.log_dict(logs, on_step=False, on_epoch=True, sync_dist=True)