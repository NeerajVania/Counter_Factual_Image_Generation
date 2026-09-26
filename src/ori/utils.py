import os
from os import path as osp

import torch
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar
from lightning.pytorch.utilities.rank_zero import rank_zero_only

def denormalize(img, mean=[0.3704248070716858, 0.2282254546880722, 0.13915641605854034],
                    std=[0.23381589353084564, 0.1512117236852646, 0.09653093665838242]):
    mean = torch.tensor(mean).view(1, 1, 3)
    std = torch.tensor(std).view(1, 1, 3)
    return torch.clip(img * std + mean, 0, 1)

class LitProgressBar(RichProgressBar):
    def __init__(self):
        super().__init__()
        self.enable = True

    def get_metrics(self, trainer, pl_module):
        # don't show the version number
        items = super().get_metrics(trainer, pl_module)
        items.pop("v_num", None)
        return items


def save_torchscript(module: torch.nn.Module, filepath: str) -> None:
    torch.save(module.state_dict(), filepath)


class TorchScriptModelCheckpoint(ModelCheckpoint):
    r"""Saves the model as an additional standalone pt file whenever a checkpoint is created."""

    def __init__(
        self,
        dirpath=None,
        filename=None,
        monitor=None,
        verbose=False,
        save_last=None,
        save_top_k=1,
        save_weights_only=False,
        mode="min",
        auto_insert_metric_name=True,
        every_n_train_steps=None,
        train_time_interval=None,
        every_n_epochs=None,
        save_on_train_epoch_end=None,
        enable_version_counter=True,
    ):
        super(TorchScriptModelCheckpoint, self).__init__(
            dirpath=dirpath,
            filename=filename,
            monitor=monitor,
            verbose=verbose,
            save_last=save_last,
            save_top_k=save_top_k,
            save_weights_only=save_weights_only,
            mode=mode,
            auto_insert_metric_name=auto_insert_metric_name,
            every_n_train_steps=every_n_train_steps,
            train_time_interval=train_time_interval,
            every_n_epochs=every_n_epochs,
            save_on_train_epoch_end=save_on_train_epoch_end,
            enable_version_counter=enable_version_counter,
        )
        self.last_kth_best_model_path = ""

    @rank_zero_only
    def on_save_checkpoint(self, trainer, pl_module, checkpoint: dict) -> dict:
        """
        Convert model to TorchScript and save it as a .pt file
        after training ends (or at any checkpoint saving step).
        """
        if not osp.exists(self.dirpath):
            os.mkdir(self.dirpath)
        callback_metrics = {
            key: int(val) if key == "step" else val
            for key, val in trainer.callback_metrics.items()
        }
        callback_metrics["epoch"] = trainer.current_epoch
        filename, _ = os.path.splitext(
            self.format_checkpoint_name(callback_metrics, self.filename)
        )
        torchscript_model_path = f"{filename}.pt"

        # Save the model
        save_torchscript(pl_module.model, torchscript_model_path)

        # Optionally, you can include the TorchScript model path in the checkpoint (if you want)
        checkpoint["torchscript_model_path"] = torchscript_model_path

        # Remove (k+1)th model
        if self.last_kth_best_model_path != "":
            filename_to_delete, _ = os.path.splitext(self.last_kth_best_model_path)
            os.remove(f"{filename_to_delete}.pt")

        # Update deadthlist
        self.last_kth_best_model_path = self.kth_best_model_path

        return checkpoint