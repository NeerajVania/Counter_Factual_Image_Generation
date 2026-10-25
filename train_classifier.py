import os

import lightning.pytorch as pl
from omegaconf import OmegaConf

from ori.config import get_config
from ori.data import get_dataloader
from ori.classifier_training import LightningWrapper
from ori.utils import TorchScriptModelCheckpoint

PATH_TO_DEFAULT_CFG = "configs/classifier.yaml"


def main(cfg):
    cfg = OmegaConf.create(cfg)
    module = LightningWrapper(cfg)
    if cfg.seed is not None:
        pl.seed_everything(cfg.seed)

    try:
        os.mkdir(cfg.training.out_dir)
    except:
        pass

    trainer = pl.Trainer(
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        strategy=(
            pl.strategies.DDPStrategy(find_unused_parameters=True)
            if len(cfg.devices) > 1
            else "auto"
        ),
        max_epochs=cfg.max_epochs,
        logger=pl.loggers.TensorBoardLogger(
            save_dir=cfg.training.out_dir, default_hp_metric=False,
        ),
        callbacks=[
            TorchScriptModelCheckpoint(
                save_top_k=cfg.training.checkpoints.save_top_k,
                monitor=cfg.training.checkpoints.monitor,
                mode=cfg.training.checkpoints.mode,
                filename=cfg.training.checkpoints.filename,
            ),
        ],
        default_root_dir=cfg.training.out_dir,
        log_every_n_steps=1,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=0,
        precision=cfg.training.precision,
        enable_progress_bar=False 
    )

    trainer.fit(
        module,
        train_dataloaders=get_dataloader(cfg, mode="train"),
        val_dataloaders=get_dataloader(cfg, mode="val"),
    )


if __name__ == "__main__":
    cfg = get_config(PATH_TO_DEFAULT_CFG)
    main(cfg)