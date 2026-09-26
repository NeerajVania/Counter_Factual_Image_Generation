import numpy as np
import torch
import lightning.pytorch as pl
import torchvision.utils as vutils
import lpips

from .models import get_model
from .optimizers import get_optimizer, get_scheduler
from .metrics import get_metrics
from .criterion import get_criterion

class AutoencoderWrapper(pl.LightningModule):
    def __init__(self, cfg):
        super(AutoencoderWrapper, self).__init__()
        self.save_hyperparameters()

        self.config = cfg

        self.model = get_model(cfg)
        self.criterion = get_criterion(cfg)

        self.lpips_fn = lpips.LPIPS(net='vgg').requires_grad_(False)

        self.beta = cfg.training.beta
        self.lpips_weight = getattr(cfg.training, 'lpips_weight')
        self.image_resolution = cfg.data.image_resolution

        self.optimizer_name = cfg.optimizer.name
        self.optimizer_kwargs = cfg.optimizer.kwargs

        self.scheduler_name = cfg.scheduler.name
        self.scheduler_kwargs = cfg.scheduler.kwargs

        self.modes = ["train"]
        self.losses = {mode: [] for mode in self.modes}
        self.recon_losses = {mode: [] for mode in self.modes}
        self.kl_losses = {mode: [] for mode in self.modes}
        self.lpips_losses = {mode: [] for mode in self.modes}

        self.metrics = torch.nn.ModuleList([get_metrics(cfg) for mode in self.modes])
        self.mode_to_metrics = {
            mode: metric for mode, metric in zip(self.modes, self.metrics)
        }
        
        # Store sample images for visualization at the end of training
        self.sample_images = None

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
        self.losses[mode] = []
        self.recon_losses[mode] = []
        self.kl_losses[mode] = []
        metrics = self.mode_to_metrics[mode]
        metrics.reset()

    def training_step(self, batch, batch_idx):
        if self.sample_images is None:
            imgs, _, _ = batch
            self.sample_images = imgs[:2].clone()
            
        return self.__step(batch, batch_idx, "train")

    def __step(self, batch, batch_idx, mode):

        imgs, _, _ = batch 

        assert not torch.isnan(imgs).any(), "Input images contain NaN values."

        recon_imgs, mu, logvar = self.model(imgs)

        loss, loss_dict = self.criterion(recon_imgs, imgs, mu, logvar, self.lpips_fn, beta=self.beta, lpips_weight=self.lpips_weight)

        if torch.isnan(loss.cpu()):
            raise Exception("Loss is NaN.")
        else:
            self.losses[mode].append(loss.detach().cpu().item())
            self.recon_losses[mode].append(loss_dict['recon_loss'].detach().cpu().item())
            self.kl_losses[mode].append(loss_dict['kl_loss'].detach().cpu().item())
            self.lpips_losses[mode].append(loss_dict['lpips_loss'].detach().cpu().item())

        return loss

    def on_train_epoch_end(self) -> None:
        self.__log_metrics(mode="train")
        if self.current_epoch % 5 == 0:
            if self.sample_images is not None:
                self._log_reconstruction_samples()

    def __log_metrics(self, mode):
        metrics = self.mode_to_metrics[mode]
        metrics_out = metrics.get_out_dict()
        logs = {f"{mode} {key}": val for key, val in metrics_out.items()}
        
        logs[f"{mode} total_loss"] = np.mean(self.losses[mode])
        logs[f"{mode} recon_loss"] = np.mean(self.recon_losses[mode])
        logs[f"{mode} kl_loss"] = np.mean(self.kl_losses[mode])
        logs[f"{mode} lpips_loss"] = np.mean(self.lpips_losses[mode])
        logs["step"] = float(self.current_epoch)
        
        self.log_dict(logs, on_step=False, on_epoch=True, sync_dist=True)

    def denormalize(self, img, mean=[0.3704248070716858, 0.2282254546880722, 0.13915641605854034],
                     std=[0.23381589353084564, 0.1512117236852646, 0.09653093665838242]):
        mean = torch.tensor(mean).view(3, 1, 1).to(self.device)
        std = torch.tensor(std).view(3, 1, 1).to(self.device)
        return img * std + mean
    
    def _log_reconstruction_samples(self):
        self.model.eval()
        with torch.no_grad():
            sample_imgs = self.sample_images.to(self.device)
            
            recon_imgs, _, _ = self.model(sample_imgs)
            
            comparison_imgs = []
            for i in range(len(sample_imgs)):
                comparison_imgs.extend([self.denormalize(sample_imgs[i]), self.denormalize(recon_imgs[i])])

            grid = vutils.make_grid(
                comparison_imgs, 
                nrow=2,
                padding=2, 
                normalize=False,
                pad_value=1.0  # White padding
            )
            
            if self.logger is not None:
                self.logger.experiment.add_image(
                    'Reconstruction_Samples', 
                    grid, 
                    global_step=self.current_epoch
                )
        
        self.model.train()