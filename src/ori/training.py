import numpy as np
import torch
import torchvision
from torchmetrics.image.fid import FrechetInceptionDistance
import lightning.pytorch as pl
import os
from .models import get_model, VAE
from .models.retfound_encoder import RETFoundDINOv2Encoder
from .optimizers import get_optimizer, get_scheduler
from .metrics import get_metrics
from .criterion import get_criterion
from .data import get_dataloader

def denormalize(img,  mean=[0.3704248070716858, 0.2282254546880722, 0.13915641605854034],
                      std=[0.23381589353084564, 0.1512117236852646, 0.09653093665838242]):
    mean = torch.tensor(mean, device=img.device).view(1, -1, 1, 1)
    std = torch.tensor(std, device=img.device).view(1, -1, 1, 1)
    return img * std + mean


class NoiseScheduler(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.timesteps = cfg.noise_scheduler.num_timesteps
        
        betas = torch.linspace(float(cfg.noise_scheduler.beta_start), float(cfg.noise_scheduler.beta_end), int(self.timesteps))
        alphas = 1.0 - betas
        alpha_hats = torch.cumprod(alphas, dim=0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_hats', alpha_hats)

    def q_sample(self, x_start, t, noise):
        sqrt_alpha_hat = self.alpha_hats[t] ** 0.5
        sqrt_one_minus_alpha_hat = (1 - self.alpha_hats[t]) ** 0.5
        return sqrt_alpha_hat[:, None, None, None] * x_start + sqrt_one_minus_alpha_hat[:, None, None, None] * noise

class LightningWrapper(pl.LightningModule):
        def __init__(self, cfg):
                super(LightningWrapper, self).__init__()
                self.save_hyperparameters()
                self.cfg = cfg

                self.val_loader = get_dataloader(cfg, mode="val")
                self.fid_metric = FrechetInceptionDistance(feature=2048, normalize=True)


                self.noise_scheduler = NoiseScheduler(cfg)
                self.T = self.noise_scheduler.timesteps

                self.model = get_model(cfg)
                self.v = []

                self.criterion = get_criterion(cfg)

                self.vae = self._load_pretrained_vae(cfg)

                if getattr(cfg.model, 'use_structure', getattr(cfg.model, 'structure', True)):
                        self.retfound_encoder = self._load_retfound_encoder(cfg)
                                
                self.latent_resolution = cfg.data.image_resolution // 4
                self.latent_channels = 4

                self.image_resolution = cfg.data.image_resolution

                self.optimizer_name = cfg.optimizer.name
                self.optimizer_kwargs = cfg.optimizer.kwargs

                self.scheduler_name = cfg.scheduler.name
                self.scheduler_kwargs = cfg.scheduler.kwargs

                self.modes = ["train", "val"]
                self.losses = {
                        mode: {"loss": []}
                        for mode in self.modes
                }
                self.metrics = torch.nn.ModuleList([get_metrics(cfg) for mode in self.modes])
                self.mode_to_metrics = {
                        mode: metric for mode, metric in zip(self.modes, self.metrics)
                }
                
        def _load_pretrained_vae(self, cfg):
                vae_checkpoint_path = cfg.vae.checkpoint
                
                if not os.path.exists(vae_checkpoint_path):
                        raise FileNotFoundError(f"VAE checkpoint not found at {vae_checkpoint_path}")

                vae = VAE(in_channels=3, latent_channels=4, ch_mult=[1, 2, 4])

                checkpoint = torch.load(vae_checkpoint_path, map_location="cpu")
                vae.load_state_dict(checkpoint)
                vae.eval()
                for param in vae.parameters():
                        param.requires_grad = False
                return vae
        
        def _load_retfound_encoder(self, cfg):
                """Load frozen RETFound-DINOv2 as structural encoder."""
                model_name = cfg.encoder.name
                checkpoint_path = cfg.encoder.checkpoint
                # treat empty string same as None
                if not checkpoint_path:
                        checkpoint_path = None
                encoder = RETFoundDINOv2Encoder(
                        model_name=model_name,
                        checkpoint_path=checkpoint_path,
                )
                # Verify frozen state
                assert all(not p.requires_grad for p in encoder.parameters()), \
                        "RETFound encoder parameters must be frozen"
                assert not encoder.training, \
                        "RETFound encoder must be in eval mode"
                return encoder


        def configure_optimizers(self):
                param_groups = []

                if getattr(self.cfg.model, 'learned_ordinal_input', getattr(self.cfg.model, 'learned', True)) and hasattr(self.model, "v"):
                        v_params = [self.model.v.weight]
                        other_params = [param for name, param in self.model.named_parameters() if name != "v.weight"]

                        param_groups.append({"params": other_params, "lr": self.optimizer_kwargs.get("lr")})
                        v_lr = self.optimizer_kwargs.get("v_lr", 1e-3)
                        param_groups.append({"params": v_params, "lr": v_lr})
                else:
                        other_params = list(self.model.parameters())
                        param_groups.append({"params": other_params, "lr": self.optimizer_kwargs.get("lr")})

                optimizer_class = get_optimizer(self.optimizer_name)
                optimizer_kwargs_clean = {k: v for k, v in self.optimizer_kwargs.items() if k not in ["lr", "v_lr"]}
                optimizer = optimizer_class(param_groups, **optimizer_kwargs_clean)

                if self.scheduler_name is None:
                        return optimizer

                scheduler_class = get_scheduler(self.scheduler_name)
                scheduler = scheduler_class(optimizer, **self.scheduler_kwargs)
                return [optimizer], [scheduler]

        def forward(self, x):
                y = self.model(x)
                return y

        def on_train_epoch_start(self) -> None:
                self.__reset_metrics("train")

        def on_validation_epoch_start(self) -> None:
                self.__reset_metrics("val")

        def __reset_metrics(self, mode: str) -> None:
                self.losses[mode] = {"loss": []}
                metrics = self.mode_to_metrics[mode]
                metrics.reset()

        def training_step(self, batch, batch_idx):
                return self.__step(batch, batch_idx, "train")
        
        def validation_step(self, batch, batch_idx):
                return self.__step(batch, batch_idx, "val")

        def __step(self, batch, batch_idx, mode):

                imgs, labels, iq = batch

                with torch.no_grad():
                        mu, log_var = self.vae.encoder(imgs)
                        latents = mu
                        if getattr(self.cfg.model, 'use_structure', getattr(self.cfg.model, 'structure', True)):
                                s = self.retfound_encoder(imgs)
                        else:
                                s = None

                if self.cfg.training.criterion == "ODloss":
                        t = torch.randint(0, self.noise_scheduler.timesteps, (1,), device=imgs.device).expand(imgs.size(0))
                else:
                        t = torch.randint(0, self.noise_scheduler.timesteps, (latents.size(0),), device=latents.device)

                noise = torch.randn_like(latents)
                x_t = self.noise_scheduler.q_sample(latents, t, noise)

                label_mask = torch.bernoulli(torch.zeros_like(labels) + 0.1)
                structure_mask = torch.bernoulli(torch.zeros_like(labels) + 0.1) if s is not None else None

                n, c, h, w = latents.shape
                assert c == self.latent_channels, f"Expected {self.latent_channels} latent channels, got {c}"
                assert h == w == self.latent_resolution, f"Expected {self.latent_resolution}x{self.latent_resolution} latent resolution, got {h}x{w}"


                noise_pred = self.model(x_t, labels, t, label_mask, s, structure_mask)
                
                if self.model.learn_sigma:      
                        noise_pred, var_pred = noise_pred.chunk(2, dim=1)

                if self.cfg.training.criterion == "ODloss":
                        loss = self.criterion(noise_pred, noise, labels, t, ordinal=True)
                else:
                        loss = self.criterion(noise_pred, noise)
                if getattr(self.cfg.model, 'learned_ordinal_input', getattr(self.cfg.model, 'learned', True)):
                        loss += 0.0001 * (1/((self.model.v.weight)**2 + 1e-8)).mean()

                if torch.isnan(loss.cpu()):
                        raise Exception("loss is Nan.")
                else:
                        self.losses[mode]["loss"].append(loss.detach().cpu().item())
                return loss

        def on_train_batch_end(self, outputs, batch, batch_idx):
                if getattr(self.cfg.model, 'learned_ordinal_input', getattr(self.cfg.model, 'learned', True)):
                        self.v.append(self.model.v.weight.detach().cpu().clone())

        def on_train_epoch_end(self) -> None:
                self.__log_metrics(mode="train")
                if getattr(self.cfg.model, 'learned_ordinal_input', getattr(self.cfg.model, 'learned', True)):
                        v_tensor = torch.stack(self.v, dim=0)
                        v_path = os.path.join(self.logger.log_dir, f"v.pt")
                        torch.save(v_tensor.cpu(), v_path)
                #if self.current_epoch > 0:
                #        torch.save(self.model.state_dict(), os.path.join(self.logger.log_dir, "checkpoints", f"last.pt"))
                #self.evaluate_fid_and_save_best()
                
        def on_validation_epoch_end(self) -> None:
            self.__log_metrics(mode="val")

        def __log_metrics(self, mode):
                metrics = self.mode_to_metrics[mode]
                metrics_out = metrics.get_out_dict()
                logs = {f"{mode} {key}": val for key, val in metrics_out.items()}


                logs[f"{mode} loss"] = np.mean(self.losses[mode]["loss"]) if len(self.losses[mode]["loss"]) > 0 else 0.0

                logs["step"] = float(self.current_epoch)
                self.log_dict(logs, on_step=False, on_epoch=True, sync_dist=True)

        @torch.no_grad()
        def generate_samples(self, num_samples, device=None, ddim_steps=50, eta=0.0):
                x_i = torch.randn((num_samples, 4, 64, 64), device=device)

                labels = torch.zeros((num_samples,), dtype=torch.long).to(device)

                context_mask = torch.ones_like(torch.arange(num_samples), dtype=torch.float).to(device)

                structure = torch.zeros(num_samples, 1024, device=device)
                structure_mask = torch.ones_like(torch.arange(num_samples), dtype=torch.float).to(device)

                T = self.noise_scheduler.timesteps-1

                with torch.no_grad():
                        ddim_steps = 50
                        ddim_timesteps = torch.linspace(0, T, ddim_steps, device=device).long()
                        for i in reversed(range(0, ddim_steps)):
                                t = ddim_timesteps[i]
                                t_prev = ddim_timesteps[i - 1] if i > 0 else 0

                                t_tensor = torch.full((len(labels),), t, device=device)

                                predicted_noise = self.model(x_i, labels, t_tensor, context_mask, structure, structure_mask = structure_mask)

                                predicted_noise = predicted_noise[:, :4, :, :]

                                alpha_hat_t = self.noise_scheduler.alpha_hats[t]
                                alpha_hat_t_prev = self.noise_scheduler.alpha_hats[t_prev] if t_prev > 0 else torch.tensor(1.0).to(device)

                                x_0_hat = (x_i - (1 - alpha_hat_t).sqrt() * predicted_noise) / alpha_hat_t.sqrt()
                                x_i = alpha_hat_t_prev.sqrt() * x_0_hat + (1 - alpha_hat_t_prev).sqrt() * predicted_noise

                with torch.no_grad():
                        x_dec = self.vae.decoder(x_i)
                return x_dec
        
        @torch.no_grad()
        def evaluate_fid_and_save_best(self):
                #num_samples = self.cfg.data.n_val
                num_samples = 0
                samples = self.generate_samples(num_samples=num_samples, device=self.device)

                samples_den = denormalize(samples)

                # save image
                samples_den = denormalize(samples).clamp(0, 1)
                val_imgs, _, _ = next(iter(self.val_loader))
                val_imgs = denormalize(val_imgs)
                val_imgs = val_imgs.to(self.device).clamp(0, 1)

                self.fid_metric.reset()
                self.fid_metric.update(val_imgs, real=True)
                self.fid_metric.update(samples_den, real=False)

                fid_ = self.fid_metric.compute().item()
                self.log("val fid", fid_, prog_bar=True, on_epoch=True, sync_dist=True)

                grid = torchvision.utils.make_grid(samples_den[:10], nrow=5, normalize=False)
                self.logger.experiment.add_image("sample_grid", grid, self.current_epoch)

                del samples, samples_den, val_imgs, grid, fid_
                torch.cuda.empty_cache()

