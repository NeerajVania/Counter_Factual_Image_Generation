"""
Rectified Flow training wrapper for the ordinal diffusion pipeline.

Replaces the DDPM noise-prediction objective with a velocity-matching
objective using linear interpolation between clean data and Gaussian noise.

Training objective (Rectified Flow):
    z_tau = (1 - tau) * z_data + tau * epsilon        (interpolation)
    v_target = epsilon - z_data                        (target velocity)
    v_pred = DiT(z_tau, tau, c_y, s)                   (model prediction)
    L_RF = mean((v_pred - v_target)^2)                 (velocity MSE loss)

Time convention:
    tau = 0: clean data latent (z_data)
    tau = 1: Gaussian noise (epsilon)
    t_scaled = tau * timestep_scale  (for the sinusoidal timestep embedder)

Preserved from original pipeline:
    - Frozen VAE encoder/decoder
    - Frozen RETFound-DINOv2 structural encoder
    - DiT Transformer backbone (with learn_sigma=False for velocity output)
    - Ordinal severity conditioning (equidistant or learned)
    - Classifier-free guidance dropout for labels and structure
    - Dataset, preprocessing, optimizer, scheduler, checkpointing, logging
"""

import numpy as np
import os

import torch
import torch.nn.functional as F
import torchvision
from torchmetrics.image.fid import FrechetInceptionDistance
import lightning.pytorch as pl

from .models import get_model, VAE
from .models.retfound_encoder import RETFoundDINOv2Encoder
from .optimizers import get_optimizer, get_scheduler
from .metrics import get_metrics
from .criterion import get_criterion
from .data import get_dataloader
from .rf_sampler import RectifiedFlowSampler


def denormalize(img, mean=[0.3704248070716858, 0.2282254546880722, 0.13915641605854034],
                      std=[0.23381589353084564, 0.1512117236852646, 0.09653093665838242]):
    mean = torch.tensor(mean, device=img.device).view(1, -1, 1, 1)
    std = torch.tensor(std, device=img.device).view(1, -1, 1, 1)
    return img * std + mean


class RFLightningWrapper(pl.LightningModule):
    """
    Lightning training wrapper for Rectified Flow with DiT backbone.

    Replaces the DDPM NoiseScheduler and noise-prediction loss with:
    - Linear interpolation between clean latents and Gaussian noise
    - Velocity-matching MSE loss
    - Euler-based RF sampler for validation/inference
    """

    def __init__(self, cfg):
        super(RFLightningWrapper, self).__init__()
        self.save_hyperparameters()
        self.cfg = cfg

        self.val_loader = get_dataloader(cfg, mode="val")
        self.fid_metric = FrechetInceptionDistance(feature=2048, normalize=True)

        # RF time parameters
        rf_cfg = getattr(cfg, 'rectified_flow', None)
        if rf_cfg is not None:
            self.timestep_scale = float(rf_cfg.timestep_scale)
            self.num_inference_steps = int(rf_cfg.num_inference_steps)
            self.cfg_scale = float(rf_cfg.cfg_scale)
            self.cfg_dropout_prob = float(rf_cfg.cfg_dropout_prob)
        else:
            # Fallback defaults
            self.timestep_scale = 999.0
            self.num_inference_steps = 50
            self.cfg_scale = 4.0
            self.cfg_dropout_prob = 0.1

        # DiT model (with learn_sigma=False for velocity prediction)
        self.model = get_model(cfg)

        # Verify DiT output channels match velocity target (4 channels, not 8)
        assert not self.model.learn_sigma, (
            "Rectified Flow requires learn_sigma=False (4-channel velocity output). "
            "Set model.learn_sigma: False in the config."
        )

        self.v = []  # for tracking learned ordinal embeddings

        self.criterion = get_criterion(cfg)

        # Frozen VAE
        self.vae = self._load_pretrained_vae(cfg)

        # Frozen RETFound structural encoder
        if cfg.model.use_structure:
            self.retfound_encoder = self._load_retfound_encoder(cfg)

        # Latent dimensions
        self.latent_resolution = cfg.data.image_resolution // 4
        self.latent_channels = 4
        self.image_resolution = cfg.data.image_resolution

        # Optimizer config
        self.optimizer_name = cfg.optimizer.name
        self.optimizer_kwargs = cfg.optimizer.kwargs

        self.scheduler_name = cfg.scheduler.name
        self.scheduler_kwargs = cfg.scheduler.kwargs

        # Metrics tracking
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
        """Load and freeze the pretrained VAE."""
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
        """Configure optimizer for trainable DiT parameters only."""
        param_groups = []

        if self.cfg.model.learned_ordinal_input and hasattr(self.model, "v"):
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
        """
        Rectified Flow training step.

        1. Encode image to latent via frozen VAE
        2. Extract structure embedding via frozen RETFound
        3. Sample continuous time tau ~ U[0,1]
        4. Interpolate: z_tau = (1-tau)*z_data + tau*epsilon
        5. Compute target velocity: v_target = epsilon - z_data
        6. Predict velocity: v_pred = DiT(z_tau, tau, c_y, s)
        7. Loss: MSE(v_pred, v_target)
        """
        imgs, labels, iq = batch

        # --- Frozen feature extraction (no gradients) ---
        with torch.no_grad():
            mu, log_var = self.vae.encoder(imgs)
            latents = mu  # z_data: deterministic encoding (mean only)
            if self.cfg.model.use_structure:
                s = self.retfound_encoder(imgs)
            else:
                s = None

        # --- Verify latent dimensions ---
        n, c, h, w = latents.shape
        assert c == self.latent_channels, f"Expected {self.latent_channels} latent channels, got {c}"
        assert h == w == self.latent_resolution, f"Expected {self.latent_resolution}x{self.latent_resolution} latent resolution, got {h}x{w}"

        # --- RF interpolation ---
        # Sample independent continuous time for each example
        tau = torch.rand(n, device=latents.device)  # [B] in [0, 1]
        epsilon = torch.randn_like(latents)          # [B, 4, 64, 64]

        # Expand tau for broadcasting: [B] -> [B, 1, 1, 1]
        tau_expanded = tau[:, None, None, None]

        # z_tau = (1 - tau) * z_data + tau * epsilon
        z_tau = (1.0 - tau_expanded) * latents + tau_expanded * epsilon

        # Target velocity: v_target = epsilon - z_data
        v_target = epsilon - latents

        # Map continuous tau to timestep embedder scale
        t_scaled = tau * self.timestep_scale  # [B], values in [0, timestep_scale]

        # --- Classifier-free guidance dropout ---
        label_mask = torch.bernoulli(
            torch.zeros_like(labels, dtype=torch.float32) + self.cfg_dropout_prob
        )
        if s is not None:
            structure_mask = torch.bernoulli(
                torch.zeros_like(labels, dtype=torch.float32) + self.cfg_dropout_prob
            )
        else:
            structure_mask = None

        # --- DiT forward: predict velocity ---
        v_pred = self.model(z_tau, labels, t_scaled, label_mask, s, structure_mask)

        # --- Velocity-matching loss ---
        loss = self.criterion(v_pred, v_target)

        # Learned ordinal embedding regularization (preserved from original)
        if self.cfg.model.learned_ordinal_input:
            loss = loss + 0.0001 * (1.0 / ((self.model.v.weight) ** 2 + 1e-8)).mean()

        # --- Sanity check ---
        if torch.isnan(loss.cpu()):
            raise Exception("loss is NaN.")
        else:
            self.losses[mode]["loss"].append(loss.detach().cpu().item())
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.cfg.model.learned_ordinal_input:
            self.v.append(self.model.v.weight.detach().cpu().clone())

    def on_train_epoch_end(self) -> None:
        self.__log_metrics(mode="train")
        if self.cfg.model.learned_ordinal_input:
            v_tensor = torch.stack(self.v, dim=0)
            v_path = os.path.join(self.logger.log_dir, f"v.pt")
            torch.save(v_tensor.cpu(), v_path)

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
    def generate_samples(self, num_samples, device=None, num_steps=None, cfg_scale=None):
        """
        Generate unconditional samples using the RF sampler.

        Uses zero structure embedding with structure masked out (unconditional).
        For counterfactual generation with real structure, use rf_sampler.generate_counterfactual().

        Args:
            num_samples: Number of images to generate.
            device: torch device.
            num_steps: Override inference steps (default from config).
            cfg_scale: Override guidance scale (default from config).

        Returns:
            images: [num_samples, 3, 256, 256] generated images.
        """
        if num_steps is None:
            num_steps = self.num_inference_steps
        if cfg_scale is None:
            cfg_scale = self.cfg_scale

        labels = torch.zeros((num_samples,), dtype=torch.float, device=device)

        sampler = RectifiedFlowSampler(
            model=self.model,
            vae=self.vae,
            num_steps=num_steps,
            cfg_scale=cfg_scale,
            timestep_scale=self.timestep_scale,
            latent_channels=self.latent_channels,
            latent_resolution=self.latent_resolution,
        )

        images = sampler.sample_and_decode(
            labels=labels,
            structure_embedding=None,  # unconditional structure
            device=device,
        )
        return images

    @torch.no_grad()
    def generate_counterfactual(self, source_imgs, target_labels, device=None,
                                 num_steps=None, cfg_scale=None):
        """
        Generate counterfactual images conditioned on source structure.

        Args:
            source_imgs: [B, 3, 256, 256] source fundus images (diffusion-normalized).
            target_labels: [B] or scalar, target DR grades (0-4).
            device: torch device.
            num_steps: Override inference steps.
            cfg_scale: Override guidance scale.

        Returns:
            images: [B, 3, 256, 256] counterfactual images.
        """
        if device is None:
            device = source_imgs.device
        if num_steps is None:
            num_steps = self.num_inference_steps
        if cfg_scale is None:
            cfg_scale = self.cfg_scale

        # Extract structural embedding from source
        s = self.retfound_encoder(source_imgs.to(device))  # [B, 1024]

        # Prepare target labels
        if isinstance(target_labels, (int, float)):
            B = source_imgs.shape[0]
            labels = torch.full((B,), float(target_labels), device=device)
        else:
            labels = target_labels.float().to(device)

        sampler = RectifiedFlowSampler(
            model=self.model,
            vae=self.vae,
            num_steps=num_steps,
            cfg_scale=cfg_scale,
            timestep_scale=self.timestep_scale,
            latent_channels=self.latent_channels,
            latent_resolution=self.latent_resolution,
        )

        images = sampler.sample_and_decode(
            labels=labels,
            structure_embedding=s,
            device=device,
        )
        return images

    @torch.no_grad()
    def evaluate_fid_and_save_best(self):
        num_samples = 0
        samples = self.generate_samples(num_samples=num_samples, device=self.device)

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
