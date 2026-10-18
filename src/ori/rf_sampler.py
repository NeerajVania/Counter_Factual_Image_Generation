"""
Rectified Flow sampler for counterfactual fundus image generation.

Implements Euler integration of the learned velocity field from tau=1 (noise)
to tau=0 (clean data), with optional classifier-free guidance.

Time convention (consistent with training):
    tau = 0: clean data latent (z_data)
    tau = 1: Gaussian noise (epsilon)
    t_scaled = tau * timestep_scale  (for the sinusoidal timestep embedder)
"""

import torch
import torch.nn.functional as F


class RectifiedFlowSampler:
    """
    First-order Euler sampler for Rectified Flow.

    Integrates the learned velocity field backward from tau=1 (noise) to tau=0
    (clean data) using uniform time steps.

    Args:
        model: DiT model that predicts velocity v(z_tau, tau, c_y, s).
        vae: Frozen VAE decoder for converting latents to images.
        num_steps: Number of Euler integration steps.
        cfg_scale: Classifier-free guidance scale. 1.0 = no guidance.
        timestep_scale: Scale factor for mapping tau to timestep embedder input.
        latent_channels: Number of latent channels (default 4).
        latent_resolution: Spatial resolution of latents (default 64).
    """

    def __init__(
        self,
        model,
        vae,
        num_steps=50,
        cfg_scale=4.0,
        timestep_scale=999,
        latent_channels=4,
        latent_resolution=64,
    ):
        self.model = model
        self.vae = vae
        self.num_steps = num_steps
        self.cfg_scale = cfg_scale
        self.timestep_scale = timestep_scale
        self.latent_channels = latent_channels
        self.latent_resolution = latent_resolution

    @torch.no_grad()
    def sample(self, labels, structure_embedding=None, device=None):
        """
        Generate latents by integrating the velocity field.

        Args:
            labels: [B] tensor of target DR grades (0-4), float.
            structure_embedding: [B, 1024] RETFound embedding of the source
                fundus image, or None for unconditional generation.
            device: torch device.

        Returns:
            z_0: [B, C, H, W] clean latent (ready for VAE decoding).
        """
        B = labels.shape[0]
        if device is None:
            device = labels.device

        # Start from Gaussian noise at tau=1
        z = torch.randn(
            B, self.latent_channels, self.latent_resolution, self.latent_resolution,
            device=device,
        )

        # Uniform time steps from tau=1 to tau=0
        tau_steps = torch.linspace(1.0, 0.0, self.num_steps + 1, device=device)

        for i in range(self.num_steps):
            tau_current = tau_steps[i]
            tau_next = tau_steps[i + 1]
            dt = tau_next - tau_current  # negative (going from 1 to 0)

            # Map tau to timestep embedder scale
            t_scaled = torch.full((B,), tau_current.item() * self.timestep_scale, device=device)

            # Predict velocity (with optional CFG)
            v = self._predict_velocity(
                z, labels, t_scaled, structure_embedding, device
            )

            # Euler step: z_next = z_current + dt * v
            z = z + dt * v

        return z

    def _predict_velocity(self, z, labels, t_scaled, structure_embedding, device):
        """
        Predict velocity with optional classifier-free guidance.

        When cfg_scale > 1.0, runs the model twice:
        - Conditioned on label (and structure if provided)
        - Unconditional (label masked out, structure preserved for counterfactual)

        CFG is applied only to the label conditioning. Structure conditioning
        is always active during counterfactual generation to preserve anatomy.
        """
        B = z.shape[0]

        if self.cfg_scale > 1.0:
            # Duplicate inputs for conditioned + unconditional pass
            z_in = z.repeat(2, 1, 1, 1)
            labels_in = labels.repeat(2)
            t_in = t_scaled.repeat(2)

            # Label mask: first half conditioned (0), second half masked (1)
            label_mask = torch.zeros(2 * B, dtype=torch.float32, device=device)
            label_mask[B:] = 1.0

            # Structure: always active for counterfactual (never masked during sampling)
            if structure_embedding is not None:
                s_in = structure_embedding.repeat(2, 1)
                structure_mask = torch.zeros(2 * B, dtype=torch.float32, device=device)
            else:
                s_in = torch.zeros(2 * B, 1024, device=device)
                structure_mask = torch.ones(2 * B, dtype=torch.float32, device=device)

            v_pred = self.model(z_in, labels_in, t_in, label_mask, s_in, structure_mask)

            # Split and apply guidance
            v_cond, v_uncond = v_pred.chunk(2, dim=0)
            v = v_uncond + self.cfg_scale * (v_cond - v_uncond)
        else:
            # No guidance: single forward pass
            label_mask = torch.zeros(B, dtype=torch.float32, device=device)
            if structure_embedding is not None:
                structure_mask = torch.zeros(B, dtype=torch.float32, device=device)
                s = structure_embedding
            else:
                s = torch.zeros(B, 1024, device=device)
                structure_mask = torch.ones(B, dtype=torch.float32, device=device)

            v = self.model(z, labels, t_scaled, label_mask, s, structure_mask)

        return v

    @torch.no_grad()
    def sample_and_decode(self, labels, structure_embedding=None, device=None):
        """
        Generate images by sampling latents and decoding through the frozen VAE.

        Args:
            labels: [B] tensor of target DR grades (0-4), float.
            structure_embedding: [B, 1024] RETFound embedding of source image.
            device: torch device.

        Returns:
            images: [B, 3, 256, 256] generated fundus images (in diffusion
                pipeline normalization space).
        """
        z_0 = self.sample(labels, structure_embedding, device)
        images = self.vae.decoder(z_0)
        return images


def generate_counterfactual(
    source_image,
    target_dr_grade,
    model,
    vae,
    retfound_encoder,
    num_steps=50,
    cfg_scale=4.0,
    timestep_scale=999,
    device=None,
):
    """
    Generate a counterfactual fundus image for a specified DR grade,
    preserving the anatomical structure of the source image.

    Args:
        source_image: [1, 3, 256, 256] or [B, 3, 256, 256] source fundus image,
            normalized with the diffusion pipeline mean/std.
        target_dr_grade: int or float (0-4), or [B] tensor of target grades.
        model: Trained DiT model (in eval mode).
        vae: Frozen VAE (with encoder and decoder).
        retfound_encoder: Frozen RETFound-DINOv2 encoder.
        num_steps: Number of Euler integration steps.
        cfg_scale: Classifier-free guidance scale.
        timestep_scale: Scale for mapping tau to timestep embedder input.
        device: torch device.

    Returns:
        generated_image: [B, 3, 256, 256] counterfactual fundus image.

    Example:
        >>> # Source image is DR2, generate DR4 counterfactual
        >>> cf_image = generate_counterfactual(
        ...     source_image=dr2_image,   # [1, 3, 256, 256]
        ...     target_dr_grade=4,
        ...     model=dit_model,
        ...     vae=vae_model,
        ...     retfound_encoder=retfound,
        ... )
    """
    if device is None:
        device = next(model.parameters()).device

    source_image = source_image.to(device)

    # Ensure batch dimension
    if source_image.ndim == 3:
        source_image = source_image.unsqueeze(0)
    B = source_image.shape[0]

    # Extract structural representation from source image
    with torch.no_grad():
        s = retfound_encoder(source_image)  # [B, 1024]

    # Prepare target label
    if isinstance(target_dr_grade, (int, float)):
        labels = torch.full((B,), float(target_dr_grade), device=device)
    else:
        labels = target_dr_grade.float().to(device)

    # Sample using RF with structure conditioning
    sampler = RectifiedFlowSampler(
        model=model,
        vae=vae,
        num_steps=num_steps,
        cfg_scale=cfg_scale,
        timestep_scale=timestep_scale,
    )

    generated_image = sampler.sample_and_decode(
        labels=labels,
        structure_embedding=s,
        device=device,
    )

    return generated_image
