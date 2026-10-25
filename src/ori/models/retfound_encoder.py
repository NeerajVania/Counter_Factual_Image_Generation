"""
RETFound-DINOv2 structural encoder wrapper.

Provides a frozen pretrained RETFound-DINOv2 ViT-Large/14 model as a
structural feature extractor for the ordinal diffusion pipeline.

The model is loaded from HuggingFace (YukunZhou/<model_name>) using the
official RETFound checkpoint format, and used only for inference — no
parameters are ever trained or updated.

Output: [B, 1024] global retinal representation used as structural
condition ``s`` in the DiT.

References:
    - RETFound repo: https://github.com/rmaphoh/RETFound
    - Checkpoint loading follows main_finetune.py (key='teacher', key remapping)
    - Feature extraction follows latent_feature.ipynb (mean-pooled patch tokens + LayerNorm)
    - Preprocessing follows latent_feature.ipynb (per-image per-channel z-score at 224x224)

Note:
    RETFound-DINOv2 checkpoints on HuggingFace may be gated.
    If downloading fails, authenticate first:
        huggingface-cli login
    or set the HF_TOKEN environment variable.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from huggingface_hub import hf_hub_download

# Available RETFound-DINOv2 checkpoints on HuggingFace
RETFOUND_MODELS = {
    "RETFound_dinov2_meh": "YukunZhou/RETFound_dinov2_meh",
    "RETFound_dinov2_shanghai": "YukunZhou/RETFound_dinov2_shanghai",
}


class RETFoundDINOv2Encoder(nn.Module):
    """
    Frozen RETFound-DINOv2 feature extractor for structural conditioning.

    Loads a ViT-Large/14 DINOv2 backbone with RETFound-specific pretrained
    weights and extracts a 1024-dimensional global retinal representation.

    The model is always in eval mode with all parameters frozen.
    Gradients are never computed through this module.

    The wrapper handles its own preprocessing internally: it undoes the
    diffusion pipeline's fundus-specific normalization, resizes to 224x224,
    and applies the per-image per-channel z-score normalization expected
    by RETFound-DINOv2.

    Args:
        model_name: Key into RETFOUND_MODELS, e.g. ``'RETFound_dinov2_meh'``
            or ``'RETFound_dinov2_shanghai'``.
        checkpoint_path: Optional explicit path to a ``.pth`` checkpoint file.
            If ``None``, the checkpoint is downloaded from HuggingFace.
    """
            
    # Diffusion pipeline normalization constants (from the EyePacs dataset)
    DIFFUSION_MEAN = [0.3704248070716858, 0.2282254546880722, 0.13915641605854034]
    DIFFUSION_STD = [0.23381589353084564, 0.1512117236852646, 0.09653093665838242]

    EMBED_DIM = 1024  # ViT-Large embedding dimension

    def __init__(
        self,
        model_name="RETFound_dinov2_meh",
        checkpoint_path=None,
    ):
        super().__init__()
        self.model_name = model_name

        # --- Build the ViT-Large/14 DINOv2 architecture via timm ---
        # Matches official RETFound: timm.create_model('vit_large_patch14_dinov2.lvd142m')
        # pretrained=False because we load RETFound-specific weights, not generic DINOv2
        # num_classes=0 removes the classification head (we only need features)
        self.backbone = timm.create_model(
            "vit_large_patch14_dinov2.lvd142m",
            pretrained=False,
            img_size=224,
            num_classes=0,
        )

        # --- Load RETFound-specific pretrained weights ---
        self._load_retfound_weights(model_name, checkpoint_path)

        # --- LayerNorm for pooled features (matches official latent_feature.ipynb) ---
        self.pool_norm = nn.LayerNorm(self.EMBED_DIM, eps=1e-6)

        # --- Register diffusion normalization constants as buffers ---
        # These move to the correct device automatically with .to(device)
        self.register_buffer(
            "_diff_mean",
            torch.tensor(self.DIFFUSION_MEAN).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "_diff_std",
            torch.tensor(self.DIFFUSION_STD).view(1, 3, 1, 1),
        )

        # --- Freeze everything ---
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def _load_retfound_weights(self, model_name, checkpoint_path):
        """Download and load RETFound-DINOv2 pretrained weights.

        Follows the official loading procedure from ``main_finetune.py``:
        1. Download checkpoint from HuggingFace (``YukunZhou/<model_name>``)
        2. Extract the ``'teacher'`` state dict
        3. Apply key remapping: ``backbone.`` prefix removal, MLP key fixes
        4. Load with ``strict=False`` (head keys may be missing/mismatched)
        """
        if checkpoint_path is None:
            if model_name not in RETFOUND_MODELS:
                raise ValueError(
                    f"Unknown RETFound model '{model_name}'. "
                    f"Available: {list(RETFOUND_MODELS.keys())}"
                )
            repo_id = RETFOUND_MODELS[model_name]
            filename = f"{model_name}.pth"
            print(f"[RETFoundEncoder] Downloading checkpoint: {repo_id}/{filename}")
            checkpoint_path = hf_hub_download(repo_id=repo_id, filename=filename)

        print(f"[RETFoundEncoder] Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # RETFound-DINOv2 stores weights under 'teacher' key
        if "teacher" not in checkpoint:
            raise KeyError(
                f"Expected key 'teacher' in checkpoint. Found keys: {list(checkpoint.keys())}"
            )
        checkpoint_model = checkpoint["teacher"]

        # Key remapping (official RETFound convention from main_finetune.py)
        checkpoint_model = {
            k.replace("backbone.", ""): v for k, v in checkpoint_model.items()
        }
        checkpoint_model = {
            k.replace("mlp.w12.", "mlp.fc1."): v for k, v in checkpoint_model.items()
        }
        checkpoint_model = {
            k.replace("mlp.w3.", "mlp.fc2."): v for k, v in checkpoint_model.items()
        }

        msg = self.backbone.load_state_dict(checkpoint_model, strict=False)
        print(f"[RETFoundEncoder] Loaded. Missing keys: {msg.missing_keys}")
        if msg.unexpected_keys:
            print(f"[RETFoundEncoder] Unexpected keys (ignored): {msg.unexpected_keys}")

    def _preprocess_for_retfound(self, imgs):
        """
        Convert images from diffusion-pipeline normalization to
        RETFound-DINOv2 expected preprocessing.

        This follows the official preprocessing from ``latent_feature.ipynb``:
        resize to 224x224, scale to [0,1], per-image per-channel z-score.

        Args:
            imgs: [B, 3, 256, 256] normalized with diffusion pipeline mean/std.

        Returns:
            [B, 3, 224, 224] with per-image per-channel z-score normalization.
        """
        # 1. Denormalize: undo diffusion pipeline normalization → approx [0, 1]
        x = imgs * self._diff_std + self._diff_mean
        x = x.clamp(0.0, 1.0)

        # 2. Resize 256×256 → 224×224 (RETFound input resolution)
        if x.shape[-1] != 224 or x.shape[-2] != 224:
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

        # 3. Per-image per-channel z-score normalization
        #    (official RETFound preprocessing from latent_feature.ipynb)
        mean = x.mean(dim=(2, 3), keepdim=True)   # [B, 3, 1, 1]
        std = x.std(dim=(2, 3), keepdim=True)      # [B, 3, 1, 1]
        std = std.clamp(min=1e-8)                   # avoid division by zero
        x = (x - mean) / std

        return x

    def train(self, mode=True):
        """Override to ensure the model always stays in eval mode."""
        return super().train(False)

    def forward(self, imgs):
        """
        Extract the global retinal representation from fundus images.

        Args:
            imgs: [B, 3, H, W] tensor, normalized with diffusion pipeline mean/std.
                  Typically [B, 3, 256, 256].

        Returns:
            s: [B, 1024] global retinal representation for structural conditioning.
        """
        # --- Preprocessing: diffusion normalization → RETFound normalization ---
        x = self._preprocess_for_retfound(imgs)

        # --- Forward through frozen ViT-Large/14 backbone ---
        # timm's forward_features returns [B, N+1, D] (N patches + 1 CLS token)
        features = self.backbone.forward_features(x)

        # --- Pool: mean of patch tokens (exclude CLS), matching official RETFound ---
        if features.ndim == 3:
            # [B, N+1, D] → mean pool patch tokens (skip CLS at index 0)
            s = features[:, 1:, :].mean(dim=1)  # [B, D]
        elif features.ndim == 2:
            # timm may return already-pooled features [B, D]
            s = features
        else:
            raise ValueError(f"Unexpected feature shape: {features.shape}")

        # --- LayerNorm (matches official RETFound latent_feature.ipynb) ---
        s = self.pool_norm(s)

        # --- Shape safety ---
        if s.ndim == 1:
            s = s.unsqueeze(0)

        # --- Assertions ---
        assert s.ndim == 2, f"Expected 2D output, got shape {s.shape}"
        assert s.shape[1] == self.EMBED_DIM, (
            f"Expected embed_dim={self.EMBED_DIM}, got {s.shape[1]}"
        )
        assert not torch.isnan(s).any(), "RETFound output contains NaN values"

        return s
