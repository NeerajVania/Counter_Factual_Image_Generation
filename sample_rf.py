"""
Counterfactual fundus image generation using a trained Rectified Flow model.

Generates fundus images at specified DR severity grades while preserving the
anatomical structure of a source fundus image via frozen RETFound-DINOv2 conditioning.

Usage:
    # Generate counterfactuals for a single source image
    python sample_rf.py --exp_path model_weights/rf/experiment --source_image path/to/image.png --target_grades 0 1 2 3 4

    # Generate class-conditional samples (no source structure)
    python sample_rf.py --exp_path model_weights/rf/experiment --num_per_class 100 --mode unconditional

    # Generate counterfactuals from a CSV of source images
    python sample_rf.py --exp_path model_weights/rf/experiment --source_csv path/to/sources.csv --target_grades 0 1 2 3 4
"""

import os
import argparse

import torch
import torchvision
from torchvision.transforms import v2
from torchvision.utils import save_image
from omegaconf import OmegaConf
from PIL import Image

from ori.models import get_model
from ori.models.autoencoder import VAE
from ori.models.retfound_encoder import RETFoundDINOv2Encoder
from ori.rf_sampler import RectifiedFlowSampler, generate_counterfactual


def denormalize(img, mean=[0.3704248070716858, 0.2282254546880722, 0.13915641605854034],
                      std=[0.23381589353084564, 0.1512117236852646, 0.09653093665838242]):
    mean = torch.tensor(mean, device=img.device).view(1, -1, 1, 1)
    std = torch.tensor(std, device=img.device).view(1, -1, 1, 1)
    return img * std + mean


class RFSampler:
    """
    Standalone Rectified Flow sampler for loading a trained model and generating images.
    """

    def __init__(self, exp_path, cfg_scale=4.0, num_steps=50):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load config from experiment
        self.cfg = OmegaConf.load(exp_path + "/hparams.yaml").cfg

        # Load DiT model
        self.model = get_model(self.cfg)
        checkpoint_dir = os.path.join(exp_path, "checkpoints")
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")]
        if not checkpoint_files:
            raise FileNotFoundError(f"No .pt checkpoint found in {checkpoint_dir}")
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_files[0])
        msg = self.model.load_state_dict(
            torch.load(checkpoint_path, map_location=self.device), strict=False
        )
        print(f"[RFSampler] Loaded DiT checkpoint: {checkpoint_path}")
        if msg.missing_keys:
            print(f"  Missing keys: {msg.missing_keys}")
        if msg.unexpected_keys:
            print(f"  Unexpected keys: {msg.unexpected_keys}")
        self.model.to(self.device)
        self.model.eval()

        # Load frozen VAE
        self.vae = VAE(in_channels=3, latent_channels=4, ch_mult=[1, 2, 4])
        self.vae.load_state_dict(
            torch.load(self.cfg.vae.checkpoint, map_location=self.device)
        )
        self.vae.to(self.device)
        self.vae.eval()

        # Load frozen RETFound encoder (for counterfactual generation)
        if self.cfg.model.use_structure:
            encoder_name = self.cfg.encoder.name
            encoder_ckpt = self.cfg.encoder.checkpoint
            if not encoder_ckpt:
                encoder_ckpt = None
            self.retfound_encoder = RETFoundDINOv2Encoder(
                model_name=encoder_name,
                checkpoint_path=encoder_ckpt,
            )
            self.retfound_encoder.to(self.device)
        else:
            self.retfound_encoder = None

        # RF parameters
        rf_cfg = getattr(self.cfg, 'rectified_flow', None)
        self.timestep_scale = float(rf_cfg.timestep_scale) if rf_cfg else 999.0
        self.num_steps = num_steps
        self.cfg_scale = cfg_scale

        # Image transform for loading source images
        image_res = self.cfg.data.image_resolution
        self.transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((image_res, image_res), antialias=True),
            v2.Normalize(
                mean=[0.3704248070716858, 0.2282254546880722, 0.13915641605854034],
                std=[0.23381589353084564, 0.1512117236852646, 0.09653093665838242],
            ),
        ])

        # Output directory
        self.output_dir = os.path.join(exp_path, f"samples_rf_gw{cfg_scale}")
        os.makedirs(self.output_dir, exist_ok=True)

    def load_source_image(self, image_path):
        """Load and preprocess a source fundus image."""
        img = Image.open(image_path).convert("RGB")
        img_tensor = self.transform(img).unsqueeze(0)  # [1, 3, 256, 256]
        return img_tensor.to(self.device)

    @torch.no_grad()
    def generate_counterfactual(self, source_image, target_grade):
        """
        Generate a counterfactual image from a source image at a target DR grade.

        Args:
            source_image: [1, 3, 256, 256] preprocessed source image, or path string.
            target_grade: int (0-4), target DR severity.

        Returns:
            generated: [1, 3, 256, 256] generated image (denormalized, [0,1]).
        """
        if isinstance(source_image, str):
            source_image = self.load_source_image(source_image)

        generated = generate_counterfactual(
            source_image=source_image,
            target_dr_grade=target_grade,
            model=self.model,
            vae=self.vae,
            retfound_encoder=self.retfound_encoder,
            num_steps=self.num_steps,
            cfg_scale=self.cfg_scale,
            timestep_scale=self.timestep_scale,
            device=self.device,
        )

        # Denormalize to [0,1] for saving
        generated = denormalize(generated).clamp(0, 1)
        return generated

    @torch.no_grad()
    def generate_unconditional(self, labels, batch_size=4):
        """
        Generate class-conditional images without source structure.

        Args:
            labels: [N] tensor of target DR grades.
            batch_size: Batch size for generation.

        Returns:
            all_samples: [N, 3, 256, 256] generated images (denormalized, [0,1]).
        """
        all_samples = []
        for i in range(0, len(labels), batch_size):
            batch_labels = labels[i:i+batch_size].float().to(self.device)

            sampler = RectifiedFlowSampler(
                model=self.model,
                vae=self.vae,
                num_steps=self.num_steps,
                cfg_scale=self.cfg_scale,
                timestep_scale=self.timestep_scale,
            )

            images = sampler.sample_and_decode(
                labels=batch_labels,
                structure_embedding=None,
                device=self.device,
            )
            images = denormalize(images).clamp(0, 1)
            all_samples.append(images.cpu())
            torch.cuda.empty_cache()

        return torch.cat(all_samples, dim=0)


def save_generated_images_by_class(samples, labels, save_dir):
    """Save generated images organized by class directory."""
    os.makedirs(save_dir, exist_ok=True)
    class_counts = {}

    for img, label in zip(samples, labels):
        label_val = int(label.item()) if torch.is_tensor(label) else int(label)
        class_dir = os.path.join(save_dir, str(label_val))
        os.makedirs(class_dir, exist_ok=True)

        if label_val not in class_counts:
            class_counts[label_val] = len(os.listdir(class_dir))

        save_path = os.path.join(class_dir, f"{class_counts[label_val]}.png")
        save_image(img, save_path)
        class_counts[label_val] += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rectified Flow counterfactual generation")
    parser.add_argument("--exp_path", required=True, help="Path to trained RF experiment directory")
    parser.add_argument("--cfg_scale", type=float, default=4.0, help="Classifier-free guidance scale")
    parser.add_argument("--num_steps", type=int, default=50, help="Number of Euler integration steps")
    parser.add_argument("--mode", choices=["counterfactual", "unconditional"], default="unconditional",
                        help="Generation mode")

    # Counterfactual mode args
    parser.add_argument("--source_image", type=str, default=None,
                        help="Path to source fundus image for counterfactual generation")
    parser.add_argument("--target_grades", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                        help="Target DR grades to generate")

    # Unconditional mode args
    parser.add_argument("--num_per_class", type=int, default=100,
                        help="Number of images per class for unconditional generation")
    parser.add_argument("--out_dir", type=str, default="out/",
                        help="Output directory for generated images")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for generation")

    args = parser.parse_args()

    sampler = RFSampler(
        exp_path=args.exp_path,
        cfg_scale=args.cfg_scale,
        num_steps=args.num_steps,
    )

    if args.mode == "counterfactual" and args.source_image is not None:
        # --- Counterfactual generation ---
        print(f"Generating counterfactuals from: {args.source_image}")
        print(f"Target grades: {args.target_grades}")

        save_dir = os.path.join(args.out_dir, f"counterfactual_rf_gw{args.cfg_scale}")
        os.makedirs(save_dir, exist_ok=True)

        source_img = sampler.load_source_image(args.source_image)
        source_basename = os.path.splitext(os.path.basename(args.source_image))[0]

        for grade in args.target_grades:
            cf_image = sampler.generate_counterfactual(source_img, grade)
            save_path = os.path.join(save_dir, f"{source_basename}_to_DR{grade}.png")
            save_image(cf_image, save_path)
            print(f"  Saved: {save_path}")

        # Also save the source for comparison
        source_denorm = denormalize(source_img).clamp(0, 1)
        save_image(source_denorm, os.path.join(save_dir, f"{source_basename}_source.png"))
        print(f"Done. Counterfactuals saved to {save_dir}")

    else:
        # --- Unconditional class-conditional generation ---
        print(f"Generating {args.num_per_class} images per class (unconditional)")
        dr_levels = [0, 1, 2, 3, 4]
        gen_save_dir = os.path.join(
            args.out_dir, f"generated_rf_gw{args.cfg_scale}"
        )

        for cls_value in dr_levels:
            print(f"  Generating class DR{cls_value}...")
            labels = torch.full((args.num_per_class,), float(cls_value))
            samples = sampler.generate_unconditional(
                labels, batch_size=args.batch_size
            )

            class_labels = torch.full((args.num_per_class,), cls_value)
            save_generated_images_by_class(samples, class_labels, gen_save_dir)
            print(f"    Saved {len(samples)} images for DR{cls_value}")

        print(f"Done. Generated images saved to {gen_save_dir}")
