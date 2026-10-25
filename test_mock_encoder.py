"""
Mock validation test for the RETFound-DINOv2 structural encoder.
This test bypasses the HuggingFace download and large ViT model instantiation
to quickly verify the architecture, shapes, parameter freezing, and DiT integration.
"""

import torch
import torch.nn as nn
import sys
import os
from unittest.mock import patch

# Add the src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Mock a ViT backbone
class MockViTBackbone(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.embed_dim = 1024
        self.patch_embed = nn.Linear(3, 1024) # dummy
    
    def forward_features(self, x):
        # x is [B, 3, 224, 224]
        B = x.shape[0]
        # Return dummy features simulating [B, N_patches+1, D]
        # ViT-L/14 with 224x224 input has (224/14)**2 = 256 patches + 1 CLS = 257
        return torch.randn(B, 257, 1024, device=x.device)
    
    def load_state_dict(self, state_dict, strict=False):
        from collections import namedtuple
        return namedtuple('IncompatibleKeys', ['missing_keys', 'unexpected_keys'])([], [])


def mock_create_model(*args, **kwargs):
    return MockViTBackbone()

def mock_hf_download(*args, **kwargs):
    return "dummy_path.pth"

def mock_torch_load(*args, **kwargs):
    return {"teacher": {"backbone.dummy": torch.tensor([1.0])}}


def run_mock_test():
    print("=" * 60)
    print("Mock RETFound-DINOv2 Structural Encoder Validation")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Patch timm and huggingface functions
    with patch("timm.create_model", side_effect=mock_create_model), \
         patch("huggingface_hub.hf_hub_download", side_effect=mock_hf_download), \
         patch("torch.load", side_effect=mock_torch_load):
        
        from ori.models.retfound_encoder import RETFoundDINOv2Encoder
        
        print("\n[Test 1] Instantiating Mock RETFound encoder...")
        encoder = RETFoundDINOv2Encoder(model_name="RETFound_dinov2_meh")
        encoder = encoder.to(device)
        print("  PASS: Encoder instantiated successfully")

        print("\n[Test 2] Forward pass with dummy batch...")
        batch_size = 4
        dummy_imgs = torch.randn(batch_size, 3, 256, 256, device=device)
        with torch.no_grad():
            s = encoder(dummy_imgs)
        print(f"  PASS: Forward pass succeeded, output shape: {list(s.shape)}")

        print("\n[Test 3] Output shape verification...")
        assert s.ndim == 2, f"Expected 2D tensor, got {s.ndim}D"
        assert s.shape[0] == batch_size, f"Expected batch size {batch_size}, got {s.shape[0]}"
        assert s.shape[1] == 1024, f"Expected dim 1024, got {s.shape[1]}"
        print(f"  PASS: Shape is correct: [{batch_size}, 1024]")

        print("\n[Test 4] NaN check...")
        assert not torch.isnan(s).any(), "Output contains NaN values!"
        print(f"  PASS: No NaN values in output")

        print("\n[Test 5] Parameter freeze verification...")
        for param in encoder.parameters():
            assert not param.requires_grad, "Found unfrozen parameter!"
        print(f"  PASS: All parameters are frozen (requires_grad=False)")

        print("\n[Test 6] Eval mode verification...")
        assert not encoder.training, "Encoder is in training mode!"
        encoder.train(True)
        assert not encoder.training, "Encoder switched to training mode!"
        print(f"  PASS: Encoder is in eval mode (and stays there)")

        print("\n[Test 7] DiT conditioning compatibility (c = t + y + s)...")
        hidden_size = 1024
        t = torch.randn(batch_size, hidden_size, device=device)
        y = torch.randn(batch_size, hidden_size, device=device)
        c = t + y + s
        assert c.shape == (batch_size, hidden_size), f"Expected [{batch_size}, {hidden_size}], got {list(c.shape)}"
        print(f"  PASS: c = t + y + s produces correct shape: {list(c.shape)}")

    print("\n" + "=" * 60)
    print("ALL MOCK TESTS PASSED")
    print("=" * 60)

if __name__ == "__main__":
    run_mock_test()
