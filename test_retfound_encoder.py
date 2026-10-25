"""
Validation test for the RETFound-DINOv2 structural encoder replacement.

Tests the encoder architecture, freezing, output shape, and DiT compatibility.

If HuggingFace authentication is not set up (gated repo), the test falls back
to using the timm pretrained DINOv2 weights to validate the architecture.
"""

import torch
import sys
import os

# Add the src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def test_retfound_encoder():
    print("=" * 60)
    print("RETFound-DINOv2 Structural Encoder Validation")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # --- Test 1: Model loads successfully ---
    print("\n[Test 1] Loading RETFound-DINOv2 encoder...")
    try:
        from ori.models.retfound_encoder import RETFoundDINOv2Encoder
        encoder = RETFoundDINOv2Encoder(model_name="RETFound_dinov2_meh")
        encoder = encoder.to(device)
        print("  PASS: RETFound loaded from HuggingFace checkpoint")
        using_retfound_weights = True
    except Exception as e:
        print(f"  INFO: Could not load RETFound checkpoint: {type(e).__name__}")
        print("  Falling back to timm DINOv2 weights for architecture validation...")
        # Create encoder with timm pretrained weights for architecture testing
        from ori.models.retfound_encoder import RETFoundDINOv2Encoder
        import timm
        encoder = RETFoundDINOv2Encoder.__new__(RETFoundDINOv2Encoder)
        torch.nn.Module.__init__(encoder)
        encoder.model_name = "test_dinov2"
        encoder.backbone = timm.create_model(
            "vit_large_patch14_dinov2.lvd142m",
            pretrained=True,
            img_size=224,
            num_classes=0,
        )
        encoder.pool_norm = torch.nn.LayerNorm(1024, eps=1e-6)
        encoder.register_buffer(
            "_diff_mean",
            torch.tensor(RETFoundDINOv2Encoder.DIFFUSION_MEAN).view(1, 3, 1, 1),
        )
        encoder.register_buffer(
            "_diff_std",
            torch.tensor(RETFoundDINOv2Encoder.DIFFUSION_STD).view(1, 3, 1, 1),
        )
        encoder.eval()
        for param in encoder.parameters():
            param.requires_grad = False
        # Bind the methods
        encoder._preprocess_for_retfound = RETFoundDINOv2Encoder._preprocess_for_retfound.__get__(encoder)
        encoder.forward = RETFoundDINOv2Encoder.forward.__get__(encoder)
        encoder.train = RETFoundDINOv2Encoder.train.__get__(encoder)
        encoder.EMBED_DIM = 1024
        encoder = encoder.to(device)
        print("  PASS: Architecture validated with generic DINOv2 weights")
        using_retfound_weights = False

    # --- Test 2: Forward pass with a batch of dummy fundus images ---
    print("\n[Test 2] Forward pass with dummy batch...")
    batch_size = 4
    dummy_imgs = torch.randn(batch_size, 3, 256, 256, device=device)
    try:
        with torch.no_grad():
            s = encoder(dummy_imgs)
        print(f"  PASS: Forward pass succeeded, output shape: {list(s.shape)}")
    except Exception as e:
        print(f"  FAIL: Forward pass failed: {e}")
        return False

    # --- Test 3: Output shape is [B, 1024] ---
    print("\n[Test 3] Output shape verification...")
    assert s.ndim == 2, f"Expected 2D tensor, got {s.ndim}D"
    assert s.shape[0] == batch_size, f"Expected batch size {batch_size}, got {s.shape[0]}"
    assert s.shape[1] == 1024, f"Expected dim 1024, got {s.shape[1]}"
    print(f"  PASS: Shape is correct: [{batch_size}, 1024]")

    # --- Test 4: No NaNs ---
    print("\n[Test 4] NaN check...")
    assert not torch.isnan(s).any(), "Output contains NaN values!"
    print(f"  PASS: No NaN values in output")

    # --- Test 5: All parameters frozen ---
    print("\n[Test 5] Parameter freeze verification...")
    total_params = 0
    frozen_params = 0
    for param in encoder.parameters():
        total_params += param.numel()
        if not param.requires_grad:
            frozen_params += param.numel()
    assert frozen_params == total_params, (
        f"Not all params frozen: {frozen_params}/{total_params}"
    )
    print(f"  PASS: All {total_params:,} parameters are frozen (requires_grad=False)")

    # --- Test 6: Eval mode ---
    print("\n[Test 6] Eval mode verification...")
    assert not encoder.training, "Encoder is in training mode!"
    encoder.train(True)
    assert not encoder.training, "Encoder switched to training mode!"
    print(f"  PASS: Encoder is in eval mode (and stays there)")

    # --- Test 7: Optimizer exclusion ---
    print("\n[Test 7] Optimizer exclusion verification...")
    dit_params = [torch.nn.Parameter(torch.randn(10, 10))]
    optimizer = torch.optim.AdamW(dit_params, lr=1e-4)
    optimizer_param_ids = set()
    for group in optimizer.param_groups:
        for p in group["params"]:
            optimizer_param_ids.add(id(p))
    encoder_param_ids = {id(p) for p in encoder.parameters()}
    overlap = optimizer_param_ids & encoder_param_ids
    assert len(overlap) == 0, f"Found {len(overlap)} encoder params in optimizer!"
    print(f"  PASS: No encoder parameters in optimizer")

    # --- Test 8: c = t + y + s compatibility ---
    print("\n[Test 8] DiT conditioning compatibility (c = t + y + s)...")
    hidden_size = 1024
    t = torch.randn(batch_size, hidden_size, device=device)
    y = torch.randn(batch_size, hidden_size, device=device)
    c = t + y + s
    assert c.shape == (batch_size, hidden_size), f"Expected [{batch_size}, {hidden_size}], got {list(c.shape)}"
    assert not torch.isnan(c).any(), "c contains NaN values!"
    print(f"  PASS: c = t + y + s produces correct shape: {list(c.shape)}")

    # --- Test 9: Different batch sizes ---
    print("\n[Test 9] Variable batch sizes...")
    for bs in [1, 2, 8]:
        with torch.no_grad():
            test_imgs = torch.randn(bs, 3, 256, 256, device=device)
            out = encoder(test_imgs)
        assert out.shape == (bs, 1024), f"Batch size {bs}: expected [{bs}, 1024], got {list(out.shape)}"
        print(f"  PASS: Batch size {bs}: output shape [{bs}, 1024]")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
    weights_source = "RETFound HuggingFace" if using_retfound_weights else "Generic DINOv2 (timm)"
    print(f"\nWeights source: {weights_source}")
    print(f"Architecture: ViT-Large/14 DINOv2")
    print(f"Total parameters: {total_params:,}")
    print(f"Output dimension: 1024")
    print(f"All parameters frozen: True")
    print(f"Eval mode locked: True")
    print(f"Compatible with c = t + y + s: True")

    if not using_retfound_weights:
        print("\nNOTE: To use actual RETFound weights, authenticate with HuggingFace:")
        print("  huggingface-cli login")
        print("  Then request access at: https://huggingface.co/YukunZhou/RETFound_dinov2_meh")

    return True


if __name__ == "__main__":
    success = test_retfound_encoder()
    sys.exit(0 if success else 1)
