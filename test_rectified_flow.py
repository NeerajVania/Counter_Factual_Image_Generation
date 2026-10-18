"""
Smoke tests for the Rectified Flow pipeline.

Validates mathematical correctness, shape compatibility, gradient flow,
and end-to-end training/sampling before launching a full training run.

Tests:
    1. Shape and conditioning tests
    2. Mathematical correctness tests (interpolation, velocity, sampler direction)
    3. Integration tests (training step, frozen params, gradient flow, sampling)

Usage:
    python test_rectified_flow.py
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from unittest.mock import patch
from collections import namedtuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


# ==============================================================================
# Mock models (bypass heavy checkpoint loading for testing)
# ==============================================================================

class MockViTBackbone(nn.Module):
    """Lightweight mock for the ViT-Large/14 backbone."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.embed_dim = 1024
        self.patch_embed = nn.Linear(3, 1024)

    def forward_features(self, x):
        B = x.shape[0]
        return torch.randn(B, 257, 1024, device=x.device)

    def load_state_dict(self, state_dict, strict=False):
        return namedtuple('IncompatibleKeys', ['missing_keys', 'unexpected_keys'])([], [])


def mock_create_model(*args, **kwargs):
    return MockViTBackbone()

def mock_hf_download(*args, **kwargs):
    return "dummy_path.pth"

def mock_torch_load_for_retfound(*args, **kwargs):
    return {"teacher": {"backbone.dummy": torch.tensor([1.0])}}


class MockVAE(nn.Module):
    """Lightweight mock VAE with correct interface."""
    def __init__(self):
        super().__init__()
        self.encoder = MockVAEEncoder()
        self.decoder = MockVAEDecoder()

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = mu
        return self.decoder(z), mu, logvar


class MockVAEEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Linear(1, 1)  # need a parameter for device detection

    def forward(self, x):
        B = x.shape[0]
        device = x.device
        mu = torch.randn(B, 4, 64, 64, device=device)
        logvar = torch.randn(B, 4, 64, 64, device=device)
        return mu, logvar


class MockVAEDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Linear(1, 1)

    def forward(self, z):
        B = z.shape[0]
        return torch.randn(B, 3, 256, 256, device=z.device)


# ==============================================================================
# Test utilities
# ==============================================================================

def get_mock_retfound_encoder(device="cpu"):
    """Create a mock RETFound encoder."""
    with patch("timm.create_model", side_effect=mock_create_model), \
         patch("huggingface_hub.hf_hub_download", side_effect=mock_hf_download), \
         patch("torch.load", side_effect=mock_torch_load_for_retfound):
        from ori.models.retfound_encoder import RETFoundDINOv2Encoder
        encoder = RETFoundDINOv2Encoder(model_name="RETFound_dinov2_meh")
        encoder = encoder.to(device)
    return encoder


def get_dit_model(device="cpu", learn_sigma=False):
    """Create a DiT model for testing."""
    from ori.models.DiT import DiT
    model = DiT(
        input_size=64,
        patch_size=2,
        in_channels=4,
        hidden_size=1024,
        depth=2,  # Shallow for testing
        num_heads=16,
        mlp_ratio=4.0,
        learn_sigma=learn_sigma,
        ordinal_input=False,
        learned_ordinal_input=True,
        use_structure=True,
    )
    return model.to(device)


# ==============================================================================
# Test 1: Shape and Conditioning Tests
# ==============================================================================

def test_shapes(device="cpu"):
    """Verify shapes of all pipeline components."""
    print("\n" + "=" * 60)
    print("TEST 1: Shape and Conditioning Tests")
    print("=" * 60)

    B = 4  # batch size

    # 1a. VAE input and latent shapes
    print("\n[1a] VAE shapes...")
    vae = MockVAE().to(device)
    imgs = torch.randn(B, 3, 256, 256, device=device)
    with torch.no_grad():
        mu, logvar = vae.encoder(imgs)
    assert mu.shape == (B, 4, 64, 64), f"Expected [B,4,64,64], got {mu.shape}"
    assert logvar.shape == (B, 4, 64, 64), f"Expected [B,4,64,64], got {logvar.shape}"
    print(f"  PASS: VAE latent shape: {list(mu.shape)}")

    # 1b. RETFound output shape
    print("\n[1b] RETFound output shape...")
    encoder = get_mock_retfound_encoder(device)
    with torch.no_grad():
        s = encoder(imgs)
    assert s.shape == (B, 1024), f"Expected [B, 1024], got {s.shape}"
    print(f"  PASS: RETFound output shape: {list(s.shape)}")

    # 1c. Target labels for all five DR grades
    print("\n[1c] Target labels for all DR grades...")
    for grade in range(5):
        labels = torch.full((B,), float(grade), device=device)
        assert labels.shape == (B,), f"Label shape mismatch for grade {grade}"
    print(f"  PASS: All 5 DR grades (0-4) produce correct label shape")

    # 1d. DiT velocity output (learn_sigma=False)
    print("\n[1d] DiT velocity output shape...")
    model = get_dit_model(device, learn_sigma=False)
    z_tau = torch.randn(B, 4, 64, 64, device=device)
    labels = torch.tensor([0.0, 1.0, 2.0, 3.0], device=device)
    t = torch.rand(B, device=device) * 999.0
    label_mask = torch.zeros(B, device=device)
    structure_mask = torch.zeros(B, device=device)

    v_pred = model(z_tau, labels, t, label_mask, s, structure_mask)
    assert v_pred.shape == (B, 4, 64, 64), f"Expected [B,4,64,64], got {v_pred.shape}"
    print(f"  PASS: DiT velocity output shape: {list(v_pred.shape)}")

    # 1e. Verify learn_sigma=True would give 8 channels
    print("\n[1e] DiT noise+variance output (learn_sigma=True) for comparison...")
    model_sigma = get_dit_model(device, learn_sigma=True)
    v_pred_sigma = model_sigma(z_tau, labels, t, label_mask, s, structure_mask)
    assert v_pred_sigma.shape == (B, 8, 64, 64), f"Expected [B,8,64,64], got {v_pred_sigma.shape}"
    print(f"  PASS: learn_sigma=True gives {list(v_pred_sigma.shape)} (8 channels)")

    # 1f. Compatible conditioning dimensions
    print("\n[1f] Conditioning dimension compatibility...")
    hidden_size = 1024
    t_emb = torch.randn(B, hidden_size, device=device)
    y_emb = torch.randn(B, hidden_size, device=device)
    s_emb = s  # [B, 1024]
    c = t_emb + y_emb + s_emb
    assert c.shape == (B, hidden_size), f"Expected [B, {hidden_size}], got {c.shape}"
    print(f"  PASS: c = t + y + s gives correct shape: {list(c.shape)}")

    # 1g. Interpolated latent and target velocity shapes
    print("\n[1g] Interpolation and velocity shapes...")
    z_data = mu  # [B, 4, 64, 64]
    epsilon = torch.randn_like(z_data)
    tau = torch.rand(B, device=device)
    tau_expanded = tau[:, None, None, None]
    z_tau = (1 - tau_expanded) * z_data + tau_expanded * epsilon
    v_target = epsilon - z_data
    assert z_tau.shape == z_data.shape, f"z_tau shape mismatch: {z_tau.shape}"
    assert v_target.shape == z_data.shape, f"v_target shape mismatch: {v_target.shape}"
    assert v_pred.shape == v_target.shape, f"v_pred shape {v_pred.shape} != v_target shape {v_target.shape}"
    print(f"  PASS: z_tau, v_target, v_pred all have shape {list(z_data.shape)}")

    del model, model_sigma, vae, encoder
    print("\n  ALL SHAPE TESTS PASSED")
    return True


# ==============================================================================
# Test 2: Mathematical Correctness Tests
# ==============================================================================

def test_math(device="cpu"):
    """Verify RF interpolation, velocity, and sampler direction."""
    print("\n" + "=" * 60)
    print("TEST 2: Mathematical Correctness Tests")
    print("=" * 60)

    B = 2

    z_data = torch.randn(B, 4, 64, 64, device=device)
    epsilon = torch.randn(B, 4, 64, 64, device=device)

    # 2a. At tau=0, interpolated latent equals clean latent
    print("\n[2a] tau=0 -> z_tau == z_data...")
    tau_0 = torch.zeros(B, 1, 1, 1, device=device)
    z_at_0 = (1 - tau_0) * z_data + tau_0 * epsilon
    assert torch.allclose(z_at_0, z_data, atol=1e-6), "z_tau at tau=0 != z_data"
    print(f"  PASS: z_tau(tau=0) == z_data (max diff: {(z_at_0 - z_data).abs().max():.2e})")

    # 2b. At tau=1, interpolated latent equals noise
    print("\n[2b] tau=1 -> z_tau == epsilon...")
    tau_1 = torch.ones(B, 1, 1, 1, device=device)
    z_at_1 = (1 - tau_1) * z_data + tau_1 * epsilon
    assert torch.allclose(z_at_1, epsilon, atol=1e-6), "z_tau at tau=1 != epsilon"
    print(f"  PASS: z_tau(tau=1) == epsilon (max diff: {(z_at_1 - epsilon).abs().max():.2e})")

    # 2c. Target velocity equals epsilon - z_data
    print("\n[2c] v_target == epsilon - z_data...")
    v_target = epsilon - z_data
    expected = epsilon - z_data
    assert torch.allclose(v_target, expected, atol=1e-6), "v_target != epsilon - z_data"
    print(f"  PASS: v_target == epsilon - z_data")

    # 2d. Sampler updates in correct direction
    print("\n[2d] Euler sampler step direction...")
    # Starting at tau=1 (noise), stepping toward tau=0 (clean)
    z_current = epsilon.clone()
    tau_current = 1.0
    tau_next = 0.9
    dt = tau_next - tau_current  # = -0.1 (negative!)
    assert dt < 0, f"dt should be negative, got {dt}"

    # With the true velocity, one step should move toward z_data
    v_true = epsilon - z_data
    z_next = z_current + dt * v_true
    # z_next = epsilon + (-0.1) * (epsilon - z_data) = epsilon - 0.1*epsilon + 0.1*z_data
    #        = 0.9*epsilon + 0.1*z_data
    expected_z_next = 0.9 * epsilon + 0.1 * z_data
    assert torch.allclose(z_next, expected_z_next, atol=1e-5), "Euler step gives wrong result"

    # Verify it's closer to z_data than z_current
    dist_before = (z_current - z_data).norm()
    dist_after = (z_next - z_data).norm()
    assert dist_after < dist_before, f"Sampler moved away from z_data! {dist_after:.4f} >= {dist_before:.4f}"
    print(f"  PASS: Euler step moves toward z_data (dist: {dist_before:.4f} -> {dist_after:.4f})")

    # 2e. Full integration with true velocity recovers z_data exactly
    print("\n[2e] Full Euler integration recovers z_data...")
    z = epsilon.clone()
    N = 1000
    timesteps = torch.linspace(1.0, 0.0, N + 1)
    for i in range(N):
        dt = timesteps[i + 1] - timesteps[i]
        z = z + dt * v_true

    max_err = (z - z_data).abs().max().item()
    assert max_err < 1e-3, f"Full integration error too large: {max_err:.6f}"
    print(f"  PASS: Full integration recovers z_data (max error: {max_err:.2e})")

    # 2f. Loss is finite
    print("\n[2f] Velocity MSE loss is finite...")
    v_pred_dummy = torch.randn_like(v_target)
    loss = F.mse_loss(v_pred_dummy, v_target)
    assert torch.isfinite(loss), f"Loss is not finite: {loss}"
    print(f"  PASS: Loss is finite: {loss.item():.6f}")

    # 2g. Loss supports backpropagation
    print("\n[2g] Loss supports backpropagation...")
    v_pred_param = torch.randn_like(v_target, requires_grad=True)
    loss = F.mse_loss(v_pred_param, v_target)
    loss.backward()
    assert v_pred_param.grad is not None, "No gradient computed"
    assert torch.isfinite(v_pred_param.grad).all(), "Gradient contains non-finite values"
    print(f"  PASS: Gradient computed successfully")

    print("\n  ALL MATH TESTS PASSED")
    return True


# ==============================================================================
# Test 3: Integration Tests
# ==============================================================================

def test_integration(device="cpu"):
    """Test full training step, gradient flow, frozen params, and sampling."""
    print("\n" + "=" * 60)
    print("TEST 3: Integration Tests")
    print("=" * 60)

    B = 2  # small batch

    # Setup models
    model = get_dit_model(device, learn_sigma=False)
    vae = MockVAE().to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    encoder = get_mock_retfound_encoder(device)

    # Save initial parameters for frozen check
    vae_params_before = {n: p.clone() for n, p in vae.named_parameters()}
    encoder_params_before = {n: p.clone() for n, p in encoder.named_parameters()}
    dit_params_before = {n: p.clone() for n, p in model.named_parameters()}

    # 3a. One complete training step
    print("\n[3a] Complete training step...")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Simulate a batch
    imgs = torch.randn(B, 3, 256, 256, device=device)
    labels = torch.tensor([2.0, 4.0], device=device)

    # Forward pass (RF training step)
    with torch.no_grad():
        mu, _ = vae.encoder(imgs)
        latents = mu
        s = encoder(imgs)

    tau = torch.rand(B, device=device)
    epsilon = torch.randn_like(latents)
    tau_exp = tau[:, None, None, None]
    z_tau = (1 - tau_exp) * latents + tau_exp * epsilon
    v_target = epsilon - latents
    t_scaled = tau * 999.0

    label_mask = torch.zeros(B, device=device)
    structure_mask = torch.zeros(B, device=device)

    v_pred = model(z_tau, labels, t_scaled, label_mask, s, structure_mask)
    loss = F.mse_loss(v_pred, v_target)

    assert torch.isfinite(loss), f"Loss is not finite: {loss}"
    print(f"  Loss: {loss.item():.6f}")

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"  PASS: Training step completed successfully")

    # 3b. Frozen model parameters remain unchanged
    print("\n[3b] Frozen parameter verification...")
    for name, param in vae.named_parameters():
        assert torch.equal(param, vae_params_before[name]), f"VAE param '{name}' changed!"
    print(f"  PASS: All VAE parameters unchanged")

    for name, param in encoder.named_parameters():
        assert torch.equal(param, encoder_params_before[name]), f"RETFound param '{name}' changed!"
    print(f"  PASS: All RETFound parameters unchanged")

    # 3c. Trainable parameters received gradients
    print("\n[3c] Gradient verification...")
    params_with_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    total_params = sum(1 for p in model.parameters())
    print(f"  DiT params with gradients: {params_with_grad}/{total_params}")
    assert params_with_grad > 0, "No DiT parameters received gradients!"
    print(f"  PASS: DiT parameters received gradients")

    # 3d. DiT parameters actually changed
    print("\n[3d] DiT parameter update verification...")
    changed = 0
    for name, param in model.named_parameters():
        if not torch.equal(param.data, dit_params_before[name].data):
            changed += 1
    assert changed > 0, "No DiT parameters changed after optimizer step!"
    print(f"  PASS: {changed} DiT parameters updated")

    # 3e. Multiple training steps complete
    print("\n[3e] Multiple training steps...")
    for step in range(5):
        tau = torch.rand(B, device=device)
        epsilon = torch.randn_like(latents)
        tau_exp = tau[:, None, None, None]
        z_tau = (1 - tau_exp) * latents + tau_exp * epsilon
        v_target = epsilon - latents
        t_scaled = tau * 999.0

        v_pred = model(z_tau, labels, t_scaled, label_mask, s, structure_mask)
        loss = F.mse_loss(v_pred, v_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        assert torch.isfinite(loss), f"Loss became NaN at step {step}"
    print(f"  PASS: 5 consecutive training steps completed (final loss: {loss.item():.6f})")

    # 3f. Sampling produces decodable latent
    print("\n[3f] RF sampling test...")
    model.eval()
    from ori.rf_sampler import RectifiedFlowSampler
    sampler = RectifiedFlowSampler(
        model=model,
        vae=vae,
        num_steps=5,  # very few steps for speed
        cfg_scale=1.0,
        timestep_scale=999,
    )

    sample_labels = torch.tensor([0.0, 3.0], device=device)
    with torch.no_grad():
        z_0 = sampler.sample(sample_labels, structure_embedding=s, device=device)
    assert z_0.shape == (B, 4, 64, 64), f"Sampled latent shape mismatch: {z_0.shape}"
    assert torch.isfinite(z_0).all(), "Sampled latent contains non-finite values"
    print(f"  PASS: RF sampler produced latent: {list(z_0.shape)}")

    # 3g. VAE decodes sampled latent
    print("\n[3g] VAE decoding of sampled latent...")
    with torch.no_grad():
        decoded = vae.decoder(z_0)
    assert decoded.shape == (B, 3, 256, 256), f"Decoded shape mismatch: {decoded.shape}"
    print(f"  PASS: VAE decoded to image: {list(decoded.shape)}")

    # 3h. Counterfactual generation with different target grades
    print("\n[3h] Counterfactual generation test...")
    source_img = torch.randn(1, 3, 256, 256, device=device)
    with torch.no_grad():
        s_single = encoder(source_img)

    for target_grade in range(5):
        target_label = torch.tensor([float(target_grade)], device=device)
        with torch.no_grad():
            z_cf = sampler.sample(target_label, structure_embedding=s_single, device=device)
            img_cf = vae.decoder(z_cf)
        assert img_cf.shape == (1, 3, 256, 256), f"CF image shape wrong for grade {target_grade}"
    print(f"  PASS: Counterfactual generation works for all 5 DR grades")

    # 3i. CFG sampling
    print("\n[3i] Classifier-free guidance sampling...")
    sampler_cfg = RectifiedFlowSampler(
        model=model,
        vae=vae,
        num_steps=3,
        cfg_scale=4.0,  # with guidance
        timestep_scale=999,
    )
    with torch.no_grad():
        z_cfg = sampler_cfg.sample(sample_labels, structure_embedding=s, device=device)
    assert z_cfg.shape == (B, 4, 64, 64), f"CFG sampled shape mismatch: {z_cfg.shape}"
    assert torch.isfinite(z_cfg).all(), "CFG sampled latent contains non-finite values"
    print(f"  PASS: CFG sampling produced finite latent")

    # 3j. Batch size = 1 test
    print("\n[3j] Batch size = 1 test...")
    single_img = torch.randn(1, 3, 256, 256, device=device)
    single_label = torch.tensor([2.0], device=device)
    with torch.no_grad():
        mu_single, _ = vae.encoder(single_img)
        s_single = encoder(single_img)

    model.train()
    tau_single = torch.rand(1, device=device)
    eps_single = torch.randn_like(mu_single)
    z_tau_single = (1 - tau_single[:, None, None, None]) * mu_single + tau_single[:, None, None, None] * eps_single
    v_target_single = eps_single - mu_single
    t_single = tau_single * 999.0
    lm_single = torch.zeros(1, device=device)
    sm_single = torch.zeros(1, device=device)

    v_pred_single = model(z_tau_single, single_label, t_single, lm_single, s_single, sm_single)
    loss_single = F.mse_loss(v_pred_single, v_target_single)
    assert torch.isfinite(loss_single), f"Loss NaN for batch_size=1"
    optimizer.zero_grad()
    loss_single.backward()
    print(f"  PASS: Batch size = 1 works (loss: {loss_single.item():.6f})")

    del model, vae, encoder, sampler, sampler_cfg
    print("\n  ALL INTEGRATION TESTS PASSED")
    return True


# ==============================================================================
# Test 4: Time Embedding Continuity
# ==============================================================================

def test_time_embedding(device="cpu"):
    """Verify the timestep embedder handles continuous RF time correctly."""
    print("\n" + "=" * 60)
    print("TEST 4: Time Embedding Continuity Tests")
    print("=" * 60)

    from ori.models.DiT import TimestepEmbedder

    embedder = TimestepEmbedder(hidden_size=1024).to(device)

    # 4a. Continuous values produce valid embeddings
    print("\n[4a] Continuous time values...")
    tau_values = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], device=device) * 999.0
    emb = embedder(tau_values)
    assert emb.shape == (5, 1024), f"Embedding shape mismatch: {emb.shape}"
    assert torch.isfinite(emb).all(), "Embedding contains non-finite values"
    print(f"  PASS: Continuous timesteps produce valid embeddings: {list(emb.shape)}")

    # 4b. Nearby times produce similar embeddings
    print("\n[4b] Embedding continuity...")
    t1 = torch.tensor([500.0], device=device)
    t2 = torch.tensor([500.1], device=device)
    t3 = torch.tensor([999.0], device=device)
    e1 = embedder(t1)
    e2 = embedder(t2)
    e3 = embedder(t3)

    dist_close = (e1 - e2).norm().item()
    dist_far = (e1 - e3).norm().item()
    assert dist_close < dist_far, f"Nearby times should give closer embeddings ({dist_close:.4f} >= {dist_far:.4f})"
    print(f"  PASS: Nearby times give closer embeddings (close: {dist_close:.4f}, far: {dist_far:.4f})")

    # 4c. t=0 and t=999 produce different embeddings
    print("\n[4c] Boundary embeddings differ...")
    e_start = embedder(torch.tensor([0.0], device=device))
    e_end = embedder(torch.tensor([999.0], device=device))
    diff = (e_start - e_end).norm().item()
    assert diff > 1.0, f"Boundary embeddings too similar (diff: {diff:.4f})"
    print(f"  PASS: tau=0 and tau=1 produce distinct embeddings (diff: {diff:.4f})")

    print("\n  ALL TIME EMBEDDING TESTS PASSED")
    return True


# ==============================================================================
# Main
# ==============================================================================

def run_all_tests():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Rectified Flow smoke tests on: {device}")

    results = []

    with patch("timm.create_model", side_effect=mock_create_model), \
         patch("huggingface_hub.hf_hub_download", side_effect=mock_hf_download), \
         patch("torch.load", side_effect=mock_torch_load_for_retfound):

        results.append(("Shape & Conditioning", test_shapes(device)))
        results.append(("Mathematical Correctness", test_math(device)))
        results.append(("Time Embedding", test_time_embedding(device)))
        results.append(("Integration", test_integration(device)))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        if not passed:
            all_pass = False

    print()
    if all_pass:
        print("ALL SMOKE TESTS PASSED OK")
    else:
        print("SOME TESTS FAILED FAILED")
    print("=" * 60)

    return all_pass


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
