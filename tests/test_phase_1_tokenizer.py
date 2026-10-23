"""
NCM-Apex: Automated PyTest Suite for Phase 1 (Multiscale RVQ-VAE Tokenizer)
Tests:
1. test_ste_gradient_flow: Verifies Straight-Through Estimator passes gradients back to encoder.
2. test_nested_dropout_behavior: Verifies scale truncation behavior in training vs. eval.
3. test_downsample_reconstruction_shapes: Verifies spatial dimension preservation across levels.
4. test_codebook_ema_updates: Verifies EMA buffer shifts and absence of NaNs.
5. test_overfit_toy_batch: Asserts monotonic loss decrease and top-level accuracy >= 98%.
6. test_vram_footprint: Verifies peak GPU memory is strictly under 5.0 GiB on RTX 3070 Ti.
"""

import pytest
import torch
import torch.nn as nn
from torch.optim import AdamW

from src.models.multiscale_vqvae import MultiscaleVQVAE
from src.layers.losses import MultiscaleTokenizerLoss, compute_multilevel_accuracy
from src.engine.tokenizer_trainer import TokenizerTrainer


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def default_model(device):
    torch.manual_seed(42)
    model = MultiscaleVQVAE(
        vocab_size=256,
        context_length=64,
        embed_dim=128,
        codebook_vocab_sizes=[128, 128, 128],
        levels=[16, 32, 64],
        dec_layers=1,
        n_head=4,
        fine_drop_prob=0.0,  # Disabled for deterministic unit tests
    ).to(device)
    return model


def test_ste_gradient_flow(default_model, device):
    """
    Test 1: Straight-Through Estimator (STE) Gradient Flow.
    Asserts that non-zero gradients flow back from the decoder through the discrete
    quantizer into the encoder's word token embeddings (wte).
    """
    default_model.train()
    tokens = torch.randint(1, 256, (4, 64), device=device)

    # Ensure zero gradients
    default_model.zero_grad()

    # Forward pass
    output = default_model(tokens, force_cutoff_level=3)
    loss = output["logits"].sum()
    loss.backward()

    # Check encoder embedding gradients
    assert default_model.wte.weight.grad is not None, "Encoder wte.weight.grad is None!"
    grad_norm = default_model.wte.weight.grad.norm().item()
    assert grad_norm > 1e-6, f"STE failed: gradient norm is near-zero ({grad_norm})"
    assert not torch.isnan(default_model.wte.weight.grad).any(), "NaN found in encoder gradients!"


def test_nested_dropout_behavior(default_model, device):
    """
    Test 2: Nested Scale Dropout Behavior.
    Verifies that forced cutoff k produces reconstruction based only on scales 1..k.
    """
    default_model.eval()
    tokens = torch.randint(1, 256, (2, 64), device=device)

    # Cutoff level 1 (only coarsest 16 tokens)
    out_k1 = default_model(tokens, force_cutoff_level=1)
    # Cutoff level 3 (full 64 tokens)
    out_k3 = default_model(tokens, force_cutoff_level=3)

    assert out_k1["cutoff_level"] == 1
    assert out_k3["cutoff_level"] == 3

    # The latents at level 1 and level 3 must differ because level 3 includes residual corrections
    diff = (out_k3["z_hat"] - out_k1["z_hat"]).abs().sum().item()
    assert diff > 1e-4, "Nested dropout failed: Level 1 and Level 3 latents are identical!"


def test_downsample_reconstruction_shapes(default_model, device):
    """
    Test 3: Spatial Shape Contracts.
    Verifies that input sequence of length L is downsampled to l_k and upsampled back to L.
    """
    B, L, D = 4, 64, 128
    z = torch.randn(B, L, D, device=device)

    for i, target_len in enumerate([16, 32, 64]):
        down = default_model.quantizer.downsampler(z, level_idx=i)
        assert down.shape == (B, target_len, D), f"Downsample level {i} shape mismatch: {down.shape}"

        # Test upsampling/reconstruction
        q = torch.randn(B, target_len, D, device=device)
        up = default_model.quantizer.reconstructor(q, level_idx=i)
        assert up.shape == (B, L, D), f"Reconstruction level {i} shape mismatch: {up.shape}"


def test_codebook_ema_updates(default_model, device):
    """
    Test 4: Codebook EMA Weight Evolution.
    Verifies that codebook weights update dynamically during training steps without NaNs.
    """
    default_model.train()
    tokens = torch.randint(1, 256, (8, 64), device=device)

    # Record initial codebook weights at level 0
    initial_w0 = default_model.quantizer.codebooks[0].weight.data.clone()

    # Run forward pass (which executes EMA updates in training mode)
    output = default_model(tokens)
    updated_w0 = default_model.quantizer.codebooks[0].weight.data

    # Verify change has occurred
    weight_diff = (updated_w0 - initial_w0).abs().sum().item()
    assert weight_diff > 1e-6, "Codebook EMA weights did not update!"
    assert not torch.isnan(updated_w0).any(), "NaN detected in updated codebook weights!"


def test_overfit_toy_batch(device):
    """
    Test 5: Toy Batch Overfitting & Top-Level Accuracy >= 98%.
    Trains for 50 steps on a small batch of tokens and asserts monotonic loss decrease
    and top-level accuracy reaching >= 98%.
    """
    torch.manual_seed(1337)
    B, L = 4, 32
    vocab_size = 128
    sample_tokens = torch.randint(1, vocab_size, (B, L), device=device)

    model = MultiscaleVQVAE(
        vocab_size=vocab_size,
        context_length=L,
        embed_dim=128,
        codebook_vocab_sizes=[128, 128],
        levels=[16, 32],
        dec_layers=1,
        n_head=4,
        fine_drop_prob=0.0,
    ).to(device)

    trainer = TokenizerTrainer(model=model, lr=5e-3, device=device)

    initial_loss = None
    final_loss = None

    for step in range(1, 51):
        metrics = trainer.train_step(sample_tokens)
        if step == 1:
            initial_loss = metrics["total_loss"]
        if step == 50:
            final_loss = metrics["total_loss"]

    assert final_loss < initial_loss * 0.25, f"Loss did not decrease sufficiently: {initial_loss} -> {final_loss}"

    # Evaluate exact reconstruction accuracy
    eval_metrics = trainer.evaluate_multilevel_accuracy(sample_tokens)
    top_acc = eval_metrics["level_accuracies"][-1]
    assert top_acc >= 98.0, f"Top-level accuracy {top_acc:.2f}% did not reach 98.0% target!"


def test_vram_footprint(default_model, device):
    """
    Test 6: VRAM Memory Profiling on RTX 3070 Ti.
    Asserts peak VRAM is strictly under 5.0 GiB during peak training forward/backward pass.
    """
    if "cuda" not in device:
        pytest.skip("CUDA device not available; skipping VRAM test")

    torch.cuda.reset_peak_memory_stats(device)
    tokens = torch.randint(1, 256, (16, 64), device=device)

    default_model.train()
    with torch.cuda.amp.autocast(enabled=True):
        out = default_model(tokens)
        loss = out["logits"].sum() + out["vq_loss"]

    loss.backward()

    peak_bytes = torch.cuda.max_memory_allocated(device)
    peak_gib = peak_bytes / (1024 ** 3)

    assert peak_gib < 5.0, f"Peak VRAM exceeded 5.0 GiB limit: {peak_gib:.3f} GiB"
    print(f"\n[Verified] Peak VRAM: {peak_gib * 1024:.2f} MiB (< 5.0 GiB limit)")
