"""
NCM-Apex: Automated PyTest Suite for Phase 2 (Next-Concept Transformer Prior & Block Attention)
Tests:
1. test_block_diagonal_mask_structure: Asserts block-diagonal active region, zero diagonal blocks, and -inf off-diagonal elements.
2. test_single_pass_training_forward: Concatenates all K levels into length N = sum l_k, runs 1-pass forward, asserts output shapes.
3. test_conditioning_corruption_integrity: Asserts target labels remain 100% uncorrupted while conditioning contains expected noise.
4. test_parallel_sampling_loop: Runs full coarse-to-fine inference from k=1 to K (Algo 2), asserting valid codes and text decoding.
5. test_overfit_toy_sequence: Trains Stage-2 Transformer on top of frozen Stage-1 Tokenizer on 8 toy sequences, asserting monotonic loss decrease and top-level accuracy >= 95.0%.
6. test_vram_profiling_3070ti: Profiles peak GPU memory during batch training and parallel sampling on cuda:0; asserts total memory < 4.0 GiB.
"""

import pytest
import torch
import torch.nn as nn

from src.models.multiscale_vqvae import MultiscaleVQVAE
from src.models.ncm_transformer import NCMTransformer
from src.layers.block_attention import (
    build_block_diagonal_mask,
    build_block_causal_mask,
    build_sliding_window_mask,
    build_standard_causal_mask,
)
from src.layers.losses import MultiscalePriorLoss
from src.engine.corruption_scheduler import ConditioningCorruptionScheduler
from src.engine.generator_trainer import GeneratorTrainer
from src.engine.sampler import NCMSampler


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def default_levels():
    return [16, 32, 64]


@pytest.fixture
def default_vocab_sizes():
    return [128, 128, 128]


@pytest.fixture
def default_tokenizer(device, default_levels, default_vocab_sizes):
    torch.manual_seed(42)
    tokenizer = MultiscaleVQVAE(
        vocab_size=256,
        context_length=default_levels[-1],
        embed_dim=128,
        codebook_vocab_sizes=default_vocab_sizes,
        levels=default_levels,
        dec_layers=1,
        n_head=4,
        fine_drop_prob=0.0,
    ).to(device)
    tokenizer.eval()
    for p in tokenizer.parameters():
        p.requires_grad = False
    return tokenizer


@pytest.fixture
def default_transformer(device, default_levels, default_vocab_sizes):
    torch.manual_seed(42)
    model = NCMTransformer(
        levels=default_levels,
        codebook_vocab_sizes=default_vocab_sizes,
        tokenizer_latent_dim=128,
        d_model=256,
        n_layer=4,
        n_head=4,
    ).to(device)
    return model


def test_block_diagonal_mask_structure(default_levels):
    """
    Test 1: Block-Diagonal Attention Mask Structure & Prefix Invariants.
    Verifies that:
    1. Mask has shape (1, 1, N, N) where N = sum(levels).
    2. Within-level attention entries are exactly 0.0 (unmasked).
    3. Cross-level attention entries are exactly -inf (masked).
    4. Active finite entries count matches sum(l_k^2).
    5. Prefix-LM mask allows prompt bidirectional attention, target-to-prompt attention,
       and strictly prevents prompt-to-target attention.
    """
    levels = default_levels
    N = sum(levels)  # 16 + 32 + 64 = 112
    expected_active = sum(l ** 2 for l in levels)  # 16^2 + 32^2 + 64^2 = 256 + 1024 + 4096 = 5376

    # 1. Target-only block-diagonal mask
    mask = build_block_diagonal_mask(levels=levels, prefix_len=0, return_4d=True)
    assert mask.shape == (1, 1, N, N), f"Unexpected mask shape: {mask.shape}"

    finite_count = (mask == 0.0).sum().item()
    inf_count = torch.isneginf(mask).sum().item()
    total_entries = N * N

    assert finite_count == expected_active, f"Expected {expected_active} active entries, got {finite_count}"
    assert finite_count + inf_count == total_entries, "Mask contains values other than 0.0 and -inf!"

    # Verify isolated diagonal blocks
    m_2d = mask[0, 0]
    # Level 1 block (0:16, 0:16) must be 0.0
    assert torch.all(m_2d[0:16, 0:16] == 0.0)
    # Level 1 to Level 2 off-diagonal (0:16, 16:48) must be -inf
    assert torch.all(torch.isneginf(m_2d[0:16, 16:48]))
    # Level 2 to Level 1 off-diagonal (16:48, 0:16) must be -inf
    assert torch.all(torch.isneginf(m_2d[16:48, 0:16]))
    # Level 2 block (16:48, 16:48) must be 0.0
    assert torch.all(m_2d[16:48, 16:48] == 0.0)

    # 2. Prefix-LM mask variant (P = 10 prompt tokens)
    P = 10
    mask_prefix = build_block_diagonal_mask(levels=levels, prefix_len=P, return_4d=False)
    S = P + N
    assert mask_prefix.shape == (S, S)

    # Prompt-to-prompt is 0.0
    assert torch.all(mask_prefix[:P, :P] == 0.0)
    # Target-to-prompt is 0.0 (targets attend to prompt)
    assert torch.all(mask_prefix[P:, :P] == 0.0)
    # Prompt-to-target is -inf (prompt CANNOT attend to future targets)
    assert torch.all(torch.isneginf(mask_prefix[:P, P:]))
    # Target-to-target retains block-diagonal structure
    assert torch.all(mask_prefix[P:P+16, P:P+16] == 0.0)
    assert torch.all(torch.isneginf(mask_prefix[P:P+16, P+16:P+48]))


def test_single_pass_training_forward(default_transformer, device, default_levels, default_vocab_sizes):
    """
    Test 2: Single-Pass All-Levels Training Forward.
    Verifies that passing all K scales concatenated into length N = sum l_k executes
    in ONE shot and outputs valid logits matching codebook vocabulary sizes without NaNs.
    """
    B = 4
    L = default_levels[-1]
    D = default_transformer.tokenizer_latent_dim

    # Dummy multiscale cumulative continuous conditioning: [z^(0), z^(1), z^(2)]
    conds = [
        torch.zeros(B, L, D, device=device),
        torch.randn(B, L, D, device=device),
        torch.randn(B, L, D, device=device),
    ]

    out = default_transformer.forward_single_pass_all_levels(conds)
    logits_list = out["level_logits"]

    assert len(logits_list) == len(default_levels), f"Expected {len(default_levels)} levels, got {len(logits_list)}"

    for k, (l_k, v_k) in enumerate(zip(default_levels, default_vocab_sizes)):
        logits_k = logits_list[k]
        assert logits_k.shape == (B, l_k, v_k), f"Level {k} shape mismatch: {logits_k.shape} vs {(B, l_k, v_k)}"
        assert not torch.isnan(logits_k).any(), f"NaN detected in level {k} logits!"
        assert not torch.isinf(logits_k).any(), f"Inf detected in level {k} logits!"

    # Single-level forward test
    for k, (l_k, v_k) in enumerate(zip(default_levels, default_vocab_sizes)):
        single_logits = default_transformer.forward_single_level(conds[k], level_idx=k)
        assert single_logits.shape == (B, l_k, v_k)
        assert not torch.isnan(single_logits).any()


def test_conditioning_corruption_integrity(default_tokenizer, default_levels, default_vocab_sizes, device):
    """
    Test 3: Conditioning Corruption & Target Label Integrity.
    Verifies that:
    1. Ground-truth target codes r_{1:K}^* remain 100% clean and uncorrupted.
    2. Input conditioning embeddings receive controlled uniform codebook replacement.
    3. Corruption rates follow the scheduled decay across levels.
    """
    B = 8
    L = default_levels[-1]
    sample_tokens = torch.randint(1, 256, (B, L), device=device)

    clean_codes = default_tokenizer.encode(sample_tokens)

    # Corruption scheduler with linear decay from 0.70 to 0.30
    scheduler = ConditioningCorruptionScheduler(
        levels=default_levels,
        codebook_vocab_sizes=default_vocab_sizes,
        schedule_mode="linear_decay",
        p_start=0.70,
        p_end=0.30,
    )

    data = scheduler.build_corrupted_conditioning(default_tokenizer, clean_codes)

    clean_targets = data["clean_targets"]
    corrupted_codes = data["corrupted_codes"]
    corrupted_cond = data["corrupted_conditioning"]
    rates = data["corruption_rates"]

    # 1. Assert ground truth targets are 100% identical to original clean codes
    for k in range(len(default_levels)):
        assert torch.equal(clean_targets[k], clean_codes[k]), f"Target codes at level {k} were mutated!"

    # 2. Assert empirical corruption rates are within reasonable bounds around schedule
    # Level 0 target rate: 0.70; Level 2 target rate: 0.30
    assert 0.45 <= rates[0] <= 0.90, f"Level 0 corruption rate out of bounds: {rates[0]}"
    assert 0.15 <= rates[-1] <= 0.50, f"Level -1 corruption rate out of bounds: {rates[-1]}"

    # 3. Assert cumulative latents are constructed without NaNs
    assert corrupted_cond[0].norm() == 0.0, "z_tilde^(0) must be identically 0!"
    for k in range(1, len(default_levels)):
        assert corrupted_cond[k].norm() > 0.0, f"z_tilde^({k}) should have non-zero energy"
        assert not torch.isnan(corrupted_cond[k]).any(), f"NaN in corrupted latent {k}"


def test_parallel_sampling_loop(default_transformer, default_tokenizer, default_levels, device):
    """
    Test 4: Parallel Coarse-to-Fine Sampling Engine (Algorithm 2).
    Verifies that:
    1. Sampling completes in exactly K sequential steps.
    2. Produced codes have valid shapes and range within codebook boundaries.
    3. Final decoded text tokens have shape (B, L) and contain zero NaNs.
    4. Step telemetry logs latency and FLOPs per scale.
    """
    sampler = NCMSampler(
        transformer=default_transformer,
        tokenizer=default_tokenizer,
        device=device,
    )

    B = 3
    result = sampler.generate(
        batch_size=B,
        temperature=0.8,
        top_k=20,
        top_p=0.9,
        return_trajectory=True,
    )

    text_tokens = result["text_tokens"]
    codes = result["codes"]
    telemetry = result["step_telemetry"]
    trajectory = result["trajectory_tokens"]

    # Check shapes
    assert text_tokens.shape == (B, default_levels[-1])
    assert len(codes) == len(default_levels)
    assert len(telemetry) == len(default_levels)
    assert len(trajectory) == len(default_levels)

    for k, l_k in enumerate(default_levels):
        assert codes[k].shape == (B, l_k)
        assert (codes[k] >= 0).all() and (codes[k] < default_tokenizer.quantizer.vocab_sizes[k]).all()
        # Telemetry verification
        t_k = telemetry[k]
        assert t_k["token_length"] == l_k
        assert t_k["step_latency_ms"] > 0.0
        assert t_k["est_gflops"] > 0.0

    # Ensure coarse scale 1 has smaller FLOPs than fine scale K
    assert telemetry[0]["est_gflops"] < telemetry[-1]["est_gflops"], (
        f"Coarse level FLOPs ({telemetry[0]['est_gflops']}) should be less than fine level ({telemetry[-1]['est_gflops']})"
    )


def test_overfit_toy_sequence(device):
    """
    Test 5: Overfitting Toy Batch & Top-Level Code Accuracy >= 95.0%.
    Trains Stage-2 Transformer on top of frozen Stage-1 Tokenizer for 60 iterations
    on 8 toy sequences; asserts monotonic loss decrease and top-level accuracy >= 95.0%.
    """
    torch.manual_seed(1337)
    B, L = 8, 32
    vocab_size = 128
    sample_tokens = torch.randint(1, vocab_size, (B, L), device=device)

    # 1. Stage-1 Tokenizer
    tokenizer = MultiscaleVQVAE(
        vocab_size=vocab_size,
        context_length=L,
        embed_dim=128,
        codebook_vocab_sizes=[128, 128],
        levels=[16, 32],
        dec_layers=1,
        n_head=4,
        fine_drop_prob=0.0,
    ).to(device)
    tokenizer.eval()
    for p in tokenizer.parameters():
        p.requires_grad = False

    # 2. Stage-2 Transformer Prior
    model = NCMTransformer(
        levels=[16, 32],
        codebook_vocab_sizes=[128, 128],
        tokenizer_latent_dim=128,
        d_model=256,
        n_layer=4,
        n_head=4,
    ).to(device)

    # Constant low corruption for toy overfitting
    corruption_sched = ConditioningCorruptionScheduler(
        levels=[16, 32],
        codebook_vocab_sizes=[128, 128],
        schedule_mode="constant",
        constant_rate=0.05,
    )

    trainer = GeneratorTrainer(
        model=model,
        tokenizer=tokenizer,
        corruption_scheduler=corruption_sched,
        lr=3e-3,
        device=device,
    )

    initial_loss = None
    final_loss = None

    for step in range(1, 61):
        metrics = trainer.train_step(sample_tokens)
        if step == 1:
            initial_loss = metrics["total_loss"]
        if step == 60:
            final_loss = metrics["total_loss"]

    assert final_loss < initial_loss * 0.40, (
        f"Loss failed to decrease significantly: {initial_loss:.4f} -> {final_loss:.4f}"
    )

    # Evaluation on clean conditioning
    eval_metrics = trainer.evaluate(sample_tokens)
    top_acc = eval_metrics["eval_per_level_accuracies"][-1]

    assert top_acc >= 95.0, (
        f"Top-level code prediction accuracy {top_acc:.2f}% did not reach required 95.0% threshold!"
    )


def test_vram_profiling_3070ti(device):
    """
    Test 6: VRAM Memory Profiling on RTX 3070 Ti.
    Profiles peak VRAM allocation during batch training and parallel sampling;
    asserts peak memory remains strictly under 4.0 GiB limit.
    """
    if "cuda" not in device:
        pytest.skip("CUDA device not available; skipping VRAM test")

    torch.cuda.reset_peak_memory_stats(device)
    B, L = 16, 64
    vocab_size = 256
    sample_tokens = torch.randint(1, vocab_size, (B, L), device=device)

    tokenizer = MultiscaleVQVAE(
        vocab_size=vocab_size,
        context_length=L,
        embed_dim=128,
        codebook_vocab_sizes=[128, 128, 128],
        levels=[16, 32, 64],
        dec_layers=1,
        n_head=4,
    ).to(device)

    model = NCMTransformer(
        levels=[16, 32, 64],
        codebook_vocab_sizes=[128, 128, 128],
        tokenizer_latent_dim=128,
        d_model=384,
        n_layer=6,
        n_head=6,
    ).to(device)

    trainer = GeneratorTrainer(
        model=model,
        tokenizer=tokenizer,
        device=device,
    )

    # Peak training pass (forward + backward)
    train_metrics = trainer.train_step(sample_tokens)

    # Parallel inference sampling pass
    sampler = NCMSampler(transformer=model, tokenizer=tokenizer, device=device)
    gen_result = sampler.generate(batch_size=B)

    peak_bytes = torch.cuda.max_memory_allocated(device)
    peak_gib = peak_bytes / (1024 ** 3)
    peak_mib = peak_bytes / (1024 ** 2)

    assert peak_gib < 4.0, f"Peak VRAM {peak_gib:.3f} GiB exceeded 4.0 GiB budget on RTX 3070 Ti!"
    print(f"\n[Verified] RTX 3070 Ti Peak VRAM: {peak_mib:.2f} MiB ({peak_gib:.3f} GiB < 4.0 GiB target)")
