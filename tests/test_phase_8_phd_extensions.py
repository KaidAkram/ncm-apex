"""
NCM-Apex: Automated PyTest Suite for Phase 8 (Novel PhD Research Extensions)
Verifies:
1. test_early_exit_stopping_condition: tau_exit=10.0 halts at k=ceil(K/2)=6, while tau_exit=0.0 forces 12 steps.
2. test_early_exit_pareto_easy_vs_hard: Easy Sudoku exits earlier than Hard (k_easy < k_hard).
3. test_confusion_matrix_properties: Rows sum to 1.0, diagonal is 0.0, nearest neighbors have highest replacement probability.
4. test_confusion_corruption_generation: Corrupted codes are valid ints in [0, V_k-1] with expected noise fraction.
5. test_copy_gate_numeral_preservation: Exact pointer routing copies out-of-codebook digits/entities when attention entropy is low.
6. test_vram_and_overhead: Peak GPU memory during combined execution remains under 3.5 GiB on RTX 3070 Ti.
"""

import os
import json
from pathlib import Path
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent

from src.utils.checkpoint_loader import load_sudoku_ncm_model, load_sudoku_dataset
from src.benchmarks.sudoku_suite import NCMSudokuRunner
from src.engine.early_exit import compute_distribution_drift, EarlyExitSudokuSolver
from src.engine.confusion_corruption import compute_codebook_confusion_matrix, ConfusionCorruptionScheduler
from src.layers.copy_gate import ConfidenceGatedCopyHead


@pytest.fixture(scope="module")
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def easy_runner(device):
    return NCMSudokuRunner(tier="easy", device=device)


@pytest.fixture(scope="module")
def hard_runner(device):
    return NCMSudokuRunner(tier="hard", device=device)


def test_early_exit_stopping_condition(easy_runner):
    """
    Test 1: Verify early-exit stopping condition.
    tau_exit = 10.0 forces early exit at k = ceil(K / 2) = 6.
    tau_exit = 0.0 forces execution of all 12 steps.
    """
    solver = EarlyExitSudokuSolver(easy_runner)
    puzzles, solutions = load_sudoku_dataset(tier="easy")
    puz = puzzles[0]
    sol = solutions[0]

    # Extreme threshold forcing exit at earliest allowed scale (min_exit_step=6)
    res_early = solver.solve_puzzle_with_early_exit(puz, sol, exit_threshold=10.0, min_exit_step=6)
    assert res_early["exited_early"] is True, "Expected early exit with tau=10.0"
    assert res_early["exit_step"] == 6, f"Expected exit at step 6, got {res_early['exit_step']}"
    assert res_early["nfe_used"] == 6

    # Zero threshold disabling early exit (forces all 12 steps)
    res_full = solver.solve_puzzle_with_early_exit(puz, sol, exit_threshold=0.0, min_exit_step=6)
    assert res_full["exited_early"] is False, "Expected no early exit with tau=0.0"
    assert res_full["exit_step"] == 12, f"Expected full 12 steps, got {res_full['exit_step']}"
    assert res_full["nfe_used"] == 12


def test_early_exit_pareto_easy_vs_hard(easy_runner, hard_runner):
    """
    Test 2: Pareto Frontier on Sudoku.
    Easy puzzles with 40 clues exit at k* ~ 7.2 (40% NFE reduction)
    while Hard puzzles with 30 clues dynamically refuse early exit (utilizing 12 steps).
    Asserts k_easy < k_hard strictly.
    """
    solver_easy = EarlyExitSudokuSolver(easy_runner)
    solver_hard = EarlyExitSudokuSolver(hard_runner)

    puzzles_easy, _ = load_sudoku_dataset(tier="easy")
    puzzles_hard, _ = load_sudoku_dataset(tier="hard")

    tau = 0.02
    easy_steps = []
    hard_steps = []

    for i in range(5):
        r_e = solver_easy.solve_puzzle_with_early_exit(puzzles_easy[i], exit_threshold=tau, min_exit_step=6)
        r_h = solver_hard.solve_puzzle_with_early_exit(puzzles_hard[i], exit_threshold=tau, min_exit_step=6)
        easy_steps.append(r_e["exit_step"])
        hard_steps.append(r_h["exit_step"])

    mean_easy = float(np.mean(easy_steps))
    mean_hard = float(np.mean(hard_steps))

    print(f"Mean Exit Steps - Easy: {mean_easy:.1f} | Hard: {mean_hard:.1f}")
    assert mean_easy < mean_hard, f"Expected mean_easy < mean_hard, got {mean_easy} >= {mean_hard}"
    assert mean_easy <= 8.0, f"Expected Easy Sudoku to exit by step 8, got {mean_easy}"
    assert mean_hard >= 11.0, f"Expected Hard Sudoku to run 12 steps, got {mean_hard}"


def test_confusion_matrix_properties():
    """
    Test 3: Confusion Matrix Mathematical Properties.
    Rows sum to 1.0, diagonal is 0.0, and closest centroid receives highest probability.
    """
    torch.manual_seed(42)
    # Synthetic 4-centroid codebook in 2D
    # e0 at (0, 0), e1 at (0.1, 0), e2 at (5.0, 5.0), e3 at (10.0, 10.0)
    # e1 is very close to e0; e2 and e3 are far away.
    emb = torch.tensor([
        [0.0, 0.0],
        [0.1, 0.0],
        [5.0, 5.0],
        [10.0, 10.0],
    ], dtype=torch.float32)

    conf_mat = compute_codebook_confusion_matrix(emb, tau_c=0.5)

    assert conf_mat.shape == (4, 4)
    # 1. Diagonal must be exactly 0.0
    for i in range(4):
        assert conf_mat[i, i].item() == 0.0, f"Diagonal at [{i}, {i}] must be 0.0"

    # 2. Rows must sum to 1.0 (valid probability distribution)
    row_sums = conf_mat.sum(dim=-1)
    for i, s in enumerate(row_sums):
        assert abs(s.item() - 1.0) < 1e-5, f"Row {i} sum {s.item()} != 1.0"

    # 3. For centroid 0, centroid 1 must have strictly higher probability than 2 and 3
    p_0_to_1 = conf_mat[0, 1].item()
    p_0_to_2 = conf_mat[0, 2].item()
    p_0_to_3 = conf_mat[0, 3].item()
    assert p_0_to_1 > p_0_to_2 > p_0_to_3, (
        f"Nearest neighbor geometry violated: p(0->1)={p_0_to_1}, p(0->2)={p_0_to_2}, p(0->3)={p_0_to_3}"
    )


def test_confusion_corruption_generation():
    """
    Test 4: Confusion Corruption Generation.
    Asserts corrupted codes remain valid ints in [0, V_k-1] and approximate target noise fraction.
    """
    torch.manual_seed(123)
    V, d = 256, 64
    emb = torch.randn(V, d)
    emb = F.normalize(emb, dim=-1)

    target_prob = 0.40
    scheduler = ConfusionCorruptionScheduler(
        codebooks=[emb],
        tau_c=0.5,
        p_start=target_prob,
        p_end=target_prob,
    )

    B, L = 16, 128
    original_codes = torch.randint(0, V, (B, L))
    corrupted_codes, mask = scheduler.corrupt_codes(original_codes, level_idx=0)

    # 1. Shape preservation
    assert corrupted_codes.shape == original_codes.shape
    assert mask.shape == original_codes.shape

    # 2. Codebook boundary bounds
    assert corrupted_codes.min() >= 0
    assert corrupted_codes.max() < V

    # 3. Corrupted codes differ where mask is True
    corrupted_subset = corrupted_codes[mask]
    original_subset = original_codes[mask]
    # Because diagonal of confusion matrix is 0, corrupted code != original code
    assert (corrupted_subset != original_subset).all(), "Corrupted codes must differ from original"

    # 4. Where mask is False, code is preserved identically
    assert (corrupted_codes[~mask] == original_codes[~mask]).all()

    # 5. Observed noise rate matches target within reasonable binomial margin
    observed_rate = mask.float().mean().item()
    assert abs(observed_rate - target_prob) < 0.05, f"Expected ~{target_prob}, got {observed_rate}"


def test_copy_gate_numeral_preservation(device):
    """
    Test 5: Confidence-Gated Residual Copy Head Numeral & Entity Preservation.
    Feed input with sharp attention on prompt entity; assert pointer routing copies exact source token.
    """
    torch.manual_seed(999)
    d_model = 256
    vocab_size = 1000
    copy_head = ConfidenceGatedCopyHead(d_model=d_model, vocab_size=vocab_size, tau_copy=0.60).to(device)

    B, L_target, P = 1, 1, 8
    h_target = torch.randn(B, L_target, d_model, device=device)

    # Prompt tokens containing a specific out-of-codebook year token: 957 (representing "1957")
    prompt_tokens = torch.tensor([[10, 42, 957, 105, 23, 7, 88, 99]], dtype=torch.long, device=device)

    # 1. Simulate sharp focus on prompt token index 2 ("1957") -> Low attention entropy
    sharp_attn = torch.zeros(B, L_target, P, device=device)
    sharp_attn[0, 0, 2] = 0.96
    sharp_attn[0, 0, 0] = 0.02
    sharp_attn[0, 0, 1] = 0.02

    discrete_logits = torch.randn(B, L_target, vocab_size, device=device)
    # Discrete codebook mistakenly predicts token 42 ("BMW") with high logit
    discrete_logits[0, 0, 42] = 5.0
    discrete_logits[0, 0, 957] = -2.0

    out = copy_head(
        hidden_states=h_target,
        discrete_logits=discrete_logits,
        cross_attention_weights=sharp_attn,
        prompt_tokens=prompt_tokens,
    )

    final_probs = out["final_probs"]
    attn_entropy = out["attention_entropy"]
    gate_values = out["gate_values"]

    # Entropy should be well below tau_copy (0.60)
    assert attn_entropy[0, 0].item() < 0.60
    assert gate_values[0, 0].item() > 0.0

    # Top prediction MUST be the copied prompt token (957, "1957"), overriding discrete mistake!
    predicted_token = final_probs.argmax(dim=-1)[0, 0].item()
    assert predicted_token == 957, f"Expected copied numeral token 957, but got {predicted_token}"


def test_vram_and_overhead(easy_runner, device):
    """
    Test 6: GPU Memory Profile & Telemetry.
    Asserts combined execution of all 3 extensions keeps peak VRAM below 3.5 GiB on RTX 3070 Ti.
    """
    if "cuda" not in str(device):
        pytest.skip("CUDA device required for VRAM profiling")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    # 1. Extension 8A run
    solver = EarlyExitSudokuSolver(easy_runner)
    puzzles, _ = load_sudoku_dataset("easy")
    _ = solver.solve_puzzle_with_early_exit(puzzles[0], exit_threshold=0.02, min_exit_step=6)

    # 2. Extension 8B run
    emb = torch.randn(512, 384, device=device)
    scheduler = ConfusionCorruptionScheduler([emb], tau_c=0.5)
    _ = scheduler.corrupt_codes(torch.randint(0, 512, (8, 64), device=device), level_idx=0)

    # 3. Extension 8C run
    copy_head = ConfidenceGatedCopyHead(d_model=384, vocab_size=2048, tau_copy=0.60).to(device)
    h = torch.randn(2, 16, 384, device=device)
    logits = torch.randn(2, 16, 2048, device=device)
    attn = F.softmax(torch.randn(2, 16, 8, device=device), dim=-1)
    prompt = torch.randint(0, 2048, (2, 8), device=device)
    _ = copy_head(
        hidden_states=h,
        codebook_logits=logits,
        prompt_tokens=prompt,
        cross_attention_weights=attn,
    )

    peak_bytes = torch.cuda.max_memory_allocated(device)
    peak_gib = peak_bytes / (1024 ** 3)
    print(f"\nPeak Allocated VRAM across all Phase 8 extensions: {peak_gib:.3f} GiB")

    # Assert under 3.5 GiB limit
    assert peak_gib < 3.5, f"Peak VRAM exceeded 3.5 GiB threshold: {peak_gib:.3f} GiB"
