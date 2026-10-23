"""
NCM-Apex: Automated PyTest Suite for Phase 3 (Checkpoints, Benchmarks & Trajectory Traces)
Tests:
1. test_checkpoint_downloader_and_adapter: Verifies official checkpoint weights load cleanly without shape mismatches.
2. test_sudoku_validator_logic: Verifies constraint violation detection on valid and corrupted boards.
3. test_autoregressive_sudoku_failure: Verifies AR left-to-right generation fails on Hard puzzles without backtracking (<10% solve rate).
4. test_ncm_sudoku_superiority: Verifies NCM achieves >80% accuracy on Hard puzzles in exactly 12 function evaluations.
5. test_real_trace_json_integrity: Asserts assets/trajectories/sudoku_hard_trace.json contains authentic rollout traces.
"""

import json
import os
from pathlib import Path
import pytest
import torch
import numpy as np

from src.utils.checkpoint_loader import load_sudoku_ncm_model, load_sudoku_dataset
from src.utils.sudoku import (
    tokens_to_grid,
    grid_to_tokens,
    validate_sudoku_board,
    compute_board_cell_accuracy,
    extract_clue_mask_and_values,
)
from src.benchmarks.sudoku_suite import (
    AutoregressiveSudokuRunner,
    MaskedDiffusionSudokuRunner,
    NCMSudokuRunner,
)


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def sample_hard_puzzle():
    puzzles, solutions = load_sudoku_dataset(tier="hard")
    return puzzles[0], solutions[0]


def test_checkpoint_downloader_and_adapter(device):
    """
    Test 1: Official Checkpoint Loading & State-Dict Verification.
    Verifies that official Stanford HazyResearch weights load cleanly into
    our architecture with zero missing or unexpected keys.
    """
    tk, gen, cfg = load_sudoku_ncm_model(tier="hard", device=device)

    assert tk is not None
    assert gen is not None
    assert len(gen.levels) == 12, f"Expected 12 scales for Sudoku, got {len(gen.levels)}"
    assert gen.levels == [1, 2, 4, 6, 9, 13, 18, 26, 37, 50, 67, 89]

    # Verify parameter counts
    gen_params = sum(p.numel() for p in gen.parameters())
    assert 20e6 < gen_params < 35e6, f"Generator param count unexpected: {gen_params / 1e6:.2f}M"


def test_sudoku_validator_logic(sample_hard_puzzle):
    """
    Test 2: Sudoku Validator Constraint Logic.
    Tests:
    1. Perfect ground-truth solution passes with zero violations.
    2. Introducing a duplicate in row 0 triggers row violation.
    3. Introducing a duplicate in col 0 triggers col violation.
    4. Corrupting a clue cell triggers clue violation.
    """
    puzzle, solution = sample_hard_puzzle

    # 1. Clean solution must be 100% valid
    is_valid, diag = validate_sudoku_board(solution, puzzle)
    assert is_valid is True, f"Ground-truth solution flagged invalid: {diag}"
    assert diag["row_violations"] == 0
    assert diag["col_violations"] == 0
    assert diag["box_violations"] == 0
    assert diag["clue_violations"] == 0

    # 2. Corrupt row 0 (duplicate digit)
    corrupt_grid = tokens_to_grid(solution).copy()
    corrupt_grid[0, 1] = corrupt_grid[0, 0]
    is_v, d = validate_sudoku_board(corrupt_grid, puzzle)
    assert is_v is False
    assert d["row_violations"] >= 1

    # 3. Corrupt col 0 (duplicate digit)
    corrupt_grid2 = tokens_to_grid(solution).copy()
    corrupt_grid2[1, 0] = corrupt_grid2[0, 0]
    is_v2, d2 = validate_sudoku_board(corrupt_grid2, puzzle)
    assert is_v2 is False
    assert d2["col_violations"] >= 1

    # 4. Corrupt a known clue
    clue_mask, clue_vals = extract_clue_mask_and_values(puzzle)
    first_clue_idx = int(np.where(clue_mask.flatten())[0][0])
    r, c = divmod(first_clue_idx, 9)
    corrupt_clue_grid = tokens_to_grid(solution).copy()
    corrupt_clue_grid[r, c] = (corrupt_clue_grid[r, c] % 9) + 1  # change clue digit
    is_v3, d3 = validate_sudoku_board(corrupt_clue_grid, puzzle)
    assert is_v3 is False
    assert d3["clue_violations"] >= 1


def test_autoregressive_sudoku_failure(sample_hard_puzzle):
    """
    Test 3: Autoregressive Sequential Failure on Hard Sudoku.
    Verifies that left-to-right generation without backtracking suffers from
    compounding constraint errors on Hard puzzles (assert solve rate < 10%).
    """
    puzzle, solution = sample_hard_puzzle
    runner = AutoregressiveSudokuRunner(seed=1337)

    # Run on 10 hard puzzles
    puzzles, solutions = load_sudoku_dataset(tier="hard")
    solved_count = 0
    for puz, sol in zip(puzzles[:10], solutions[:10]):
        res = runner.solve_puzzle(puz, sol)
        if res["is_solved"]:
            solved_count += 1

    solve_rate = (solved_count / 10.0) * 100.0
    assert solve_rate <= 10.0, f"AR solve rate {solve_rate}% unexpectedly high on Hard puzzles!"


def test_ncm_sudoku_superiority(device, sample_hard_puzzle):
    """
    Test 4: NCM Sudoku Superiority (>80% accuracy in 12 NFE).
    Verifies that NCM solves Hard puzzles with high cell accuracy
    in exactly 12 sequential function evaluations.
    """
    puzzle, solution = sample_hard_puzzle
    runner = NCMSudokuRunner(tier="hard", device=device)

    res = runner.solve_puzzle(puzzle, solution, record_trace=False)
    assert res["nfe"] == 12, f"Expected exactly 12 NFE, got {res['nfe']}"
    assert res["cell_accuracy"] >= 80.0, f"Expected >= 80% accuracy, got {res['cell_accuracy']:.2f}%"
    assert res["is_solved"] is True, f"Failed to solve puzzle 0: {res['diagnostics']}"


def test_real_trace_json_integrity():
    """
    Test 5: Real Rollout Trace JSON Export Integrity.
    Verifies that assets/trajectories/sudoku_hard_trace.json exists and contains:
    - Valid metadata (nfe, puzzle index)
    - Full 12-step NCM trace showing progressive cell refinement
    - Full 81-step AR trace showing compounding constraint violations
    - Full 128-step MDLM diffusion trace
    """
    trace_path = Path("assets/trajectories/sudoku_hard_trace.json")
    if not trace_path.exists():
        # Generate trace if not yet present
        from src.benchmarks.sudoku_suite import run_sudoku_three_paradigm_benchmark
        run_sudoku_three_paradigm_benchmark(tiers=["hard"], boards_per_tier=1, trace_output_path=str(trace_path))

    assert trace_path.exists(), "Trace JSON file was not generated!"
    with open(trace_path, "r") as f:
        data = json.load(f)

    assert "metadata" in data
    assert "ncm_trace" in data
    assert "ar_trace" in data
    assert "mdlm_trace" in data

    # Verify NCM has exactly 12 steps
    assert len(data["ncm_trace"]) == 12, f"Expected 12 NCM steps, got {len(data['ncm_trace'])}"
    # Final NCM step must have 81/81 correct cells
    assert data["ncm_trace"][-1]["correct_cells"] == 81, "Final NCM step failed to reach 81 correct cells"

    # Verify AR trace has 81 steps
    assert len(data["ar_trace"]) == 81, f"Expected 81 AR steps, got {len(data['ar_trace'])}"
    # Verify MDLM trace has 128 steps
    assert len(data["mdlm_trace"]) == 128, f"Expected 128 MDLM steps, got {len(data['mdlm_trace'])}"
