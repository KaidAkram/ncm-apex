"""
NCM-Apex: Sudoku Grid Engine, Tokenizer & Vectorized Constraint Evaluator
Supports 9x9 board representations, row/col/box uniqueness validation,
clue preservation checks, and token sequence conversions.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch


ROW_SEPARATOR_ID = 10
BOS_TOKEN_ID = 11
BLANK_TOKEN_ID = 0


def tokens_to_grid(tokens: Sequence[int]) -> np.ndarray:
    """
    Extracts an 81-cell 9x9 integer grid from a token sequence.
    Handles:
    - 89 tokens: 81 digits + 8 row separators (token 10)
    - 91 tokens: <BOS> (11) + 81 digits + 8 row separators (10) + <BOS> (11)
    - 81 tokens: raw digits
    """
    if isinstance(tokens, np.ndarray):
        if tokens.shape == (9, 9):
            return tokens.astype(np.int32)
        tokens = tokens.flatten().tolist()
    elif isinstance(tokens, torch.Tensor):
        if tokens.shape == (9, 9):
            return tokens.detach().cpu().numpy().astype(np.int32)
        tokens = tokens.detach().cpu().flatten().numpy().tolist()

    # Filter out BOS tokens and row separators
    digits = [int(t) for t in tokens if t not in (ROW_SEPARATOR_ID, BOS_TOKEN_ID)]
    if len(digits) != 81:
        # If length is truncated or padded, clamp or pad to 81
        if len(digits) < 81:
            digits = digits + [0] * (81 - len(digits))
        else:
            digits = digits[:81]

    grid = np.array(digits, dtype=np.int32).reshape((9, 9))
    return grid


def grid_to_tokens(grid: np.ndarray, add_bos: bool = False) -> List[int]:
    """
    Converts a 9x9 grid into standard NCM token sequence.
    Returns:
    - If add_bos is False: 89 tokens (9 rows of 9 digits separated by token 10).
    - If add_bos is True: 91 tokens (<BOS> + 89 tokens + <BOS>).
    """
    grid = np.asarray(grid, dtype=int).reshape((9, 9))
    tokens = []
    if add_bos:
        tokens.append(BOS_TOKEN_ID)

    for r in range(9):
        tokens.extend(grid[r].tolist())
        if r < 8:
            tokens.append(ROW_SEPARATOR_ID)

    if add_bos:
        tokens.append(BOS_TOKEN_ID)

    return tokens


def extract_clue_mask_and_values(puzzle_tokens: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts a boolean mask (81,) of clue positions and integer values (81,) from puzzle tokens.
    Clues are cells where digit in [1..9]. Blank cells are 0.
    """
    grid = tokens_to_grid(puzzle_tokens)
    clue_mask = (grid > 0) & (grid <= 9)
    return clue_mask, grid


def validate_sudoku_board(
    board: Union[Sequence[int], np.ndarray, torch.Tensor],
    puzzle: Optional[Union[Sequence[int], np.ndarray, torch.Tensor]] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Validates whether a 9x9 Sudoku board satisfies all mathematical constraints:
    1. Every cell contains a digit in {1, ..., 9} (no blanks).
    2. Each of the 9 rows contains all digits 1..9 without duplicates.
    3. Each of the 9 columns contains all digits 1..9 without duplicates.
    4. Each of the 9 3x3 subgrids contains all digits 1..9 without duplicates.
    5. All given clues from the initial puzzle are preserved exactly.

    Returns:
        is_valid: True if board has zero violations and solves the puzzle.
        diagnostics: Dictionary containing violation counts and cell accuracy.
    """
    grid = tokens_to_grid(board)
    row_violations = 0
    col_violations = 0
    box_violations = 0
    clue_violations = 0

    blank_count = int(np.sum(grid == 0))
    invalid_range_count = int(np.sum((grid < 0) | (grid > 9)))

    # 1. Row uniqueness
    for r in range(9):
        row_vals = grid[r, grid[r] > 0]
        if len(row_vals) != len(np.unique(row_vals)) or len(row_vals) != 9:
            row_violations += 1

    # 2. Column uniqueness
    for c in range(9):
        col_vals = grid[grid[:, c] > 0, c]
        if len(col_vals) != len(np.unique(col_vals)) or len(col_vals) != 9:
            col_violations += 1

    # 3. 3x3 Box uniqueness
    for br in range(3):
        for bc in range(3):
            subbox = grid[br*3:(br+1)*3, bc*3:(bc+1)*3]
            box_vals = subbox[subbox > 0]
            if len(box_vals) != len(np.unique(box_vals)) or len(box_vals) != 9:
                box_violations += 1

    # 4. Clue preservation
    clue_matches = 0
    total_clues = 0
    if puzzle is not None:
        p_grid = tokens_to_grid(puzzle)
        clue_mask = (p_grid > 0) & (p_grid <= 9)
        total_clues = int(np.sum(clue_mask))
        matching_clues = (grid[clue_mask] == p_grid[clue_mask])
        clue_matches = int(np.sum(matching_clues))
        clue_violations = total_clues - clue_matches

    # Board is fully valid if zero violations in all categories
    is_valid = (
        blank_count == 0
        and invalid_range_count == 0
        and row_violations == 0
        and col_violations == 0
        and box_violations == 0
        and clue_violations == 0
    )

    return is_valid, {
        "is_valid": is_valid,
        "blank_cells": blank_count,
        "row_violations": row_violations,
        "col_violations": col_violations,
        "box_violations": box_violations,
        "clue_violations": clue_violations,
        "total_clues": total_clues,
        "clue_matches": clue_matches,
    }


def compute_board_cell_accuracy(
    pred_board: Union[Sequence[int], np.ndarray, torch.Tensor],
    target_solution: Union[Sequence[int], np.ndarray, torch.Tensor],
) -> float:
    """
    Computes percentage of exactly matching cells against ground-truth solution [0.0 - 100.0].
    """
    pred_grid = tokens_to_grid(pred_board)
    sol_grid = tokens_to_grid(target_solution)
    matches = np.sum(pred_grid == sol_grid)
    return float(matches / 81.0) * 100.0


def print_sudoku_grid(grid: np.ndarray, title: str = "Sudoku Board") -> str:
    """Formats 9x9 grid into human-readable ASCII layout."""
    lines = [f"=== {title} ==="]
    for r in range(9):
        if r in (3, 6):
            lines.append("------+-------+------")
        row_str = []
        for c in range(9):
            if c in (3, 6):
                row_str.append("|")
            val = str(grid[r, c]) if grid[r, c] != 0 else "."
            row_str.append(val)
        lines.append(" ".join(row_str))
    return "\n".join(lines)
