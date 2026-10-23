"""
NCM-Apex: Block-Diagonal Self-Attention Kernel & Attention Masks
Implements hierarchical multi-scale attention patterns for Stage-2 Transformer Prior:
1. Block-Diagonal Attention Mask (NCM default: isolated diagonal blocks)
2. Prefix-LM Attention Mask (for conditional generation and prompt conditioning)
3. Comparative Attention Masks: Standard Causal, Block-Causal, Sliding Window
4. SDPA integration for memory-efficient attention on Ampere GPUs (RTX 3070 Ti)
"""

from typing import List, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_block_diagonal_mask(
    levels: Sequence[int],
    prefix_len: int = 0,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float32,
    return_4d: bool = True,
) -> torch.Tensor:
    """
    Constructs the Block-Diagonal Attention Mask M in {0, -inf}^{S x S}.

    Mathematical Formulation:
        Let N = sum_{k=1}^K l_k be the target sequence length and P be the prefix length.
        S = P + N.
        For target positions i, j in [0, N-1]:
            M_{P+i, P+j} = 0.0  if level(i) == level(j)
                         = -inf otherwise
        If P > 0 (Prefix-LM Mode):
            M_{:P, :P}   = 0.0  (Prompt tokens attend bidirectionally to all prompt tokens)
            M_{P:, :P}   = 0.0  (Target tokens at any level attend to all prompt tokens)
            M_{:P, P:}   = -inf (Prompt tokens CANNOT attend to target tokens)

    Why this works (Paper Section F.2.1 & Table 9):
        Tokens at level k attend ONLY to peer tokens within level k. Cross-level
        attention is completely redundant and wasteful because the input conditioning
        vector z_hat^(k-1) already aggregates and summarizes all prior scales 1..k-1.
        Block-diagonal isolation reduces self-attention FLOPs from O((sum l_k)^2)
        to O(sum l_k^2), yielding ~40% faster training throughput with zero loss
        in generative modeling quality.

    Args:
        levels: List of token lengths per scale [l_1, l_2, ..., l_K].
        prefix_len: Number of prompt/prefix conditioning tokens P (default: 0).
        device: PyTorch device (default: None).
        dtype: Data type (default: torch.float32).
        return_4d: If True, shapes tensor as (1, 1, S, S) for direct broadcast in SDPA.

    Returns:
        torch.Tensor of shape (1, 1, S, S) or (S, S).
    """
    P = int(prefix_len)
    levels = list(levels)
    N = sum(levels)
    S = P + N

    # Target level indicators
    level_list = []
    for k, l_k in enumerate(levels):
        level_list.append(torch.full((l_k,), k, dtype=torch.long, device=device))
    target_levels = torch.cat(level_list)  # (N,)

    # Target-to-target block-diagonal equality
    d_row = target_levels.unsqueeze(1)  # (N, 1)
    d_col = target_levels.unsqueeze(0)  # (1, N)
    target_target_mask = torch.where(
        d_row == d_col,
        torch.tensor(0.0, dtype=dtype, device=device),
        torch.tensor(float("-inf"), dtype=dtype, device=device),
    )

    if P == 0:
        mask_2d = target_target_mask
    else:
        # P > 0: construct full (P + N, P + N) matrix
        prompt_prompt = torch.zeros((P, P), dtype=dtype, device=device)
        prompt_target = torch.full((P, N), float("-inf"), dtype=dtype, device=device)
        target_prompt = torch.zeros((N, P), dtype=dtype, device=device)

        top_row = torch.cat([prompt_prompt, prompt_target], dim=1)  # (P, S)
        bottom_row = torch.cat([target_prompt, target_target_mask], dim=1)  # (N, S)
        mask_2d = torch.cat([top_row, bottom_row], dim=0)  # (S, S)

    if return_4d:
        return mask_2d.unsqueeze(0).unsqueeze(0)  # (1, 1, S, S)
    return mask_2d


def build_block_causal_mask(
    levels: Sequence[int],
    prefix_len: int = 0,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float32,
    return_4d: bool = True,
) -> torch.Tensor:
    """
    Constructs the Block-Causal Attention Mask (where level k attends to all prior levels <= k).
    """
    P = int(prefix_len)
    levels = list(levels)
    N = sum(levels)
    S = P + N

    level_list = [torch.full((l,), k, dtype=torch.long, device=device) for k, l in enumerate(levels)]
    target_levels = torch.cat(level_list)

    d_row = target_levels.unsqueeze(1)  # (N, 1)
    d_col = target_levels.unsqueeze(0)  # (1, N)
    target_target_mask = torch.where(
        d_row >= d_col,
        torch.tensor(0.0, dtype=dtype, device=device),
        torch.tensor(float("-inf"), dtype=dtype, device=device),
    )

    if P == 0:
        mask_2d = target_target_mask
    else:
        prompt_prompt = torch.zeros((P, P), dtype=dtype, device=device)
        prompt_target = torch.full((P, N), float("-inf"), dtype=dtype, device=device)
        target_prompt = torch.zeros((N, P), dtype=dtype, device=device)

        top_row = torch.cat([prompt_prompt, prompt_target], dim=1)
        bottom_row = torch.cat([target_prompt, target_target_mask], dim=1)
        mask_2d = torch.cat([top_row, bottom_row], dim=0)

    if return_4d:
        return mask_2d.unsqueeze(0).unsqueeze(0)
    return mask_2d


def build_sliding_window_mask(
    levels: Sequence[int],
    prefix_len: int = 0,
    window_levels: int = 1,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float32,
    return_4d: bool = True,
) -> torch.Tensor:
    """
    Constructs the Sliding Window Attention Mask (where level k attends to current level
    and preceding window_levels levels).
    """
    P = int(prefix_len)
    levels = list(levels)
    N = sum(levels)
    S = P + N

    level_list = [torch.full((l,), k, dtype=torch.long, device=device) for k, l in enumerate(levels)]
    target_levels = torch.cat(level_list)

    d_row = target_levels.unsqueeze(1)
    d_col = target_levels.unsqueeze(0)
    # Attends to j if: col <= row and col >= row - window_levels
    valid = (d_col <= d_row) & (d_col >= d_row - window_levels)
    target_target_mask = torch.where(
        valid,
        torch.tensor(0.0, dtype=dtype, device=device),
        torch.tensor(float("-inf"), dtype=dtype, device=device),
    )

    if P == 0:
        mask_2d = target_target_mask
    else:
        prompt_prompt = torch.zeros((P, P), dtype=dtype, device=device)
        prompt_target = torch.full((P, N), float("-inf"), dtype=dtype, device=device)
        target_prompt = torch.zeros((N, P), dtype=dtype, device=device)

        top_row = torch.cat([prompt_prompt, prompt_target], dim=1)
        bottom_row = torch.cat([target_prompt, target_target_mask], dim=1)
        mask_2d = torch.cat([top_row, bottom_row], dim=0)

    if return_4d:
        return mask_2d.unsqueeze(0).unsqueeze(0)
    return mask_2d


def build_standard_causal_mask(
    total_len: int,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float32,
    return_4d: bool = True,
) -> torch.Tensor:
    """
    Standard autoregressive lower-triangular causal mask.
    M_{i, j} = 0.0 if j <= i else -inf.
    """
    mask_2d = torch.full((total_len, total_len), float("-inf"), dtype=dtype, device=device)
    mask_2d = torch.triu(mask_2d, diagonal=1)
    mask_2d = torch.where(
        mask_2d == 0.0,
        torch.tensor(float("-inf"), dtype=dtype, device=device),
        torch.tensor(0.0, dtype=dtype, device=device),
    )
    # More simply:
    lower_tri = torch.tril(torch.ones((total_len, total_len), dtype=torch.bool, device=device))
    mask_2d = torch.where(
        lower_tri,
        torch.tensor(0.0, dtype=dtype, device=device),
        torch.tensor(float("-inf"), dtype=dtype, device=device),
    )

    if return_4d:
        return mask_2d.unsqueeze(0).unsqueeze(0)
    return mask_2d


def prepare_sdpa_mask(
    mask: torch.Tensor,
    batch_size: Optional[int] = None,
    n_heads: Optional[int] = None,
) -> torch.Tensor:
    """
    Adapts an attention mask tensor for seamless execution in PyTorch SDPA
    (F.scaled_dot_product_attention).

    Ensures the mask is at least 4D: (B, H, S_q, S_k) or (1, 1, S_q, S_k).
    """
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 3:
        mask = mask.unsqueeze(1)

    if batch_size is not None and mask.shape[0] != batch_size and mask.shape[0] == 1:
        mask = mask.expand(batch_size, -1, -1, -1)
    if n_heads is not None and mask.shape[1] != n_heads and mask.shape[1] == 1:
        mask = mask.expand(-1, n_heads, -1, -1)

    return mask
