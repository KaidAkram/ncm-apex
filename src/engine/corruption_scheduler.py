"""
NCM-Apex: Conditioning Corruption Scheduler (Exposure Bias Mitigation)
Implements stochastic code corruption to combat compounding exposure bias
in parallel coarse-to-fine generation:
1. Schedule-controlled uniform codebook replacement
2. Linearly decreasing corruption rate schedule across scales (Appendix F.3)
3. Cumulative corrupted conditioning reconstruction:
   z_tilde^(k-1) = sum_{j=1}^{k-1} R_j(e_j(r_tilde_j))
4. Strict integrity preservation of clean ground-truth target codes r_{1:K}^*
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple
import torch
import torch.nn as nn


class ConditioningCorruptionScheduler:
    """
    Stochastic Code Corruption Scheduler.

    During inference, the generator feeds its own discrete code predictions forward,
    accumulating small errors. If trained exclusively on teacher-forced ground-truth
    codes, the model suffers from severe exposure bias (paper Section 3.3 and Appendix F.3).

    This scheduler replaces a fraction p_corrupt(k) of codes at scale k with uniformly
    sampled indices from codebook V_k:
        r_tilde_k^i = Uniform({0, ..., |V_k|-1}) with probability p_corrupt(k)
                    = r_k^i                      with probability 1 - p_corrupt(k)

    The conditioning latents z_tilde^(k-1) are reconstructed from these corrupted codes,
    while the prediction targets r_k^* remain 100% clean.
    """

    def __init__(
        self,
        levels: Sequence[int],
        codebook_vocab_sizes: Sequence[int],
        schedule_mode: str = "linear_decay",
        p_start: float = 0.70,
        p_end: float = 0.30,
        constant_rate: float = 0.50,
        random_rate_bounds: Tuple[float, float] = (0.0, 1.0),
        skip_coarsest: bool = False,
    ):
        """
        Args:
            levels: Sequence lengths per scale [l_1, ..., l_K].
            codebook_vocab_sizes: Cardinality |V_k| per scale.
            schedule_mode: 'linear_decay', 'constant', 'random_uniform', or 'disabled'.
            p_start: Initial corruption probability for coarsest level (default: 0.70).
            p_end: Final corruption probability for finest conditioning level (default: 0.30).
                   Mean rate = (0.70 + 0.30) / 2 = 0.50 (optimal as reported in Appendix F.3).
            constant_rate: Used when schedule_mode == 'constant'.
            random_rate_bounds: (low, high) used when schedule_mode == 'random_uniform'.
            skip_coarsest: If True, level 0 is never corrupted.
        """
        self.levels = list(levels)
        self.num_levels = len(levels)
        self.codebook_vocab_sizes = list(codebook_vocab_sizes)
        self.schedule_mode = schedule_mode
        self.p_start = float(p_start)
        self.p_end = float(p_end)
        self.constant_rate = float(constant_rate)
        self.random_rate_bounds = random_rate_bounds
        self.skip_coarsest = skip_coarsest

    def get_corruption_rates(self) -> List[float]:
        """
        Computes the target corruption probability p_corrupt(k) for each scale k in [0, K-1].
        """
        if self.schedule_mode == "disabled":
            return [0.0] * self.num_levels

        if self.schedule_mode == "constant":
            rates = [self.constant_rate] * self.num_levels
        elif self.schedule_mode == "random_uniform":
            low, high = self.random_rate_bounds
            rate = float(torch.empty(1).uniform_(low, high).item())
            rates = [rate] * self.num_levels
        elif self.schedule_mode == "linear_decay":
            if self.num_levels == 1:
                rates = [self.p_start]
            else:
                rates = []
                for k in range(self.num_levels):
                    alpha = k / (self.num_levels - 1)
                    p_k = self.p_start + alpha * (self.p_end - self.p_start)
                    rates.append(p_k)
        else:
            raise ValueError(f"Unknown schedule_mode: {self.schedule_mode}")

        if self.skip_coarsest and len(rates) > 0:
            rates[0] = 0.0

        return rates

    def corrupt_codes(
        self,
        clean_codes: List[torch.Tensor],
        rates: Optional[List[float]] = None,
    ) -> Tuple[List[torch.Tensor], List[float]]:
        """
        Applies stochastic uniform replacement to a list of discrete code tensors.

        Args:
            clean_codes: List of K tensors [r_1, ..., r_K], shape (B, l_k).
            rates: Optional precomputed corruption rates per scale.

        Returns:
            corrupted_codes: List of K tensors [r_tilde_1, ..., r_tilde_K].
            empirical_rates: List of observed corruption fractions per scale.
        """
        if rates is None:
            rates = self.get_corruption_rates()

        corrupted_codes = []
        empirical_rates = []

        for k, codes_k in enumerate(clean_codes):
            p_k = rates[k]
            if p_k <= 0.0:
                corrupted_codes.append(codes_k.clone())
                empirical_rates.append(0.0)
                continue

            B, l_k = codes_k.shape
            vocab_size = self.codebook_vocab_sizes[k]
            device = codes_k.device

            # Sample Bernoulli corruption mask
            corrupt_mask = torch.rand((B, l_k), device=device) < p_k

            # Sample uniform random discrete indices from {0, ..., vocab_size - 1}
            random_indices = torch.randint(
                low=0,
                high=vocab_size,
                size=(B, l_k),
                device=device,
                dtype=codes_k.dtype,
            )

            corrupted_k = torch.where(corrupt_mask, random_indices, codes_k)
            corrupted_codes.append(corrupted_k)

            actual_rate = (corrupt_mask.float().mean().item()) if corrupt_mask.numel() > 0 else 0.0
            empirical_rates.append(actual_rate)

        return corrupted_codes, empirical_rates

    @torch.no_grad()
    def build_corrupted_conditioning(
        self,
        tokenizer: nn.Module,
        clean_codes: List[torch.Tensor],
    ) -> Dict[str, Any]:
        """
        Constructs cumulative continuous conditioning reconstructions from corrupted codes:
            z_tilde^(0) = 0 in R^{B x L x d}
            z_tilde^(k-1) = sum_{j=1}^{k-1} R_j(e_j(r_tilde_j))

        Args:
            tokenizer: Frozen Stage-1 MultiscaleVQVAE model.
            clean_codes: List of K ground-truth code tensors [r_1^*, ..., r_K^*].

        Returns:
            Dictionary containing:
                - corrupted_conditioning: List of K cumulative continuous tensors [z_tilde^(0), ..., z_tilde^(K-1)]
                - clean_targets: clean_codes unchanged!
                - corrupted_codes: List of corrupted code tensors
                - corruption_rates: List of float empirical corruption rates per level
        """
        B = clean_codes[0].shape[0]
        L = tokenizer.levels[-1]
        D = tokenizer.embed_dim
        device = clean_codes[0].device

        # Corrupt discrete codes
        corrupted_codes, emp_rates = self.corrupt_codes(clean_codes)

        # Build cumulative continuous latents
        # z_tilde^(0) is zero (coarse prior)
        z_accum = torch.zeros(B, L, D, device=device, dtype=torch.float32)
        multiscale_conditioning = [z_accum.clone()]

        # For level k >= 2 (index 1..K-1), condition on accumulated reconstructions of levels 0..k-1
        for j in range(self.num_levels - 1):
            code_j = corrupted_codes[j]
            # Embed via Stage-1 codebook
            q_j = tokenizer.quantizer.codebooks[j](code_j)
            # Upsample / reconstruct to full length L
            reconstructed_q_j = tokenizer.quantizer.reconstructor(q_j, level_idx=j)
            z_accum = z_accum + reconstructed_q_j
            multiscale_conditioning.append(z_accum.clone())

        return {
            "corrupted_conditioning": multiscale_conditioning,
            "clean_targets": clean_codes,
            "corrupted_codes": corrupted_codes,
            "corruption_rates": emp_rates,
        }
