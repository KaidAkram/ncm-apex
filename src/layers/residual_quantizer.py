"""
NCM-Apex: Multiscale Residual Vector Quantizer (RVQ)
Implements hierarchical discrete codebook quantization with:
1. Straight-Through Estimation (STE)
2. Exponential Moving Average (EMA) codebook updates with Laplace smoothing
3. Dead codebook cluster restarting (anti-collapse mechanism)
4. Nested Scale Dropout for forcing hierarchical coarse-to-fine representations
5. Comprehensive codebook utilization & perplexity tracking
"""

import math
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.layers.downsampling import MultiscaleDownsampler, MultiscaleReconstructor


class MultiscaleResidualVectorQuantizer(nn.Module):
    """
    Multiscale Residual Vector Quantizer (RVQ).
    Factorizes a continuous latent sequence z in R^{L x D} into K discrete code sequences
    r_1, r_2, ..., r_K of varying lengths l_1 < l_2 < ... < l_K = L.
    """

    def __init__(
        self,
        vocab_sizes: List[int],
        embed_dim: int,
        levels: List[int],
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        restart_threshold: float = 1.0,
        fine_drop_prob: float = 0.5,
        fine_drop_min_keep: int = 1,
        learned_l1_sampling: bool = True,
        blend_ratio: float = 0.5,
    ):
        super().__init__()
        assert len(vocab_sizes) == len(levels), (
            f"Vocab sizes ({len(vocab_sizes)}) must match levels ({len(levels)})"
        )
        self.vocab_sizes = list(vocab_sizes)
        self.embed_dim = embed_dim
        self.levels = list(levels)
        self.num_levels = len(levels)
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        self.restart_threshold = restart_threshold
        self.fine_drop_prob = fine_drop_prob
        self.fine_drop_min_keep = fine_drop_min_keep

        # Codebook embeddings per level
        self.codebooks = nn.ModuleList([
            nn.Embedding(vs, embed_dim) for vs in self.vocab_sizes
        ])
        for emb in self.codebooks:
            emb.weight.data.normal_(0.0, 1.0 / math.sqrt(embed_dim))
            emb.weight.requires_grad_(False)  # Updated via EMA, not SGD

        # Buffers for EMA tracking & codebook hit analytics
        for i, vs in enumerate(self.vocab_sizes):
            self.register_buffer(f"ema_cluster_size_{i}", torch.zeros(vs, dtype=torch.float32))
            self.register_buffer(f"ema_w_{i}", self.codebooks[i].weight.data.clone())
            self.register_buffer(f"codebook_hits_{i}", torch.zeros(vs, dtype=torch.long))

        # Downsamplers and Reconstructors
        self.downsampler = MultiscaleDownsampler(
            embed_dim=embed_dim,
            levels=self.levels,
            learned_l1=learned_l1_sampling,
        )
        self.reconstructor = MultiscaleReconstructor(
            embed_dim=embed_dim,
            levels=self.levels,
            learned_l1=learned_l1_sampling,
            blend_ratio=blend_ratio,
        )

    def _compute_distances(self, x_flat: torch.Tensor, codebook_w: torch.Tensor) -> torch.Tensor:
        """
        Computes pairwise squared Euclidean distances: ||x - e||^2 = ||x||^2 + ||e||^2 - 2 x e^T.
        x_flat: (N, D), codebook_w: (V, D) -> returns (N, V)
        """
        x_sq = torch.sum(x_flat**2, dim=-1, keepdim=True)
        e_sq = torch.sum(codebook_w**2, dim=-1)
        cross = torch.matmul(x_flat, codebook_w.t())
        return x_sq + e_sq - 2.0 * cross

    @torch.no_grad()
    def _restart_dead_codes(
        self,
        inputs_flat: torch.Tensor,
        ema_cluster: torch.Tensor,
        ema_w: torch.Tensor,
        codebook: nn.Embedding,
    ) -> int:
        """
        Reinitializes codebook centroids that have cluster counts below restart_threshold.
        Prevents codebook collapse.
        """
        dead_mask = ema_cluster < self.restart_threshold
        num_dead = int(dead_mask.sum().item())
        if num_dead == 0 or inputs_flat.shape[0] == 0:
            return 0

        # Sample random vectors from current batch inputs
        rand_idx = torch.randint(0, inputs_flat.shape[0], (num_dead,), device=inputs_flat.device)
        replacement_vectors = inputs_flat[rand_idx].to(codebook.weight.dtype)

        codebook.weight.data[dead_mask] = replacement_vectors
        ema_w[dead_mask] = replacement_vectors
        ema_cluster[dead_mask] = self.restart_threshold
        return num_dead

    def forward(
        self,
        z: torch.Tensor,
        force_cutoff_level: Optional[int] = None,
    ) -> Dict[str, Union[torch.Tensor, List[torch.Tensor], float]]:
        """
        Hierarchical Residual Vector Quantization Forward Pass.
        Args:
            z: Continuous latent representation (B, L, D) from Encoder.
            force_cutoff_level: Optional integer in [1, K] for deterministic truncation.
        Returns:
            Dictionary containing:
                - z_hat: Final reconstructed latent (B, L, D) with STE gradient flow.
                - vq_loss: Combined commitment & codebook alignment loss.
                - codes: List of discrete code index tensors [ (B, l_1), ..., (B, l_K) ].
                - multiscale_z_hat: List of cumulative reconstructions up to each scale.
                - per_level_utilization: List of float utilization percentages.
                - global_utilization: Overall unique codebook utilization float.
                - codebook_perplexities: List of perplexity scores per level.
                - cutoff_level: The scale index used for decoder reconstruction.
        """
        z_float = z.float()
        B, L, D = z_float.shape
        z_no_grad = z_float.detach()

        z_residual = z_no_grad.clone()
        z_hat_accumulated = torch.zeros_like(z_residual)

        multiscale_z_hat: List[torch.Tensor] = []
        codes: List[torch.Tensor] = []
        quantized_residuals: List[torch.Tensor] = []

        total_vq_loss = torch.tensor(0.0, device=z.device, dtype=torch.float32)

        for i, target_len in enumerate(self.levels):
            codebook_i = self.codebooks[i]
            vocab_size_i = self.vocab_sizes[i]

            # 1. Downsample current continuous residual to level resolution
            residual_down = self.downsampler(z_residual, level_idx=i)  # (B, l_i, D)
            flat_input = residual_down.reshape(-1, D)  # (B * l_i, D)

            # 2. Nearest codebook vector lookup
            distances = self._compute_distances(flat_input, codebook_i.weight.data)
            indices_flat = torch.argmin(distances, dim=-1)  # (B * l_i,)
            indices = indices_flat.view(B, target_len)
            codes.append(indices)

            # 3. EMA Codebook Updates (during training)
            if self.training:
                ema_cluster = getattr(self, f"ema_cluster_size_{i}")
                ema_w = getattr(self, f"ema_w_{i}")

                # One-hot encodings for cluster assignment
                encodings = torch.zeros(B * target_len, vocab_size_i, device=z.device)
                encodings.scatter_(1, indices_flat.unsqueeze(1), 1.0)

                cluster_counts = encodings.sum(dim=0)  # (V_i,)
                dw = torch.matmul(encodings.t(), flat_input.detach())  # (V_i, D)

                # Exponential Moving Average update
                ema_cluster.mul_(self.decay).add_(cluster_counts, alpha=1.0 - self.decay)
                total_cluster_sum = ema_cluster.sum()

                # Laplace smoothing
                smoothed_cluster = (
                    (ema_cluster + self.epsilon)
                    / (total_cluster_sum + vocab_size_i * self.epsilon)
                    * total_cluster_sum
                )

                ema_w.mul_(self.decay).add_(dw, alpha=1.0 - self.decay)
                codebook_i.weight.data.copy_(ema_w / smoothed_cluster.unsqueeze(1))

                # Anti-collapse restart of dead codes
                if self.restart_threshold > 0:
                    self._restart_dead_codes(flat_input.detach(), ema_cluster, ema_w, codebook_i)

            # Record codebook hits
            getattr(self, f"codebook_hits_{i}")[indices_flat] = 1

            # 4. Lookup Quantized Embeddings
            q_level = codebook_i(indices)  # (B, l_i, D)
            quantized_residuals.append(q_level)

            # 5. Spatial Reconstruction (Upsample to length L) & Phi Conv Blend
            reconstructed_q = self.reconstructor(q_level, level_idx=i)  # (B, L, D)

            # 6. Update Cumulative Reconstruction & Residual
            z_hat_accumulated = z_hat_accumulated + reconstructed_q
            multiscale_z_hat.append(z_hat_accumulated.clone())
            z_residual = z_residual - reconstructed_q

            # 7. Compute Residual Quantization Losses at level i
            # Commitment loss: pushes encoder latents towards codebook vectors
            commit_loss = F.mse_loss(z_hat_accumulated.detach(), z_float)
            # Alignment loss: pulls codebook vectors towards encoder latents
            codebook_align_loss = F.mse_loss(z_hat_accumulated, z_no_grad)
            total_vq_loss = total_vq_loss + self.commitment_cost * commit_loss + codebook_align_loss

        total_vq_loss = total_vq_loss / self.num_levels

        # 8. Nested Scale Dropout Implementation
        cutoff_level = self.num_levels
        if force_cutoff_level is not None:
            cutoff_level = min(max(1, force_cutoff_level), self.num_levels)
            z_hat_selected = multiscale_z_hat[cutoff_level - 1]
        elif self.training and self.fine_drop_prob > 0.0:
            if float(torch.rand(())) < self.fine_drop_prob and len(multiscale_z_hat) > 1:
                min_k = max(1, self.fine_drop_min_keep)
                # Sample random truncation level k in [min_k, K]
                cutoff_level = int(torch.randint(min_k, self.num_levels + 1, ()))
                z_hat_selected = multiscale_z_hat[cutoff_level - 1]
            else:
                z_hat_selected = multiscale_z_hat[-1]
        else:
            # During evaluation, use full fine reconstruction by default
            z_hat_selected = multiscale_z_hat[-1]

        # 9. Master Straight-Through Estimator (STE)
        # Allows gradients from decoder to propagate back into continuous encoder latents z
        z_hat_ste = z_float + (z_hat_selected - z_float).detach()

        # 10. Diagnostics Telemetry
        per_level_utilization = []
        codebook_perplexities = []
        total_used_slots = 0
        total_slots = sum(self.vocab_sizes)

        for i, vs in enumerate(self.vocab_sizes):
            hits = getattr(self, f"codebook_hits_{i}")
            used = int(torch.sum(hits).item())
            total_used_slots += used
            util_rate = (used / vs) * 100.0
            per_level_utilization.append(util_rate)

            # Codebook perplexity from cluster probabilities
            ema_cs = getattr(self, f"ema_cluster_size_{i}")
            total_count = ema_cs.sum().item()
            if total_count > 0:
                probs = (ema_cs / total_count).clamp(min=1e-12)
                entropy = -(probs * torch.log(probs)).sum().item()
                perplexity = math.exp(entropy)
            else:
                perplexity = 1.0
            codebook_perplexities.append(perplexity)

        global_utilization = (total_used_slots / total_slots) * 100.0

        return {
            "z_hat": z_hat_ste,
            "vq_loss": total_vq_loss,
            "codes": codes,
            "multiscale_z_hat": multiscale_z_hat,
            "per_level_utilization": per_level_utilization,
            "global_utilization": global_utilization,
            "codebook_perplexities": codebook_perplexities,
            "cutoff_level": cutoff_level,
        }

    @torch.no_grad()
    def codes_to_z_q(self, codes: List[torch.Tensor]) -> torch.Tensor:
        """
        Reconstructs the continuous latent representation from a list of code indices.
        codes: List of (B, l_k) tensors -> returns (B, L, D).
        """
        B = codes[0].shape[0]
        L = self.levels[-1]
        z_hat = torch.zeros(B, L, self.embed_dim, device=codes[0].device, dtype=torch.float32)

        for i, code_level in enumerate(codes):
            q_level = self.codebooks[i](code_level)
            reconstructed_q = self.reconstructor(q_level, level_idx=i)
            z_hat = z_hat + reconstructed_q

        return z_hat

    @torch.no_grad()
    def get_multiscale_reconstructions_from_codes(
        self, codes: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Returns cumulative continuous reconstructions [z^(1), z^(2), ..., z^(K)].
        """
        B = codes[0].shape[0]
        L = self.levels[-1]
        z_hat = torch.zeros(B, L, self.embed_dim, device=codes[0].device, dtype=torch.float32)
        multiscale_z = []

        for i, code_level in enumerate(codes):
            q_level = self.codebooks[i](code_level)
            reconstructed_q = self.reconstructor(q_level, level_idx=i)
            z_hat = z_hat + reconstructed_q
            multiscale_z.append(z_hat.clone())

        return multiscale_z
