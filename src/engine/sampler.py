"""
NCM-Apex: Parallel Coarse-to-Fine Generation & Sampling Engine
Implements Paper Algorithm 2:
1. Parallel coarse-to-fine generation across K abstraction scales
2. Fast single-step forward passes through NCM Transformer Prior (O(l_k^2))
3. Parallel multinomial/nucleus (top-k, top-p) code sampling across all l_k positions
4. Cumulative latent crystallization: z_hat^(k) = z_hat^(k-1) + R_k(e_k(r_k))
5. Step-by-step latency & compute telemetry (verifying coarse scale speed advantages)
"""

import time
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


def sample_discrete_codes(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> torch.Tensor:
    """
    Samples discrete codes in parallel across sequence length l_k.

    Args:
        logits: Logits tensor in R^{B x l_k x VocabSize}.
        temperature: Sampling temperature (<= 1e-5 or 0.0 implies greedy argmax).
        top_k: Top-k filtering threshold (0 disables).
        top_p: Nucleus top-p filtering threshold (1.0 disables).

    Returns:
        samples: Sampled discrete indices in R^{B x l_k}.
    """
    if temperature <= 1e-5:
        return torch.argmax(logits, dim=-1)

    B, L, V = logits.shape
    logits_flat = (logits / temperature).reshape(-1, V)

    # Top-K filtering
    if top_k > 0 and top_k < V:
        topk_vals, _ = torch.topk(logits_flat, min(top_k, V), dim=-1)
        kth_threshold = topk_vals[:, -1, None]
        logits_flat = torch.where(
            logits_flat < kth_threshold,
            torch.tensor(float("-inf"), device=logits.device, dtype=logits.dtype),
            logits_flat,
        )

    # Top-P (Nucleus) filtering
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits_flat, descending=True, dim=-1)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Remove tokens with cumulative probability above top_p (shift by 1 to keep first token above threshold)
        sorted_indices_to_remove = cumulative_probs - sorted_probs > top_p
        sorted_logits = torch.where(
            sorted_indices_to_remove,
            torch.tensor(float("-inf"), device=logits.device, dtype=logits.dtype),
            sorted_logits,
        )
        logits_flat = torch.zeros_like(logits_flat).scatter_(-1, sorted_indices, sorted_logits)

    probs = F.softmax(logits_flat, dim=-1)
    samples = torch.multinomial(probs, num_samples=1).view(B, L)
    return samples


def estimate_transformer_step_flops(
    d_model: int,
    n_layer: int,
    l_k: int,
    batch_size: int = 1,
) -> float:
    """
    Analytically estimates floating point operations (FLOPs) for one forward pass of scale k:
    - Multi-Head Attention QKV proj: 2 * 3 * d_model^2 * l_k
    - Scaled dot-product attention: 2 * (d_model * l_k^2 + d_model * l_k^2) = 4 * d_model * l_k^2
    - Out projection: 2 * d_model^2 * l_k
    - MLP (expand 4x and project back): 2 * 2 * 4 * d_model^2 * l_k = 16 * d_model^2 * l_k
    Total per layer: ~ 24 * d_model^2 * l_k + 4 * d_model * l_k^2.
    """
    flops_per_layer = (24.0 * (d_model ** 2) * l_k) + (4.0 * d_model * (l_k ** 2))
    total_flops = float(batch_size * n_layer * flops_per_layer)
    return total_flops


class NCMSampler:
    """
    Coarse-to-Fine Parallel Inference & Sampling Engine.

    Executes generation in exactly K sequential steps (one per abstraction level).
    At step k:
        1. Pool accumulated continuous latent z_hat^(k-1) to length l_k.
        2. Execute single forward evaluation of NCM Transformer at scale k.
        3. Predict and sample all l_k codes in parallel.
        4. Accumulate quantized residual: z_hat^(k) = z_hat^(k-1) + R_k(e_k(r_k)).
    Final:
        Pass z_hat^(K) through frozen Stage-1 Tokenizer Decoder to yield text tokens.
    """

    def __init__(
        self,
        transformer: nn.Module,
        tokenizer: nn.Module,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.transformer = transformer
        self.tokenizer = tokenizer
        self.device = device or next(transformer.parameters()).device
        self.levels = transformer.levels
        self.num_levels = len(self.levels)
        self.L = tokenizer.levels[-1]
        self.embed_dim = tokenizer.embed_dim
        self.d_model = transformer.d_model
        self.n_layer = transformer.n_layer

    @torch.no_grad()
    def generate(
        self,
        batch_size: int = 1,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 1.0,
        prefix_tokens: Optional[torch.Tensor] = None,
        return_trajectory: bool = True,
    ) -> Dict[str, Any]:
        """
        Executes Paper Algorithm 2: Parallel coarse-to-fine sampling.

        Args:
            batch_size: Number of parallel sequences to generate.
            temperature: Sampling temperature (0.0 = greedy argmax).
            top_k: Top-k filtering.
            top_p: Nucleus filtering.
            prefix_tokens: Optional prompt tokens tensor (B, P).
            return_trajectory: If True, captures intermediate text reconstructions per level.

        Returns:
            Dictionary with:
                - text_tokens: Final predicted token IDs in R^{B x L}
                - codes: List of K discrete code tensors [r_1, ..., r_K]
                - final_z_hat: Cumulative continuous latent R^{B x L x d}
                - step_telemetry: List of step timing, FLOPs, and compute metrics
                - trajectory_tokens: List of intermediate decodings if return_trajectory is True
                - total_time_ms: Total sampling wall-clock time in milliseconds
        """
        self.transformer.eval()
        self.tokenizer.eval()

        dev = self.device
        B = batch_size
        L = self.L
        D = self.embed_dim

        # Step 0: Initialize empty prior conditioning
        z_hat = torch.zeros(B, L, D, device=dev, dtype=torch.float32)

        # Prefix representation if supplied
        prefix_emb = None
        if prefix_tokens is not None:
            prefix_tokens = prefix_tokens.to(dev)
            prefix_z = self.tokenizer.encode_continuous(prefix_tokens)
            prefix_emb = self.transformer.cond_proj(prefix_z)

        generated_codes = []
        step_telemetry = []
        trajectory_tokens = []

        total_start_time = time.perf_counter()

        # Iterate coarse-to-fine through all K levels
        for k in range(self.num_levels):
            l_k = self.levels[k]

            # Timing & CUDA synchronization
            if "cuda" in str(dev) and torch.cuda.is_available():
                torch.cuda.synchronize(dev)
            step_start = time.perf_counter()

            # 1. Parallel forward evaluation for scale k
            logits_k = self.transformer.forward_single_level(
                z_hat_prev=z_hat,
                level_idx=k,
                prefix_emb=prefix_emb,
            )  # (B, l_k, |V_k|)

            # 2. Parallel code sampling across all l_k positions simultaneously
            codes_k = sample_discrete_codes(
                logits=logits_k,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )  # (B, l_k)
            generated_codes.append(codes_k)

            # 3. Codebook embedding lookup & upsampling
            q_k = self.tokenizer.quantizer.codebooks[k](codes_k)  # (B, l_k, d)
            reconstructed_q_k = self.tokenizer.quantizer.reconstructor(q_k, level_idx=k)  # (B, L, d)

            # 4. Cumulative latent update
            z_hat = z_hat + reconstructed_q_k

            if "cuda" in str(dev) and torch.cuda.is_available():
                torch.cuda.synchronize(dev)
            step_end = time.perf_counter()
            step_ms = (step_end - step_start) * 1000.0

            # 5. Compute FLOPs & metrics
            step_flops = estimate_transformer_step_flops(
                d_model=self.d_model,
                n_layer=self.n_layer,
                l_k=l_k,
                batch_size=B,
            )
            gflops = step_flops / 1e9

            step_telemetry.append({
                "scale_level": k + 1,
                "token_length": l_k,
                "step_latency_ms": step_ms,
                "est_gflops": gflops,
                "latent_norm": float(z_hat.norm().item()),
            })

            # Intermediate text decodability (crystallization progression)
            if return_trajectory:
                interm_logits = self.tokenizer.decode_latent(z_hat)
                interm_tokens = torch.argmax(interm_logits, dim=-1)
                trajectory_tokens.append(interm_tokens.cpu())

        total_end_time = time.perf_counter()
        total_time_ms = (total_end_time - total_start_time) * 1000.0

        # Final Step: Pass completed continuous representation through frozen Tokenizer Decoder
        final_logits = self.tokenizer.decode_latent(z_hat)
        final_tokens = torch.argmax(final_logits, dim=-1)

        return {
            "text_tokens": final_tokens,
            "codes": generated_codes,
            "final_z_hat": z_hat,
            "step_telemetry": step_telemetry,
            "trajectory_tokens": trajectory_tokens,
            "total_time_ms": total_time_ms,
        }
