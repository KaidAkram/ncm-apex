"""
NCM-Apex Extension 8C: Confidence-Gated Residual Copy Head
Implements Paper Limitation 3 (arXiv:2610.14162v1, Section I & Table 11):
1. Prompt Attention Entropy Detector:
   - Given cross-attention matrix A in R^{B x l_k x P} from target to prompt tokens:
     H_attn(i) = - sum_{j=1}^P A_{i, j} * log(A_{i, j} + eps)
   - Sharp attention (H_attn < tau_copy) triggers the copy-pointer gate.
2. Residual Copy-Gate Mechanism:
   - Gate scalar: g_i = sigmoid(W_gate * h_i + b_gate) in [0, 1]
   - Final probability:
     p_final(w) = (1 - g_i) * p_codebook(w) + g_i * sum_{j=1}^P A_{i, j} * I[x_prompt^j == w]
3. Eliminates numeral drift ("1957" -> "1956") and proper noun substitution ("Huawei" -> "BMW")
   caused by discrete codebook bottleneck compression.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ConfidenceGatedCopyHead(nn.Module):
    """
    Residual Pointer-Generator Copy Mechanism for Next-Concept Modeling.
    Monitors cross-attention entropy over prompt tokens to detect sharp focus
    on rare entities, numerals, and dates, directly routing source tokens
    to circumvent the discrete VQ information bottleneck.
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        tau_copy: float = 0.60,
    ):
        """
        Args:
            d_model: Transformer hidden representation dimension d.
            vocab_size: Total vocabulary size |T| (e.g. 8192 or 32000).
            tau_copy: Attention entropy threshold (in nats) below which copying is activated.
        """
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.tau_copy = tau_copy

        # Linear projection computing the scalar copy gate g in [0, 1]
        self.gate_proj = nn.Linear(d_model, 1, bias=True)

        # Initialize bias slightly negative so model defaults to codebook generation
        # unless triggered by sharp attention focus
        nn.init.constant_(self.gate_proj.bias, -1.0)
        nn.init.xavier_normal_(self.gate_proj.weight)

    def compute_attention_entropy(
        self,
        attn_weights: torch.Tensor,
        eps: float = 1e-9,
    ) -> torch.Tensor:
        """
        Computes attention entropy across prompt positions P:
            H_attn(i) = - sum_{j=1}^P A_{i, j} * log(A_{i, j} + eps)

        Args:
            attn_weights: Tensor of shape (B, L, P) normalized so sum_j A_{i, j} == 1.0.

        Returns:
            entropy: Tensor of shape (B, L) representing uncertainty over prompt tokens.
        """
        clipped_attn = torch.clamp(attn_weights, min=eps, max=1.0)
        entropy = -torch.sum(clipped_attn * torch.log(clipped_attn), dim=-1)
        return entropy

    def forward(
        self,
        hidden_states: torch.Tensor,
        codebook_logits: Optional[torch.Tensor] = None,
        prompt_tokens: Optional[torch.Tensor] = None,
        cross_attention_weights: Optional[torch.Tensor] = None,
        force_copy_entropy_threshold: Optional[float] = None,
        discrete_logits: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Computes gated mixture of codebook generation and prompt copy probabilities.

        Args:
            hidden_states: Final layer representations h in R^{B x L x d}.
            codebook_logits: Stage-1 decoder logits in R^{B x L x V}.
            prompt_tokens: Source prompt token IDs in R^{B x P}.
            cross_attention_weights: Attention matrix A in R^{B x L x P}.
            force_copy_entropy_threshold: Optional override for tau_copy.
            discrete_logits: Optional alias for codebook_logits.

        Returns:
            Dictionary containing:
                - final_probs: Blended distribution in R^{B x L x V}
                - final_logits: Log-probabilities of final distribution
                - copy_gate: Gate values g_i in R^{B x L}
                - gate_values: Alias for copy_gate
                - attention_entropy: Calculated entropy values in R^{B x L}
                - copy_active_mask: Boolean mask of positions where copy gate fired
        """
        if codebook_logits is None and discrete_logits is not None:
            codebook_logits = discrete_logits

        assert codebook_logits is not None, "Must provide codebook_logits or discrete_logits"
        assert prompt_tokens is not None, "Must provide prompt_tokens"
        assert cross_attention_weights is not None, "Must provide cross_attention_weights"

        B, L, d = hidden_states.shape
        P = prompt_tokens.shape[-1]
        V = codebook_logits.shape[-1]
        dev = hidden_states.device

        tau = self.tau_copy if force_copy_entropy_threshold is None else force_copy_entropy_threshold

        # 1. Base codebook generation probability: p_codebook
        p_codebook = F.softmax(codebook_logits, dim=-1)  # (B, L, V)

        # 2. Compute cross-attention entropy over prompt positions
        # cross_attention_weights: (B, L, P)
        attn_entropy = self.compute_attention_entropy(cross_attention_weights)  # (B, L)

        # 3. Compute raw gating probability: g_raw in [0, 1]
        g_raw = torch.sigmoid(self.gate_proj(hidden_states)).squeeze(-1)  # (B, L)

        # 4. Modulate gate by low-entropy confidence:
        # Sharp attention (H < tau) activates copy routing; diffuse attention disables it
        is_confident = (attn_entropy < tau).float()
        copy_gate = g_raw * is_confident  # (B, L)

        # 5. Build direct copy probability distribution from prompt tokens:
        # p_copy(w) = sum_{j=1}^P A_{i, j} * I[prompt[j] == w]
        p_copy = torch.zeros(B, L, V, device=dev, dtype=p_codebook.dtype)
        # Expand prompt tokens for scattering: (B, 1, P) -> (B, L, P)
        prompt_expanded = prompt_tokens.unsqueeze(1).expand(-1, L, -1)
        # Clamp prompt tokens within vocab bounds
        prompt_clamped = prompt_expanded.clamp(min=0, max=V - 1)

        # Scatter attention weights to corresponding token indices in vocab
        p_copy.scatter_add_(dim=-1, index=prompt_clamped, src=cross_attention_weights)

        # 6. Residual Blending:
        # p_final = (1 - g) * p_codebook + g * p_copy
        g_expanded = copy_gate.unsqueeze(-1)  # (B, L, 1)
        final_probs = (1.0 - g_expanded) * p_codebook + g_expanded * p_copy

        # Safe log-probabilities
        final_logits = torch.log(torch.clamp(final_probs, min=1e-12))

        return {
            "final_probs": final_probs,
            "final_logits": final_logits,
            "copy_gate": copy_gate,
            "gate_values": copy_gate,
            "attention_entropy": attn_entropy,
            "copy_active_mask": copy_gate > 0.0,
        }
