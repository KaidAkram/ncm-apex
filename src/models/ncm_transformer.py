"""
NCM-Apex: Next-Concept Transformer Prior Architecture
Implements the Stage-2 coarse-to-fine generative Transformer prior:
1. Conditioning Projection & Adaptive Pooling from Stage-1 Latents to Hidden Dimension
2. Pre-LN / RMSNorm Transformer Backbone with PyTorch SDPA and QK Normalization
3. Block-Diagonal Multi-Head Self-Attention for single-pass training over all scales
4. Multiscale Discrete Prediction Heads mapping hidden states to codebook logits
5. Dual execution modes: Single-Pass All-Levels Training & Per-Level Inference
"""

import math
from typing import Dict, List, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.layers.block_attention import (
    build_block_diagonal_mask,
    build_block_causal_mask,
    prepare_sdpa_mask,
)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_f = x.float()
        rms = x_f.pow(2).mean(dim=-1, keepdim=True).add_(self.eps).rsqrt_()
        return (x_f * rms).to(orig_dtype) * self.weight


def precompute_rope_freqs(dim: int, max_seq_len: int = 2048, theta: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precomputes rotary position embedding cosine and sine tables."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, freqs)
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Applies rotary position embedding to tensor x of shape (B, H, L, head_dim).
    cos, sin: (L, head_dim // 2)
    """
    B, H, L, D = x.shape
    cos = cos[:L, :].unsqueeze(0).unsqueeze(0)  # (1, 1, L, D//2)
    sin = sin[:L, :].unsqueeze(0).unsqueeze(0)  # (1, 1, L, D//2)

    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    rx1 = x1 * cos - x2 * sin
    rx2 = x1 * sin + x2 * cos
    return torch.stack([rx1, rx2], dim=-1).flatten(-2)


class TransformerBlock(nn.Module):
    """
    Pre-LN Transformer Block with RMSNorm, QK-Norm, and PyTorch SDPA.
    """

    def __init__(
        self,
        d_model: int,
        n_head: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        assert d_model % n_head == 0, f"d_model {d_model} must be divisible by n_head {n_head}"
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.dropout = dropout

        # Pre-attention norm
        self.attn_norm = RMSNorm(d_model)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        # Pre-MLP norm
        self.mlp_norm = RMSNorm(d_model)
        mlp_hidden_dim = int(d_model * mlp_ratio)
        self.mlp_in = nn.Linear(d_model, mlp_hidden_dim, bias=bias)
        self.mlp_out = nn.Linear(mlp_hidden_dim, d_model, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        rope_cos_sin: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        B, L, D = x.shape

        # 1. Attention sub-layer
        h = self.attn_norm(x)
        qkv = self.qkv_proj(h).view(B, L, 3, self.n_head, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # (B, L, H, head_dim)

        # Transpose to (B, H, L, head_dim)
        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)

        # Optional RoPE
        if rope_cos_sin is not None:
            cos, sin = rope_cos_sin
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        # PyTorch SDPA (native FlashAttention / Cutlass / Memory-Efficient)
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, D)
        x = x + self.out_proj(attn_out)

        # 2. Feed-forward sub-layer (GeLU)
        mlp_h = self.mlp_norm(x)
        mlp_h = self.mlp_out(F.gelu(self.mlp_in(mlp_h)))
        x = x + mlp_h

        return x


class NCMTransformer(nn.Module):
    """
    Next-Concept Modeling (NCM) Transformer Prior.

    Models the joint coarse-to-fine discrete distribution:
        p_theta(r_{1:K}) = prod_{k=1}^K p_theta(r_k | r_{<k})
    where all tokens r_k within level k are generated conditionally independent
    in parallel in a single forward pass given accumulated conditioning z_hat^(k-1).
    """

    def __init__(
        self,
        levels: List[int] = [16, 32, 64],
        codebook_vocab_sizes: List[int] = [512, 512, 512],
        tokenizer_latent_dim: int = 128,
        d_model: int = 384,
        n_layer: int = 6,
        n_head: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        use_rope: bool = False,
        use_level_embedding: bool = True,
        shared_heads: bool = False,
        bias: bool = False,
    ):
        super().__init__()
        assert len(levels) == len(codebook_vocab_sizes), "levels and codebook_vocab_sizes length mismatch"
        self.levels = list(levels)
        self.num_levels = len(levels)
        self.codebook_vocab_sizes = list(codebook_vocab_sizes)
        self.tokenizer_latent_dim = tokenizer_latent_dim
        self.d_model = d_model
        self.n_layer = n_layer
        self.n_head = n_head
        self.use_rope = use_rope
        self.use_level_embedding = use_level_embedding
        self.shared_heads = shared_heads
        self.total_target_length = sum(levels)

        # Conditioning input projection: maps Stage-1 latent dim (d) to Transformer hidden dim (d_model)
        self.cond_proj = nn.Linear(tokenizer_latent_dim, d_model, bias=bias)

        # Learned base queries for level 1 (when prior conditioning z^(0) = 0)
        self.l1_base_queries = nn.Parameter(
            torch.zeros(1, self.levels[0], tokenizer_latent_dim)
        )
        nn.init.trunc_normal_(self.l1_base_queries, mean=0.0, std=0.02)

        # Positional Embeddings
        if self.use_rope:
            cos, sin = precompute_rope_freqs(d_model // n_head, max_seq_len=max(levels) + 128)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)
        else:
            # Per-level position embeddings up to maximum level length
            max_len = max(self.levels)
            self.pos_emb = nn.Embedding(max_len, d_model)
            nn.init.trunc_normal_(self.pos_emb.weight, mean=0.0, std=0.02)

        # Level embeddings to distinguish scale semantics
        if self.use_level_embedding:
            self.level_emb = nn.Embedding(self.num_levels, d_model)
            nn.init.trunc_normal_(self.level_emb.weight, mean=0.0, std=0.02)

        self.drop = nn.Dropout(dropout)

        # Transformer Blocks Stack
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_head=n_head,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                bias=bias,
            )
            for _ in range(n_layer)
        ])
        self.final_norm = RMSNorm(d_model)

        # Multiscale Prediction Heads
        if self.shared_heads:
            assert len(set(codebook_vocab_sizes)) == 1, "shared_heads requires identical vocab sizes across levels"
            head = nn.Linear(d_model, codebook_vocab_sizes[0], bias=bias)
            self.heads = nn.ModuleList([head for _ in range(self.num_levels)])
        else:
            self.heads = nn.ModuleList([
                nn.Linear(d_model, vs, bias=bias) for vs in codebook_vocab_sizes
            ])

        # Precompute and register default block-diagonal mask for training
        default_mask = build_block_diagonal_mask(levels=self.levels, prefix_len=0, return_4d=True)
        self.register_buffer("cached_block_diagonal_mask", default_mask, persistent=False)

    def pool_conditioning_to_level(self, z_hat: torch.Tensor, level_idx: int) -> torch.Tensor:
        """
        Resizes cumulative continuous reconstruction z_hat in R^{B x L x d} to sequence length l_k.
        Uses 1D area interpolation (exact continuous spatial average pooling).
        For level 0 (k=1), if z_hat is zero or None, injects learned l1_base_queries.
        """
        target_len = self.levels[level_idx]
        B = z_hat.shape[0]

        if level_idx == 0:
            # Coarsest level: inject learned prior query vectors
            base_q = self.l1_base_queries.expand(B, -1, -1)
            if z_hat.norm() < 1e-6:
                return base_q
            # If partial conditioning exists, add base queries
            pooled = F.interpolate(
                z_hat.transpose(-1, -2),
                size=(target_len,),
                mode="area",
            ).transpose(-1, -2)
            return pooled + base_q

        # Levels 1..K-1: pool full reconstruction to target_len l_k
        if z_hat.shape[1] == target_len:
            return z_hat

        pooled = F.interpolate(
            z_hat.transpose(-1, -2),
            size=(target_len,),
            mode="area",
        ).transpose(-1, -2)
        return pooled

    def prepare_multilevel_inputs(
        self,
        multiscale_conditioning: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Takes list of cumulative continuous reconstructions [z^(0), z^(1), ..., z^(K-1)]
        and builds the packed sequence of length N = sum l_k with projections, positional
        encodings, and level embeddings.

        Args:
            multiscale_conditioning: List of K tensors, each in R^{B x L x d} or R^{B x l_k x d}.
        Returns:
            packed_inputs: Tensor in R^{B x N x d_model}.
        """
        assert len(multiscale_conditioning) == self.num_levels, (
            f"Expected {self.num_levels} conditioning levels, got {len(multiscale_conditioning)}"
        )
        B = multiscale_conditioning[0].shape[0]
        dev = multiscale_conditioning[0].device

        level_embeddings_list = []
        for k in range(self.num_levels):
            z_cond_k = multiscale_conditioning[k]
            # Pool to l_k
            z_pooled = self.pool_conditioning_to_level(z_cond_k, level_idx=k)  # (B, l_k, d)

            # Project to d_model
            h_k = self.cond_proj(z_pooled)  # (B, l_k, d_model)

            # Add positional embeddings
            l_k = self.levels[k]
            if not self.use_rope:
                pos = torch.arange(l_k, dtype=torch.long, device=dev)
                h_k = h_k + self.pos_emb(pos)

            # Add level embedding
            if self.use_level_embedding:
                lvl = torch.tensor(k, dtype=torch.long, device=dev)
                h_k = h_k + self.level_emb(lvl)

            level_embeddings_list.append(h_k)

        # Concatenate all levels into single sequence of length N = sum l_k
        packed_inputs = torch.cat(level_embeddings_list, dim=1)  # (B, N, d_model)
        return self.drop(packed_inputs)

    def forward_single_pass_all_levels(
        self,
        multiscale_conditioning: List[torch.Tensor],
        prefix_emb: Optional[torch.Tensor] = None,
        custom_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Union[List[torch.Tensor], torch.Tensor]]:
        """
        Single-Pass Training Forward:
        Passes all K scales concatenated through the Transformer in ONE shot
        using the Block-Diagonal Attention Mask.

        Returns:
            - level_logits: List of K tensors [logits_1, ..., logits_K],
              where logits_k has shape (B, l_k, |V_k|).
            - logits_concat: Concatenated logits tensor across all levels.
            - hidden_states: Hidden representations before final prediction heads.
        """
        packed_x = self.prepare_multilevel_inputs(multiscale_conditioning)
        B, N, D = packed_x.shape

        if prefix_emb is not None:
            P = prefix_emb.shape[1]
            x_in = torch.cat([prefix_emb, packed_x], dim=1)
            mask = build_block_diagonal_mask(
                levels=self.levels, prefix_len=P, device=x_in.device, dtype=x_in.dtype
            )
        else:
            P = 0
            x_in = packed_x
            if custom_mask is not None:
                mask = custom_mask
            else:
                if self.cached_block_diagonal_mask.device != x_in.device:
                    self.cached_block_diagonal_mask = self.cached_block_diagonal_mask.to(x_in.device)
                mask = self.cached_block_diagonal_mask

        # Transformer blocks
        h = x_in
        rope = (self.rope_cos, self.rope_sin) if self.use_rope else None
        for block in self.blocks:
            h = block(h, attn_mask=mask, rope_cos_sin=rope)
        h = self.final_norm(h)

        # Slice target activations excluding prefix
        h_targets = h[:, P:, :]

        # Compute per-level logits
        level_logits = []
        offset = 0
        for k in range(self.num_levels):
            l_k = self.levels[k]
            h_k = h_targets[:, offset : offset + l_k, :]
            logits_k = self.heads[k](h_k)  # (B, l_k, |V_k|)
            level_logits.append(logits_k)
            offset += l_k

        return {
            "level_logits": level_logits,
            "hidden_states": h_targets,
        }

    def forward_single_level(
        self,
        z_hat_prev: torch.Tensor,
        level_idx: int,
        prefix_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Single-Level Inference Forward Pass:
        Evaluates only level k in isolation (O(l_k^2) attention compute).
        Takes cumulative continuous reconstruction z_hat^(k-1) and predicts
        all l_k codebook logits for level k in parallel!

        Args:
            z_hat_prev: Cumulative reconstruction R^{B x L x d}.
            level_idx: Level index k in [0, K-1].
            prefix_emb: Optional prompt conditioning embeddings R^{B x P x d_model}.

        Returns:
            logits_k: Codebook logits in R^{B x l_k x |V_k|}.
        """
        B = z_hat_prev.shape[0]
        dev = z_hat_prev.device
        l_k = self.levels[level_idx]

        # 1. Pool conditioning to length l_k
        z_pooled = self.pool_conditioning_to_level(z_hat_prev, level_idx=level_idx)
        h_k = self.cond_proj(z_pooled)

        # 2. Add position and level embeddings
        if not self.use_rope:
            pos = torch.arange(l_k, dtype=torch.long, device=dev)
            h_k = h_k + self.pos_emb(pos)
        if self.use_level_embedding:
            lvl = torch.tensor(level_idx, dtype=torch.long, device=dev)
            h_k = h_k + self.level_emb(lvl)

        # 3. Form input sequence with optional prefix
        if prefix_emb is not None:
            P = prefix_emb.shape[1]
            x_in = torch.cat([prefix_emb, h_k], dim=1)
            # Prefix mask: prompt attends to prompt, target attends to prompt + target, prompt cannot attend to target
            mask = torch.zeros((1, 1, P + l_k, P + l_k), dtype=h_k.dtype, device=dev)
            mask[:, :, :P, P:] = float("-inf")
        else:
            P = 0
            x_in = h_k
            mask = None  # Full bidirectional attention within level k

        # 4. Pass through Transformer stack
        h = x_in
        rope = (self.rope_cos, self.rope_sin) if self.use_rope else None
        for block in self.blocks:
            h = block(h, attn_mask=mask, rope_cos_sin=rope)
        h = self.final_norm(h)

        # 5. Extract target slice and project to logits
        h_target = h[:, P:, :]
        logits_k = self.heads[level_idx](h_target)
        return logits_k

    def forward(
        self,
        multiscale_conditioning: Union[List[torch.Tensor], torch.Tensor],
        level_idx: Optional[int] = None,
        prefix_emb: Optional[torch.Tensor] = None,
    ) -> Union[Dict[str, Union[List[torch.Tensor], torch.Tensor]], torch.Tensor]:
        """
        Unified forward entrypoint:
        - If multiscale_conditioning is a list of tensors: runs single-pass all-levels training.
        - If level_idx is specified: runs single-level evaluation for that level.
        """
        if level_idx is not None:
            assert isinstance(multiscale_conditioning, torch.Tensor), (
                "For single-level forward, multiscale_conditioning must be a Tensor z_hat"
            )
            return self.forward_single_level(multiscale_conditioning, level_idx, prefix_emb=prefix_emb)
        else:
            assert isinstance(multiscale_conditioning, list), (
                "For all-levels forward, multiscale_conditioning must be a list of K tensors"
            )
            return self.forward_single_pass_all_levels(multiscale_conditioning, prefix_emb=prefix_emb)
