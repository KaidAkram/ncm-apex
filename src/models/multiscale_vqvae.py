"""
NCM-Apex: Multiscale VQ-VAE Architecture (Stage-1 Tokenizer)
Integrates Token Encoder, Multiscale RVQ with Nested Scale Dropout & STE,
and SDPA-accelerated Transformer Decoder.
"""

import math
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.layers.residual_quantizer import MultiscaleResidualVectorQuantizer


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


class TransformerAttentionBlock(nn.Module):
    """
    Standard Transformer Decoder Block accelerated via PyTorch SDPA
    (scaled_dot_product_attention) with QK normalization and pre-norm.
    """

    def __init__(
        self,
        embed_dim: int,
        n_head: int = 6,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        assert embed_dim % n_head == 0, f"embed_dim {embed_dim} must be divisible by n_head {n_head}"
        self.embed_dim = embed_dim
        self.n_head = n_head
        self.head_dim = embed_dim // n_head
        self.dropout = dropout

        # Pre-attention norm
        self.norm1 = RMSNorm(embed_dim)
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Pre-MLP norm
        self.norm2 = RMSNorm(embed_dim)
        self.mlp_fc1 = nn.Linear(embed_dim, 4 * embed_dim, bias=bias)
        self.mlp_fc2 = nn.Linear(4 * embed_dim, embed_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape

        # Attention sub-layer
        normed = self.norm1(x)
        qkv = self.qkv_proj(normed).view(B, L, 3, self.n_head, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # Each is (B, L, H, head_dim)

        # Transpose to (B, H, L, head_dim) for SDPA
        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)

        # PyTorch SDPA (native FlashAttention / Memory-Efficient backend)
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, D)
        x = x + self.out_proj(attn_out)

        # Feed-forward MLP sub-layer (GeLU)
        mlp_in = self.norm2(x)
        mlp_out = self.mlp_fc2(F.gelu(self.mlp_fc1(mlp_in)))
        x = x + mlp_out
        return x


class MultiscaleVQVAE(nn.Module):
    """
    Complete Multiscale Residual VQ-VAE for text tokenization and continuous latent representation.
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        embed_dim: int = 256,
        codebook_vocab_sizes: List[int] = [512, 512, 512],
        levels: List[int] = [16, 32, 64],
        dec_layers: int = 2,
        n_head: int = 4,
        dropout: float = 0.0,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        fine_drop_prob: float = 0.5,
        fine_drop_min_keep: int = 1,
        restart_threshold: float = 1.0,
        use_pos_emb: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.embed_dim = embed_dim
        self.levels = list(levels)
        self.num_levels = len(levels)

        # Encoder: Word token embedding + position embedding
        self.wte = nn.Embedding(vocab_size, embed_dim)
        self.wpe = nn.Embedding(context_length, embed_dim) if use_pos_emb else None
        self.drop = nn.Dropout(dropout)

        # Pre-quantization normalization (elementwise_affine=False to keep scale bounded)
        self.pre_quant_norm = nn.LayerNorm(embed_dim, elementwise_affine=False)

        # Hierarchical Multiscale Residual Quantizer
        self.quantizer = MultiscaleResidualVectorQuantizer(
            vocab_sizes=codebook_vocab_sizes,
            embed_dim=embed_dim,
            levels=levels,
            commitment_cost=commitment_cost,
            decay=decay,
            restart_threshold=restart_threshold,
            fine_drop_prob=fine_drop_prob,
            fine_drop_min_keep=fine_drop_min_keep,
        )

        # Decoder: Transformer Decoder Blocks + Vocab Projection
        self.decoder_blocks = nn.ModuleList([
            TransformerAttentionBlock(embed_dim=embed_dim, n_head=n_head, dropout=dropout)
            for _ in range(dec_layers)
        ])
        self.dec_norm = RMSNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

        # Weight tying: tie output projection weights to encoder token embeddings
        self.head.weight = self.wte.weight

    def encode_continuous(self, x: torch.Tensor) -> torch.Tensor:
        """
        Maps discrete tokens (B, L) to continuous latent representations (B, L, D).
        """
        B, L = x.shape
        tok_emb = self.wte(x)
        if self.wpe is not None:
            pos = torch.arange(0, L, dtype=torch.long, device=x.device)
            tok_emb = tok_emb + self.wpe(pos)
        z = self.drop(tok_emb)
        z = self.pre_quant_norm(z)
        return z

    def decode_latent(self, z_hat: torch.Tensor) -> torch.Tensor:
        """
        Maps continuous/quantized latent (B, L, D) to vocabulary logits (B, L, VocabSize).
        """
        h = z_hat
        for block in self.decoder_blocks:
            h = block(h)
        h = self.dec_norm(h)
        logits = self.head(h)
        return logits

    def forward(
        self,
        x: torch.Tensor,
        force_cutoff_level: Optional[int] = None,
    ) -> Dict[str, Union[torch.Tensor, List[torch.Tensor], float]]:
        """
        End-to-end forward pass:
        Tokens -> Continuous Latent -> Hierarchical RVQ -> Decoder -> Logits
        """
        # 1. Continuous Encoding
        z = self.encode_continuous(x)

        # 2. Residual Vector Quantization with STE & Nested Dropout
        quant_out = self.quantizer(z, force_cutoff_level=force_cutoff_level)
        z_hat = quant_out["z_hat"]

        # 3. Decoding to vocabulary logits
        logits = self.decode_latent(z_hat)

        return {
            "logits": logits,
            "z": z,
            "z_hat": z_hat,
            "vq_loss": quant_out["vq_loss"],
            "codes": quant_out["codes"],
            "multiscale_z_hat": quant_out["multiscale_z_hat"],
            "per_level_utilization": quant_out["per_level_utilization"],
            "global_utilization": quant_out["global_utilization"],
            "codebook_perplexities": quant_out["codebook_perplexities"],
            "cutoff_level": quant_out["cutoff_level"],
        }

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Tokenizes text into discrete hierarchical code sequences [r_1, ..., r_K].
        """
        z = self.encode_continuous(x)
        quant_out = self.quantizer(z)
        return quant_out["codes"]

    @torch.no_grad()
    def decode_codes(self, codes: List[torch.Tensor]) -> torch.Tensor:
        """
        Decodes discrete codes back to token predictions.
        """
        z_q = self.quantizer.codes_to_z_q(codes)
        logits = self.decode_latent(z_q)
        return torch.argmax(logits, dim=-1)

    @torch.no_grad()
    def reconstruct_multiscales(self, x: torch.Tensor) -> Dict[str, List[torch.Tensor]]:
        """
        Evaluates cumulative reconstructions level-by-level across all scales k in [1, K].
        Returns:
            - level_predictions: List of (B, L) tensor token predictions for each scale k.
            - level_logits: List of (B, L, VocabSize) logits for each scale k.
        """
        z = self.encode_continuous(x)
        quant_out = self.quantizer(z)
        multiscale_z = quant_out["multiscale_z_hat"]

        level_predictions = []
        level_logits = []

        for z_k in multiscale_z:
            logits_k = self.decode_latent(z_k)
            preds_k = torch.argmax(logits_k, dim=-1)
            level_logits.append(logits_k)
            level_predictions.append(preds_k)

        return {
            "level_predictions": level_predictions,
            "level_logits": level_logits,
            "codes": quant_out["codes"],
        }
