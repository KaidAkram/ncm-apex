"""
NCM-Apex: Tokenizer Training Engine & Diagnostic Harness
Manages training loops, codebook telemetry, multi-level exact accuracy logging,
and RTX 3070 Ti mixed precision memory management.
"""

import os
import time
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.models.multiscale_vqvae import MultiscaleVQVAE
from src.layers.losses import MultiscaleTokenizerLoss, compute_multilevel_accuracy


class TokenizerTrainer:
    """
    Production-grade trainer and telemetry harness for Multiscale VQ-VAE.
    """

    def __init__(
        self,
        model: MultiscaleVQVAE,
        lr: float = 3e-4,
        weight_decay: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.95,
        grad_clip: float = 1.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_amp: bool = True,
    ):
        self.model = model.to(device)
        self.device = device
        self.use_amp = use_amp and ("cuda" in device)
        self.grad_clip = grad_clip

        # AdamW optimizer on non-codebook parameters (codebooks updated via EMA)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = AdamW(
            trainable_params,
            lr=lr,
            betas=(beta1, beta2),
            weight_decay=weight_decay,
        )

        self.criterion = MultiscaleTokenizerLoss()
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.history: Dict[str, List[float]] = {
            "total_loss": [],
            "recon_loss": [],
            "vq_loss": [],
            "token_accuracy": [],
            "global_utilization": [],
            "peak_vram_mb": [],
        }

    def train_step(self, batch_tokens: torch.Tensor) -> Dict[str, float]:
        """
        Executes a single forward-backward optimization step.
        """
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        batch_tokens = batch_tokens.to(self.device)

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            output = self.model(batch_tokens)
            loss_dict = self.criterion(
                logits=output["logits"],
                targets=batch_tokens,
                vq_loss=output["vq_loss"],
            )
            total_loss = loss_dict["total_loss"]

        # Scaled backward pass
        self.scaler.scale(total_loss).backward()

        # Gradient clipping
        if self.grad_clip > 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

        self.scaler.step(self.optimizer)
        self.scaler.update()

        # Telemetry
        peak_vram = 0.0
        if "cuda" in self.device:
            peak_vram = torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)

        metrics = {
            "total_loss": loss_dict["total_loss"].item(),
            "recon_loss": loss_dict["recon_loss"].item(),
            "vq_loss": loss_dict["vq_loss"].item(),
            "token_accuracy": loss_dict["token_accuracy"].item(),
            "global_utilization": output["global_utilization"],
            "cutoff_level": float(output["cutoff_level"]),
            "peak_vram_mb": peak_vram,
        }

        for k, v in metrics.items():
            if k in self.history:
                self.history[k].append(v)

        return metrics

    @torch.no_grad()
    def evaluate_multilevel_accuracy(self, batch_tokens: torch.Tensor) -> Dict[str, Union[List[float], float]]:
        """
        Evaluates exact token reconstruction accuracy independently across all K scales.
        """
        self.model.eval()
        batch_tokens = batch_tokens.to(self.device)

        recon_out = self.model.reconstruct_multiscales(batch_tokens)
        level_logits = recon_out["level_logits"]

        level_accs = compute_multilevel_accuracy(level_logits, batch_tokens)

        quant_out = self.model.quantizer(self.model.encode_continuous(batch_tokens))

        return {
            "level_accuracies": level_accs,
            "per_level_utilization": quant_out["per_level_utilization"],
            "codebook_perplexities": quant_out["codebook_perplexities"],
            "global_utilization": quant_out["global_utilization"],
        }
