"""
NCM-Apex: Stage-2 Prior Generator Trainer
Executes single-pass training of NCM Transformer Prior over all K abstraction scales:
1. Frozen Stage-1 Tokenizer extraction of ground-truth hierarchical discrete codes
2. Stochastic conditioning corruption to prevent compounding exposure bias
3. Single-forward-pass training with Block-Diagonal Attention across all K scales
4. Weighted Multiscale Cross-Entropy Loss optimization (Equation 10)
5. Gradient clipping, AMP autocast acceleration, and VRAM monitoring
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
from torch.optim import Optimizer

from src.engine.corruption_scheduler import ConditioningCorruptionScheduler
from src.layers.losses import MultiscalePriorLoss
from src.models.ncm_transformer import NCMTransformer


class GeneratorTrainer:
    """
    Stage-2 Generator Training Engine.

    Trains the NCM Transformer prior to predict discrete multiscale codes coarse-to-fine.
    In training mode, all K levels are evaluated in a single forward pass over sequence length
    N = sum l_k using the Block-Diagonal Attention mask.
    """

    def __init__(
        self,
        model: NCMTransformer,
        tokenizer: nn.Module,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[Any] = None,
        corruption_scheduler: Optional[ConditioningCorruptionScheduler] = None,
        loss_fn: Optional[MultiscalePriorLoss] = None,
        device: Optional[Union[str, torch.device]] = None,
        grad_clip: float = 1.0,
        use_amp: bool = True,
        lr: float = 3e-4,
        weight_decay: float = 0.01,
        loss_alpha: float = 0.0,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device
        self.grad_clip = grad_clip
        self.use_amp = use_amp and ("cuda" in str(self.device))

        # Freeze Stage-1 tokenizer permanently
        self.tokenizer.eval()
        for p in self.tokenizer.parameters():
            p.requires_grad = False

        # Default optimizer if none passed
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=(0.9, 0.95),
            )
        else:
            self.optimizer = optimizer

        self.scheduler = scheduler

        # Default corruption scheduler
        if corruption_scheduler is None:
            self.corruption_scheduler = ConditioningCorruptionScheduler(
                levels=model.levels,
                codebook_vocab_sizes=model.codebook_vocab_sizes,
                schedule_mode="linear_decay",
                p_start=0.70,
                p_end=0.30,
            )
        else:
            self.corruption_scheduler = corruption_scheduler

        # Default multiscale prior loss
        if loss_fn is None:
            self.loss_fn = MultiscalePriorLoss(
                levels=model.levels,
                alpha=loss_alpha,
            )
        else:
            self.loss_fn = loss_fn

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def train_step(
        self,
        tokens: torch.Tensor,
        prefix_tokens: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Executes a single Stage-2 optimization step:
        1. Tokenizer encodes tokens -> clean codes r_1^*, ..., r_K^*
        2. CorruptionScheduler applies uniform noise -> corrupted conditioning z_tilde^(0..K-1)
        3. Single forward pass through NCM Transformer with Block-Diagonal Attention
        4. Cross-entropy loss against clean targets & backprop
        """
        self.model.train()
        self.tokenizer.eval()

        tokens = tokens.to(self.device)
        self.optimizer.zero_grad()

        # 1. Extract clean discrete codes from frozen tokenizer
        with torch.no_grad():
            clean_codes = self.tokenizer.encode(tokens)

            # 2. Build corrupted cumulative conditioning representations
            corruption_data = self.corruption_scheduler.build_corrupted_conditioning(
                tokenizer=self.tokenizer,
                clean_codes=clean_codes,
            )
            corrupted_cond = corruption_data["corrupted_conditioning"]
            clean_targets = corruption_data["clean_targets"]

            # Optional prefix encoding
            prefix_emb = None
            if prefix_tokens is not None:
                prefix_tokens = prefix_tokens.to(self.device)
                prefix_z = self.tokenizer.encode_continuous(prefix_tokens)
                prefix_emb = self.model.cond_proj(prefix_z)

        # 3. Single forward pass in AMP
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            out = self.model.forward_single_pass_all_levels(
                multiscale_conditioning=corrupted_cond,
                prefix_emb=prefix_emb,
            )
            level_logits = out["level_logits"]

            # 4. Compute weighted multiscale cross-entropy loss (Eq 10)
            loss_dict = self.loss_fn(level_logits, clean_targets)
            total_loss = loss_dict["total_loss"]

        # 5. Backprop with AMP GradScaler
        self.scaler.scale(total_loss).backward()

        if self.grad_clip > 0.0:
            self.scaler.unscale_(self.optimizer)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip).item())
        else:
            grad_norm = 0.0

        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.scheduler is not None:
            self.scheduler.step()

        per_level_losses_f = [l.item() for l in loss_dict["per_level_losses"]]

        return {
            "total_loss": float(total_loss.item()),
            "per_level_losses": per_level_losses_f,
            "per_level_accuracies": loss_dict["per_level_accuracies"],
            "global_accuracy": float(loss_dict["global_accuracy"]),
            "grad_norm": grad_norm,
            "corruption_rates": corruption_data["corruption_rates"],
        }

    @torch.no_grad()
    def evaluate(
        self,
        tokens: torch.Tensor,
        prefix_tokens: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Evaluates model on validation data with clean (uncorrupted) conditioning.
        """
        self.model.eval()
        self.tokenizer.eval()

        tokens = tokens.to(self.device)

        # 1. Clean codes & uncorrupted cumulative conditioning
        clean_codes = self.tokenizer.encode(tokens)
        multiscale_z = self.tokenizer.quantizer.get_multiscale_reconstructions_from_codes(clean_codes)

        # Build [z^(0), z^(1), ..., z^(K-1)]
        B, L, D = multiscale_z[0].shape
        clean_conditioning = [torch.zeros(B, L, D, device=self.device)]
        clean_conditioning.extend(multiscale_z[:-1])

        prefix_emb = None
        if prefix_tokens is not None:
            prefix_tokens = prefix_tokens.to(self.device)
            prefix_z = self.tokenizer.encode_continuous(prefix_tokens)
            prefix_emb = self.model.cond_proj(prefix_z)

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            out = self.model.forward_single_pass_all_levels(
                multiscale_conditioning=clean_conditioning,
                prefix_emb=prefix_emb,
            )
            level_logits = out["level_logits"]
            loss_dict = self.loss_fn(level_logits, clean_codes)

        per_level_losses_f = [l.item() for l in loss_dict["per_level_losses"]]

        return {
            "eval_loss": float(loss_dict["total_loss"].item()),
            "eval_per_level_losses": per_level_losses_f,
            "eval_per_level_accuracies": loss_dict["per_level_accuracies"],
            "eval_global_accuracy": float(loss_dict["global_accuracy"]),
        }
