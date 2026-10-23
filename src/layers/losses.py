"""
NCM-Apex: Tokenizer & Prior Loss Functions & Accuracy Metrics
Implements:
1. Stage-1 Tokenizer Objective: L_tok = CrossEntropy(logits, targets) + VQ_Loss
2. Stage-2 Multiscale Prior Objective (Equation 10):
   L_NCM = - sum_{k=1}^K w_k sum_{i=1}^{l_k} log p_theta(r_k^{*i} | z_tilde^(k-1))
   where w_k = l_k^(-alpha) / sum_{j=1}^K l_j^(1 - alpha)
   with alpha in [0.0, 1.5] (alpha=0: ELBO equivalence, alpha=1: level-balanced)
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiscaleTokenizerLoss(nn.Module):
    """
    Combined Stage-1 loss function for Multiscale VQ-VAE.
    Computes cross-entropy reconstruction loss alongside the multi-level commitment/alignment loss.
    """

    def __init__(
        self,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        vq_loss: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            logits: Predicted vocabulary logits (B, L, VocabSize)
            targets: Ground-truth target token indices (B, L)
            vq_loss: Multiscale vector quantizer commitment loss (scalar)
        Returns:
            Dictionary with:
                - total_loss: L_recon + vq_loss
                - recon_loss: Cross-entropy token loss
                - vq_loss: Residual commitment/alignment loss
                - token_accuracy: Percentage of correctly predicted tokens [0.0 - 100.0]
        """
        B, L, V = logits.shape
        flat_logits = logits.view(-1, V)
        flat_targets = targets.view(-1)

        recon_loss = F.cross_entropy(
            flat_logits,
            flat_targets,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
        )

        total_loss = recon_loss + vq_loss

        with torch.no_grad():
            preds = torch.argmax(flat_logits, dim=-1)
            valid_mask = flat_targets != self.ignore_index
            if valid_mask.sum() > 0:
                correct = (preds[valid_mask] == flat_targets[valid_mask]).float().sum()
                accuracy = (correct / valid_mask.sum()) * 100.0
            else:
                accuracy = torch.tensor(0.0, device=logits.device)

        return {
            "total_loss": total_loss,
            "recon_loss": recon_loss,
            "vq_loss": vq_loss,
            "token_accuracy": accuracy,
        }


class MultiscalePriorLoss(nn.Module):
    """
    Stage-2 Weighted Multiscale Cross-Entropy Loss for NCM Prior (Paper Equation 10).

    Mathematical Objective:
        L_NCM = - sum_{k=1}^K w_k sum_{i=1}^{l_k} log p_theta(r_k^{*i} | z_tilde^(k-1))

    Weight formulation:
        w_k = l_k^(-alpha) / sum_{j=1}^K l_j^(1 - alpha)

    Weighting Regimes:
        - alpha = 0.0 (ELBO Equivalence):
          w_k = 1 / sum l_j.
          Multiplier on mean cross-entropy is W_k = w_k * l_k = l_k / sum l_j.
          Every token in the sequence receives identical weight, exactly matching the
          Evidence Lower Bound (ELBO) / standard token-level negative log-likelihood.
        - alpha = 1.0 (Level-Balanced Weighting):
          w_k = l_k^(-1) / K.
          Multiplier on mean cross-entropy is W_k = w_k * l_k = 1 / K.
          Every hierarchical scale contributes equally to the loss gradient regardless
          of whether it contains 16 coarse tokens or 64 fine tokens.
        - alpha in (0.0, 1.5): Smoothly balances macro-semantic vs. micro-syntactic learning.
    """

    def __init__(
        self,
        levels: Sequence[int],
        alpha: float = 0.0,
        label_smoothing: float = 0.0,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.levels = [float(l) for l in levels]
        self.num_levels = len(levels)
        self.alpha = float(alpha)
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index

        # Precompute normalized multipliers W_k = w_k * l_k = l_k^(1 - alpha) / sum_j l_j^(1 - alpha)
        den = sum(l ** (1.0 - self.alpha) for l in self.levels)
        weights = [(l ** (1.0 - self.alpha)) / den for l in self.levels]
        self.register_buffer("level_weights", torch.tensor(weights, dtype=torch.float32))

    def forward(
        self,
        level_logits: List[torch.Tensor],
        clean_targets: List[torch.Tensor],
    ) -> Dict[str, Union[torch.Tensor, List[torch.Tensor], List[float]]]:
        """
        Args:
            level_logits: List of K tensors [logits_1, ..., logits_K], shape (B, l_k, |V_k|).
            clean_targets: List of K tensors [r_1^*, ..., r_K^*], shape (B, l_k).

        Returns:
            Dictionary containing:
                - total_loss: Weighted scalar loss L_NCM
                - per_level_losses: List of scalar CE losses per scale
                - per_level_accuracies: List of accuracy percentages per scale
                - global_accuracy: Weighted accuracy across all scales
        """
        assert len(level_logits) == self.num_levels, (
            f"Expected {self.num_levels} levels, got {len(level_logits)}"
        )
        assert len(clean_targets) == self.num_levels, (
            f"Expected {self.num_levels} targets, got {len(clean_targets)}"
        )

        device = level_logits[0].device
        if self.level_weights.device != device:
            self.level_weights = self.level_weights.to(device)

        per_level_losses = []
        per_level_accuracies = []
        weighted_loss = torch.tensor(0.0, device=device)

        for k in range(self.num_levels):
            logits_k = level_logits[k]  # (B, l_k, V_k)
            targets_k = clean_targets[k]  # (B, l_k)
            V_k = logits_k.shape[-1]

            flat_logits = logits_k.reshape(-1, V_k)
            flat_targets = targets_k.reshape(-1)

            loss_k = F.cross_entropy(
                flat_logits,
                flat_targets,
                ignore_index=self.ignore_index,
                label_smoothing=self.label_smoothing,
            )
            per_level_losses.append(loss_k)

            # Accumulate weighted loss
            w_k = self.level_weights[k]
            weighted_loss = weighted_loss + w_k * loss_k

            # Compute scale accuracy
            with torch.no_grad():
                preds_k = torch.argmax(flat_logits, dim=-1)
                valid = flat_targets != self.ignore_index
                if valid.sum() > 0:
                    acc_k = ((preds_k[valid] == flat_targets[valid]).float().sum() / valid.sum()).item() * 100.0
                else:
                    acc_k = 0.0
                per_level_accuracies.append(acc_k)

        # Global weighted accuracy
        global_acc = sum(
            self.level_weights[k].item() * per_level_accuracies[k]
            for k in range(self.num_levels)
        )

        return {
            "total_loss": weighted_loss,
            "per_level_losses": per_level_losses,
            "per_level_accuracies": per_level_accuracies,
            "global_accuracy": global_acc,
        }


def compute_multilevel_accuracy(
    multiscale_logits: list,
    targets: torch.Tensor,
    ignore_index: int = -100,
) -> list:
    """
    Computes exact token accuracy across every individual hierarchical scale k in [1, K].
    Returns list of float accuracies [Acc_1, Acc_2, ..., Acc_K].
    """
    accuracies = []
    flat_targets = targets.view(-1)
    valid_mask = flat_targets != ignore_index

    for logits_k in multiscale_logits:
        preds = torch.argmax(logits_k.view(-1, logits_k.shape[-1]), dim=-1)
        if valid_mask.sum() > 0:
            acc = ((preds[valid_mask] == flat_targets[valid_mask]).float().sum() / valid_mask.sum()).item() * 100.0
        else:
            acc = 0.0
        accuracies.append(acc)

    return accuracies
