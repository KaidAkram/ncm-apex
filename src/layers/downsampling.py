"""
NCM-Apex: Downsampling and Reconstruction Operators
Implements spatial compression (Down_k) and spatial expansion (R_k) across hierarchical scales,
along with residual convolutional blending (Phi).
"""

import math
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualConvBlend(nn.Conv1d):
    """
    Phi residual convolution for post-upsampling smoothing.
    Blends identity and convolution: y = (1 - r) * x + r * Conv1d(x).
    """

    def __init__(
        self,
        embed_dim: int,
        ratio: float = 0.5,
        kernel_size: int = 3,
        learnable_ratio: bool = False,
    ):
        super().__init__(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )
        self.learnable_ratio = learnable_ratio
        if learnable_ratio:
            init_r = min(max(abs(ratio), 1e-4), 1.0 - 1e-4)
            init_logit = math.log(init_r / (1.0 - init_r))
            self.ratio_logit = nn.Parameter(torch.tensor(init_logit, dtype=torch.float32))
        else:
            self.register_buffer("ratio_const", torch.tensor(abs(ratio), dtype=torch.float32))

    @property
    def blend_ratio(self) -> torch.Tensor:
        if self.learnable_ratio:
            return torch.sigmoid(self.ratio_logit)
        return self.ratio_const

    def forward(self, x_bcl: torch.Tensor) -> torch.Tensor:
        r = self.blend_ratio
        conv_out = super().forward(x_bcl)
        return x_bcl * (1.0 - r) + conv_out * r


class MultiscaleDownsampler(nn.Module):
    """
    Downsamples continuous latent tensor from length L to scale length l_k.
    Supports area pooling, learned 1D convolutions, and linear interpolation.
    """

    def __init__(
        self,
        embed_dim: int,
        levels: List[int],
        learned_l1: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.levels = levels
        self.num_levels = len(levels)
        self.target_full_len = levels[-1]

        # Optional learned downsampling for level 0
        self.learned_down_l1: Optional[nn.Conv1d] = None
        if learned_l1 and levels[0] < self.target_full_len:
            stride = self.target_full_len // levels[0]
            if self.target_full_len % levels[0] == 0:
                conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=stride, stride=stride, bias=True)
                with torch.no_grad():
                    conv.weight.zero_()
                    eye_idx = torch.arange(embed_dim)
                    conv.weight[eye_idx, eye_idx, :] = 1.0 / stride
                    conv.bias.zero_()
                self.learned_down_l1 = conv

    def forward(self, z_blc: torch.Tensor, level_idx: int) -> torch.Tensor:
        """
        Downsamples z_blc from (B, L, D) to (B, l_k, D).
        """
        target_len = self.levels[level_idx]
        current_len = z_blc.shape[1]

        if current_len == target_len:
            return z_blc

        z_bcl = z_blc.transpose(1, 2)  # (B, D, L)

        if level_idx == 0 and self.learned_down_l1 is not None and current_len == self.target_full_len:
            down_bcl = self.learned_down_l1(z_bcl)
        else:
            down_bcl = F.interpolate(z_bcl, size=(target_len,), mode="area")

        return down_bcl.transpose(1, 2).contiguous()


class MultiscaleReconstructor(nn.Module):
    """
    Upsamples quantized vectors from scale length l_k back to full target length L.
    Applies linear/nearest interpolation, optional learned transposed convolution,
    and residual conv blend Phi.
    """

    def __init__(
        self,
        embed_dim: int,
        levels: List[int],
        learned_l1: bool = True,
        blend_ratio: float = 0.5,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.levels = levels
        self.num_levels = len(levels)
        self.target_full_len = levels[-1]

        # Learned upsampler for level 0
        self.learned_up_l1: Optional[nn.ConvTranspose1d] = None
        if learned_l1 and levels[0] < self.target_full_len:
            stride = self.target_full_len // levels[0]
            if self.target_full_len % levels[0] == 0:
                conv_t = nn.ConvTranspose1d(embed_dim, embed_dim, kernel_size=stride, stride=stride, bias=True)
                with torch.no_grad():
                    conv_t.weight.zero_()
                    eye_idx = torch.arange(embed_dim)
                    conv_t.weight[eye_idx, eye_idx, :] = 1.0
                    conv_t.bias.zero_()
                self.learned_up_l1 = conv_t

        # Per-level residual conv blend Phi
        self.phi_blocks = nn.ModuleList([
            ResidualConvBlend(embed_dim, ratio=blend_ratio, kernel_size=3)
            for _ in range(self.num_levels)
        ])

    def forward(self, q_blc: torch.Tensor, level_idx: int) -> torch.Tensor:
        """
        Upsamples q_blc from (B, l_k, D) to (B, L, D) and applies smoothing blend.
        """
        target_len = self.target_full_len
        current_len = q_blc.shape[1]

        if current_len == target_len:
            up_blc = q_blc
        else:
            q_bcl = q_blc.transpose(1, 2)  # (B, D, l_k)
            if level_idx == 0 and self.learned_up_l1 is not None:
                up_bcl = self.learned_up_l1(q_bcl)
            else:
                up_bcl = F.interpolate(q_bcl, size=(target_len,), mode="linear", align_corners=False)
            up_blc = up_bcl.transpose(1, 2).contiguous()

        # Apply Phi smoothing convolution
        up_bcl = up_blc.transpose(1, 2)
        blended_bcl = self.phi_blocks[level_idx](up_bcl)
        return blended_bcl.transpose(1, 2).contiguous()
