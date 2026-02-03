from __future__ import annotations
from typing import Literal, Optional

import torch
import torch.nn as nn


class SharedEncoder(nn.Module):
    """
    VAE latent -> shared feature map [B, C, h, w] (keeps spatial size).
    """

    def __init__(
        self,
        in_channels: int = 3,  # ✓ VAE latent 是4通道
        context_dim: int = 768,
        base_channels: int = 64,
        num_blocks: int = 2,
        act: Optional[nn.Module] = None,
    ):
        super().__init__()
        act = act or nn.SiLU()

        ch = base_channels
        self.in_proj = nn.Sequential(
            nn.Conv2d(in_channels, ch, 3, padding=1),
            nn.GroupNorm(8, ch),
            act,
        )

        blocks = []
        for _ in range(max(0, num_blocks)):
            blocks += [
                nn.Conv2d(ch, ch, 3, padding=1),
                nn.GroupNorm(8, ch),
                act,
                nn.Conv2d(ch, ch, 3, padding=1),
                nn.GroupNorm(8, ch),
                act,
            ]
        self.blocks = nn.Sequential(*blocks)

        self.out_proj = nn.Conv2d(ch, context_dim, 3, padding=1)
        self.skip = nn.Conv2d(in_channels, context_dim, 1) if in_channels != context_dim else None
        self.context_dim = context_dim

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: VAE latent [B, 4, h, w]
        Returns:
            shared_feat: [B, C, h, w]
        """
        feat = self.in_proj(z)
        if len(self.blocks) > 0:
            feat = feat + self.blocks(feat)
        feat = self.out_proj(feat)
        if self.skip is not None:
            feat = feat + self.skip(z)
        return feat

    @staticmethod
    def pool_tokens(tokens: torch.Tensor, mode: Literal["mean", "max"] = "mean") -> torch.Tensor:
        if tokens.dim() == 4:
            if mode == "mean":
                return tokens.mean(dim=(2, 3))
            if mode == "max":
                return tokens.amax(dim=(2, 3))
        elif tokens.dim() == 3:
            if mode == "mean":
                return tokens.mean(dim=1)
            if mode == "max":
                return tokens.max(dim=1).values
        raise ValueError(f"Unsupported shape {tuple(tokens.shape)} or mode={mode}")
