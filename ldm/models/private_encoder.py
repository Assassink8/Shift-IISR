from __future__ import annotations
import torch
import torch.nn as nn


class PrivateEncoder(nn.Module):
    """
    VAE latent (4 channels) -> private feature map [B, C, h, w]
    用于提取模态特定特征，生成FiLM参数
    """

    def __init__(
        self,
        in_channels: int = 3,  # ✓ VAE latent 是4通道
        out_channels: int = 256,
        base_channels: int = 64,
        num_down: int = 3,
        act: nn.Module | None = None,
    ):
        super().__init__()
        act = act or nn.SiLU()

        ch = base_channels
        layers = [nn.Conv2d(in_channels, ch, 3, padding=1), nn.GroupNorm(8, ch), act]
        for _ in range(num_down):
            layers += [
                nn.Conv2d(ch, ch * 2, 4, stride=2, padding=1),
                nn.GroupNorm(8, ch * 2),
                act,
            ]
            ch *= 2

        layers += [nn.Conv2d(ch, out_channels, 1), nn.GroupNorm(8, out_channels), act]
        self.net = nn.Sequential(*layers)
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: VAE latent [B, 4, h, w]
        Returns:
            feat: private features [B, out_channels, h', w']
        """
        return self.net(x)

    @staticmethod
    def pool_feat(feat: torch.Tensor) -> torch.Tensor:
        """全局平均池化，用于判别器等"""
        return feat.mean(dim=(2, 3))