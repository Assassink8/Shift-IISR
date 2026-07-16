from __future__ import annotations
import torch
import torch.nn as nn


class GRMFeatureExtractor(nn.Module):
    """
    Extract the modality-aware GRM feature from the VAE latent.
    """

    def __init__(
        self,
        in_channels: int = 3, 
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
            x: VAE latent [B, 3, h, w]
        Returns:
            feat: GRM features [B, out_channels, h', w']
        """
        return self.net(x)

    @staticmethod
    def pool_feat(feat: torch.Tensor) -> torch.Tensor:
        """Global average pooling used by the modality classifier."""
        return feat.mean(dim=(2, 3))


# Backward-compatible import name used by released configs/checkpoints.
PrivateEncoder = GRMFeatureExtractor
