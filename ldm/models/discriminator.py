import torch.nn as nn

class DomainDiscriminatorP(nn.Module):
    def __init__(self, in_ch, hidden=256, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden, hidden, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden, hidden, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, feat):  # feat: [B,C,H,W]
        return self.net(feat)
    
class DomainDiscriminatorS(nn.Module):
    def __init__(self, in_ch, hidden=256, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden, hidden, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden, hidden, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, feat):  # feat: [B,C,H,W]
        return self.net(feat)