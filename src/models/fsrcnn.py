import torch
import torch.nn as nn

class FSRCNN(nn.Module):
    """
    FSRCNN for x2 super-resolution.
    Input : (B,3,H,W)  LR in [0,1]
    Output: (B,3,2H,2W) SR in [0,1]
    """
    def __init__(self, scale=2, d=56, s=12, m=4):
        super().__init__()
        self.scale = scale

        self.feature = nn.Sequential(
            nn.Conv2d(3, d, kernel_size=5, padding=2),
            nn.PReLU(d),
        )
        self.shrink = nn.Sequential(
            nn.Conv2d(d, s, kernel_size=1),
            nn.PReLU(s),
        )

        mapping = []
        for _ in range(m):
            mapping += [nn.Conv2d(s, s, kernel_size=3, padding=1), nn.PReLU(s)]
        self.map = nn.Sequential(*mapping)

        self.expand = nn.Sequential(
            nn.Conv2d(s, d, kernel_size=1),
            nn.PReLU(d),
        )

        self.deconv = nn.ConvTranspose2d(
            d, 3, kernel_size=9, stride=scale, padding=4, output_padding=scale - 1
        )

    def forward(self, x):
        x = self.feature(x)
        x = self.shrink(x)
        x = self.map(x)
        x = self.expand(x)
        x = self.deconv(x)
        return torch.clamp(x, 0.0, 1.0)
