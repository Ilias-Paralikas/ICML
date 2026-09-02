"""Plain 2D U-Net — the standard backbone for semi-supervised medical-image
segmentation baselines. SSL4MIS / BCP / UA-MT / MC-Net all use this shape
(4 down, 4 up, base 16-32 channels, BatchNorm, optional dropout).

Parametric in `in_channels` (1 grayscale, 3 RGB) and `num_classes`, so the same
code covers every dataset in this repo. Input side must be divisible by 16
(256x256 is).
"""
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, cin, cout, dropout=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=2, base=32, dropout=0.0):
        """dropout: applied in the bottleneck + the two deepest decoder blocks.
        Set > 0 for UA-MT (needs MC-dropout); 0 for Mean Teacher / BCP."""
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        c = [base, base * 2, base * 4, base * 8, base * 16]

        self.enc1 = DoubleConv(in_channels, c[0])
        self.enc2 = DoubleConv(c[0], c[1])
        self.enc3 = DoubleConv(c[1], c[2])
        self.enc4 = DoubleConv(c[2], c[3])
        self.bottleneck = DoubleConv(c[3], c[4], dropout=dropout)
        self.pool = nn.MaxPool2d(2)

        self.up4 = nn.ConvTranspose2d(c[4], c[3], 2, stride=2)
        self.dec4 = DoubleConv(c[4], c[3], dropout=dropout)
        self.up3 = nn.ConvTranspose2d(c[3], c[2], 2, stride=2)
        self.dec3 = DoubleConv(c[3], c[2], dropout=dropout)
        self.up2 = nn.ConvTranspose2d(c[2], c[1], 2, stride=2)
        self.dec2 = DoubleConv(c[2], c[1])
        self.up1 = nn.ConvTranspose2d(c[1], c[0], 2, stride=2)
        self.dec1 = DoubleConv(c[1], c[0])
        self.head = nn.Conv2d(c[0], num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.head(d1)   # (B, num_classes, H, W) logits


class SegWrapper(nn.Module):
    """Adapts a seg net (returns logits) to the ``(reconstruction, segmentation)``
    tuple that ``utils.evaluation.evaluate`` expects, so the baselines are scored
    by the exact same code path as our own model."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return None, self.net(x)
