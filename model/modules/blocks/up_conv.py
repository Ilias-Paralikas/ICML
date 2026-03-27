import torch.nn as nn
from .conv_block import ConvBlock
from .residual_double_conv import ResidualDoubleConv
import torch.nn.functional as F


class UpConv(nn.Module):
    def __init__(self, in_channels, 
                 out_channels,
                 norm,
                 scale=2,
                 mode='bilinear',
                 double_conv=True):
        super().__init__()
        self.scale = scale
        self.mode  = mode
        if double_conv:
            self.double_conv = ResidualDoubleConv(in_channels,norm=norm)
        else:
            self.double_conv = None
        self.channel_conv =  ConvBlock(in_channels, 
                                    out_channels, 
                                    kernel_size=3,
                                    padding=1,
                                    norm=norm)
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale, mode=self.mode, align_corners=False)
        if self.double_conv is not None:
            x = self.double_conv(x) 
        x = self.channel_conv(x)

        return x
    