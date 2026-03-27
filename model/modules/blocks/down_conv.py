import torch.nn as nn
from .conv_block import ConvBlock
from .residual_double_conv import ResidualDoubleConv

class DownConv(nn.Module):
    def __init__(self,
                  in_channels,
                  out_channels,
                  norm,
                  double_conv=True):
        super().__init__()
        self.downconv =   ConvBlock(in_channels, 
                                    out_channels, 
                                    kernel_size=2,
                                    stride=2,
                                    padding=0, 
                                    norm=norm,
                                    bias=False)
        if double_conv:
            self.double_conv = ResidualDoubleConv(out_channels,norm=norm)
        else:
            self.double_conv = None

    def forward(self, x):
        x = self.downconv(x)
        if self.double_conv is not None:
            x = self.double_conv(x)
        return x 