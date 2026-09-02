import torch.nn as nn


class ConvBlock(nn.Module):
    """(Transposed) conv -> norm -> ReLU, the atomic building block of both stacks.

    Args:
        in_channels, out_channels: channel counts for the conv.
        norm: a norm layer *class* (e.g. nn.BatchNorm2d), instantiated here as
            ``norm(out_channels)``.
        kernel_size, stride, padding, bias: passed straight to the conv.
        transpose: if True use nn.ConvTranspose2d (upsampling / decoder side),
            otherwise nn.Conv2d.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 norm,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 bias=False,
                 transpose=False):
        super().__init__()
        if transpose:
            self.conv = nn.ConvTranspose2d(in_channels,
                                            out_channels,
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            padding=padding,
                                            bias=bias)
        else:
            self.conv = nn.Conv2d(  in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        bias=bias)

        self.norm =nn.Sequential(
            norm(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x
