import torch.nn as nn
class ConvBlock(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 norm,
                 kernel_size=3, 
                 stride=1, 
                 padding=1,
                 bias=False,
                 Transpose=False):
        super().__init__()
        if Transpose:
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
    

