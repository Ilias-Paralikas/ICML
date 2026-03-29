import torch.nn as nn
import torch
from .blocks import ConvBlock, UpConv
class MaskDecoder(nn.Module):
    def __init__(self,
                  bottleneck_dim=128,
                 out_channels=2,
                channels= [1024,512, 256, 128, 64,32,16],
                first_conv_size = 4,
                double_conv=True,
                norm=nn.BatchNorm2d):
 
        super().__init__()
        self.bottleneck_dim = bottleneck_dim
        self.out_channels =out_channels
        self.channels = channels.copy()
        self.first_conv_size=first_conv_size
        self.norm = norm
        self.double_conv = double_conv
        dec_layers = nn.ModuleList([ConvBlock(self.bottleneck_dim, 
                                              self.channels[0], 
                                              kernel_size=self.first_conv_size, 
                                              stride=1, 
                                              padding=0, 
                                              bias=False,
                                              norm=self.norm,
                                              Transpose=True)])
        for i in range(len(channels)-1):
            dec_layers.append(UpConv(self.channels[i], 
                                     self.channels[i+1],
                                     norm=self.norm,
                                    double_conv=self.double_conv,
            ))

        dec_layers.append(nn.Conv2d(self.channels[-1],
                                     self.out_channels, 
                                     kernel_size=1, 
                                     stride=1, 
                                     padding=0))
        self.decoder = nn.Sequential(*dec_layers)
        self.sigmoid_layer = nn.Sigmoid()
    def forward(self, x):
       
        x = torch.cat(x, dim=1)
      
        batch_size = x.shape[0]
        number_of_vectors = x.shape[1]
        x = x.view(batch_size* number_of_vectors, -1,1,1)
        x = self.decoder(x)
        x = x.view(batch_size, number_of_vectors, *x.shape[1:])
        if x.shape[2] ==2:
            reconstruction  =x[:,:,0]
        else:
            reconstruction  =x[:,:,:-1]
        reconstruction= self.sigmoid_layer(reconstruction)
        segmentation= x[:,:,-1]
        return reconstruction, segmentation
