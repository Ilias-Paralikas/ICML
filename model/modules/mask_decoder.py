import torch.nn as nn
import torch
from .blocks import ConvBlock, UpConv


class MaskDecoder(nn.Module):
    """Decode every per-class vector independently through one shared decoder.

    The list of per-class vectors from :class:`VectorEncoder` is concatenated and
    reshaped so the batch and vector axes are folded together — every vector then
    passes through the *same* transposed-conv upsampling stack — and split back
    apart afterwards. There are no skip connections from the encoder, so the
    decoder learns a fixed spatial template per class.

    The stack ends on ``out_channels`` channels, of which the **last** is the
    segmentation logit and the rest are the reconstruction (sigmoid-activated).
    ``EncoderDecoder`` builds this with ``out_channels = in_channels + 1``.

    forward:
        input : list/tuple of ``number_of_vectors`` tensors, each ``(B, 1, bottleneck_dim)``
        returns ``(reconstruction, segmentation)`` where
            segmentation    : ``(B, number_of_vectors, H, W)`` logits
            reconstruction  : ``(B, number_of_vectors, H, W)``      if out_channels == 2
                              ``(B, number_of_vectors, C, H, W)``   otherwise (C = out_channels - 1)
    """

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
                                              transpose=True)])
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
        x = x.view(batch_size, number_of_vectors, *x.shape[1:])   # (B, N, out_channels, H, W)

        # out_channels = (reconstruction channels) + 1: last channel is the seg logit,
        # the rest are the reconstruction. Collapse the channel axis for a
        # single-channel reconstruction (grayscale) so downstream sees (B, N, H, W);
        # keep it for multi-channel (RGB) as (B, N, C, H, W).
        recon_channels = self.out_channels - 1
        if recon_channels == 1:
            reconstruction = x[:, :, 0]
        else:
            reconstruction = x[:, :, :recon_channels]
        reconstruction = self.sigmoid_layer(reconstruction)
        segmentation = x[:, :, -1]
        return reconstruction, segmentation
