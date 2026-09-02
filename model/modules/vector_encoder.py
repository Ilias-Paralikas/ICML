import torch.nn as nn
import torch
from .blocks import ConvBlock, DownConv, Vectorizer


class VectorEncoder(nn.Module):
    """Conv downsampling stack followed by ``number_of_vectors`` per-class heads.

    A shared conv encoder reduces ``(B, in_channels, *input_size)`` to a small
    feature map, which is flattened and fed independently to each of
    ``number_of_vectors`` :class:`Vectorizer` heads (one per class/channel being
    trained). ``forward`` returns the list of per-class vectors, each shaped
    ``(B, 1, bottleneck_dim)``.

    The flattened-encoder width feeding the heads is discovered at build time with
    a dummy forward pass, so ``input_size`` / ``channels`` can change without
    hand-computing it.

    Args:
        in_channels: input image channels (1 for grayscale ultrasound, 3 for RGB).
        norm: norm layer class passed down to every conv block.
        channels: per-stage channel widths of the downsampling stack.
        bottleneck_dim: length of each per-class vector.
        vectorizers_mat_mul: per-head bool list (len == number_of_vectors) for the
            :class:`Vectorizer` ``use_matrix_multiplication`` prototype path.
        vectorizer_linear_layer_dim: hidden widths inside each head's MLP.
        number_of_vectors: number of per-class heads.
        input_size: spatial size of the input, used only for the shape-probe.
    """

    def __init__(self,
                 in_channels=1,
                 norm=nn.BatchNorm2d,
                 channels= [32, 64, 128, 256, 512, 1024],
                 bottleneck_dim=128,
                    vectorizers_mat_mul= [True,True],
                 vectorizer_linear_layer_dim=[1024],
                 number_of_vectors=4,
                 input_size=(256,256)):
        super().__init__()

        self.in_channels = in_channels
        self.channels = channels.copy()
        self.vectorizers_mat_mul=  vectorizers_mat_mul.copy()
        self.number_of_vectors=  number_of_vectors
        self.bottleneck_dim =bottleneck_dim
        self.vectorizer_linear_layer_dim= vectorizer_linear_layer_dim
        self.norm = norm
        self.input_size= input_size

        encoder_layers = nn.ModuleList([ConvBlock(self.in_channels,
                                                  self.channels[0],
                                                  kernel_size=4,
                                                  stride=2,
                                                  padding=1,
                                                  norm=self.norm)])

        for i in range(len(self.channels)-1):
            encoder_layers.append(DownConv(self.channels[i],
                                           self.channels[i+1],
                                           norm=self.norm))

        self.encoder = nn.Sequential(*encoder_layers)


        dummy_input=  torch.randn(1, self.in_channels, *self.input_size)
        with torch.no_grad():
            encoder_output = self.encoder(dummy_input)
            flat_enc_output = nn.Flatten()(encoder_output).shape[1]

        self.vectorizers = nn.ModuleList([Vectorizer( in_features=flat_enc_output,
                                                    bottleneck_dim= self.bottleneck_dim,
                                                    number_of_vectors=self.number_of_vectors,
                                                    linear_layer_dim=self.vectorizer_linear_layer_dim,
                                                     use_matrix_multiplication = mat_mul)
                                                     for mat_mul in self.vectorizers_mat_mul])


    def forward(self, x):
        x = self.encoder(x)
        vectors = []
        for v in self.vectorizers:
            vectors.append(v(x))


        return vectors
