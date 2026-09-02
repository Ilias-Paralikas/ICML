from .modules.vector_encoder import VectorEncoder
from .modules.mask_decoder import MaskDecoder
import torch.nn as nn


class EncoderDecoder(nn.Module):
    """Full model: :class:`VectorEncoder` + :class:`MaskDecoder`.

    The encoder produces one vector per class; the decoder decodes each vector
    independently into its own reconstruction channel(s) + one segmentation
    logit. The decoder is given ``out_channels = in_channels + 1`` (reconstruction
    channels + the seg logit).

    forward(x): ``x`` is ``(B, in_channels, *input_size)``; returns
    ``(reconstruction, segmentation)`` — see :class:`MaskDecoder` for their shapes.

    Args:
        bottleneck_dim: length of each per-class vector.
        number_of_vectors: number of per-class heads (== classes being trained).
        vectorizers_mat_mul: per-head bool list for the Vectorizer prototype path;
            must have length ``number_of_vectors``.
        in_channels: input image channels (1 grayscale ultrasound, 3 RGB).
        input_size: spatial input size (used by the encoder's shape probe).
        decoder_channels / encoder_channels: per-stage channel widths.
        first_conv_size: kernel size of the decoder's first (transposed) conv.
    """

    def __init__(self,
                 bottleneck_dim=128,
                 number_of_vectors=2,
                vectorizers_mat_mul = [True,True],
                 in_channels=1,
                 input_size=(256,256),
                 decoder_channels=[1024,512, 256, 128, 64,32,16],
                 encoder_channels=[32, 64, 128, 256, 512, 1024],
                 first_conv_size=4):
        super().__init__()
        self.bottleneck_dim = bottleneck_dim
        self.number_of_vectors = number_of_vectors
        self.vectorizers_mat_mul = vectorizers_mat_mul.copy()
        self.in_channels = in_channels
        self.input_size = input_size
        self.decoder_channels = decoder_channels
        self.encoder_channels = encoder_channels
        self.first_conv_size = first_conv_size
        self.encoder = VectorEncoder(bottleneck_dim=self.bottleneck_dim,
                                     in_channels=self.in_channels,
                                     channels=self.encoder_channels,
                                     input_size=self.input_size,
                                    number_of_vectors = self.number_of_vectors,
                                    vectorizers_mat_mul=self.vectorizers_mat_mul)

        self.decoder = MaskDecoder(bottleneck_dim=self.bottleneck_dim,
                                   out_channels =self.in_channels+1,
                                   first_conv_size=self.first_conv_size,
                                   channels=self.decoder_channels)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
