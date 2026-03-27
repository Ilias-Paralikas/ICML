from .modules.vector_encoder import VectorEncoder
from .modules.mask_decoder import MaskDecoder
import torch.nn as nn

class EncoderDecoder(nn.Module):
    def __init__(self,
                 bottleneck_dim=128,
                 number_of_vectors=2,
                vectorizers_mat_mul = [True,True],
                 in_channels=1,
                 input_size=(256,256),
                 decoder_channels=[1024,512, 256, 128, 64,32,16],
                 encoder_channels=[32, 64, 128, 256, 512, 1024],
                 first_conv_size=4,
                save_path='cache/test.pt'):
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
                                   

        self.save_path = save_path

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
