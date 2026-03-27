import torch.nn as nn
import torch
from .blocks import ConvBlock, DownConv, Vectorizer

class VectorEncoder(nn.Module):
    def __init__(self, 
                 in_channels=1, 
                 norm=nn.BatchNorm2d,
                 channels= [32, 64, 128, 256, 512, 1024],
                 bottleneck_dim=128,
                 number_of_vectorizers=2,
                 vectorizer_linear_layer_dim=[1024],
                 number_of_vectors=4,
                 input_size=(256,256),
                 save_path='./cache/model_weights/hydra_encoder.pt'):
        super().__init__()
        
        self.in_channels = in_channels
        self.channels = channels.copy()
        self.save_path = save_path
        self.number_of_vectors=  number_of_vectors
        self.number_of_vectorizers =  number_of_vectorizers
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
     
        self.vectorizers = nn.ModuleList([Vectorizer( in_neuroes=flat_enc_output,
                                                    vector_dim= self.bottleneck_dim,
                                                    number_of_vectors=self.number_of_vectors,
                                                    linear_layer_dim=self.vectorizer_linear_layer_dim)
                                                     for _ in range(self.number_of_vectorizers)])


     
     
    def forward(self, x):
        x = self.encoder(x)
        vectors = []
        for v in self.vectorizers:
            vectors.append(v(x))


        return vectors

