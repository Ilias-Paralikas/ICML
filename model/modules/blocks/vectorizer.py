import torch.nn as nn
import torch


class Vectorizer(nn.Module):
    def __init__(self,
                 in_neuroes,
                 vector_dim,
                 number_of_vectors,
                 use_matrix_multiplication=True,
                linear_layer_dim=[]):
        super().__init__()
        self.in_neuroes = in_neuroes
        self.vector_dim = vector_dim
        self.number_of_vectors = number_of_vectors
        self.linear_layer_dim = linear_layer_dim.copy()
        self.use_matrix_multiplication  = use_matrix_multiplication

        
        self.linear_layer_dim.append(self.number_of_vectors)

        linear_layer =nn.ModuleList([nn.Flatten()])
        linear_layer.append(nn.Linear(self.in_neuroes,self.linear_layer_dim[0]))

        
        for i in range(len(self.linear_layer_dim)-1):
            linear_layer.append(nn.LayerNorm(self.linear_layer_dim[i]))
            linear_layer.append(nn.ReLU(inplace=True))
            linear_layer.append(nn.Linear(self.linear_layer_dim[i], self.linear_layer_dim[i+1]))

        
        
        self.linear = nn.Sequential(*linear_layer)

        self.vectors = nn.Parameter(torch.randn(self.number_of_vectors, self.vector_dim))

    def forward(self, x):
        batch_size= x.size(0)
        x = self.linear(x)
        if self.use_matrix_multiplication:
            x = x.view(batch_size,self.number_of_vectors)
            x = torch.matmul(x, self.vectors)
        x = x.view(batch_size, 1,x.shape[1])
        return x
    
