import torch.nn as nn
import torch


class Vectorizer(nn.Module):
    """One per-class head: maps the shared encoder bottleneck to a single vector.

    An MLP takes the flattened encoder output (``in_features`` neurons) and
    produces a ``bottleneck_dim`` vector, returned shaped ``(B, 1, bottleneck_dim)``
    so the caller can concatenate the per-class heads along dim 1.

    If ``use_matrix_multiplication`` is set, the MLP instead outputs
    ``number_of_vectors`` coefficients which are combined with a learned set of
    ``number_of_vectors`` prototype vectors (``self.vectors``,
    shape ``(number_of_vectors, bottleneck_dim)``) by a plain matmul — i.e. the
    head's output is constrained to a (softmax-free) linear combination of shared
    prototypes rather than a free MLP output. This is the per-vectorizer
    ``vectorizers_mat_mul`` config knob.

    Args:
        in_features: size of the flattened encoder output feeding this head.
        bottleneck_dim: length of the produced vector.
        number_of_vectors: number of prototype vectors (only used when
            ``use_matrix_multiplication`` is True).
        use_matrix_multiplication: enable the prototype-combination path above.
        linear_layer_dim: hidden widths of the MLP before the final layer.
    """

    def __init__(self,
                 in_features,
                 bottleneck_dim,
                 number_of_vectors,
                 use_matrix_multiplication=True,
                linear_layer_dim=[]):
        super().__init__()
        self.in_features = in_features
        self.bottleneck_dim = bottleneck_dim
        self.number_of_vectors = number_of_vectors
        self.linear_layer_dim = linear_layer_dim.copy()
        self.use_matrix_multiplication  = use_matrix_multiplication

        if  self.use_matrix_multiplication:
            self.linear_layer_dim.append(self.number_of_vectors)
            self.vectors = nn.Parameter(torch.randn(self.number_of_vectors, self.bottleneck_dim))
        else:
            self.linear_layer_dim.append(self.bottleneck_dim)

        linear_layer =nn.ModuleList([nn.Flatten()])
        linear_layer.append(nn.Linear(self.in_features,self.linear_layer_dim[0]))


        for i in range(len(self.linear_layer_dim)-1):
            linear_layer.append(nn.LayerNorm(self.linear_layer_dim[i]))
            linear_layer.append(nn.ReLU(inplace=True))
            linear_layer.append(nn.Linear(self.linear_layer_dim[i], self.linear_layer_dim[i+1]))



        self.linear = nn.Sequential(*linear_layer)


    def forward(self, x):
        batch_size= x.size(0)
        x = self.linear(x)
        if self.use_matrix_multiplication:
            x = x.view(batch_size,self.number_of_vectors)
            x = torch.matmul(x, self.vectors)
        x = x.view(batch_size, 1,x.shape[1])
        return x
