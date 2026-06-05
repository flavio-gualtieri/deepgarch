import torch
import torch.nn as nn
from torch import Tensor

class ParamNet(nn.Module):

    def __init__(self, embedding_dim: int, hidden_dims: list[int], n_params: int, dropout: float = 0.0):

        super().__init__()

        self._embedding_dim = embedding_dim
        self._hidden_dims = hidden_dims
        self._n_params = n_params
        self._dropout = dropout

        layers = []
        input_dim = embedding_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, n_params))

        self._mlp = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self._mlp(x)