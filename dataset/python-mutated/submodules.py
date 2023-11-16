"""
MLP implementation
"""
import torch
from torch import nn

class FullyConnectedModule(nn.Module):

    def __init__(self, input_size: int, output_size: int, hidden_size: int, n_hidden_layers: int, activation_class: nn.ReLU, dropout: float=None, norm: bool=True):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_hidden_layers = n_hidden_layers
        self.activation_class = activation_class
        self.dropout = dropout
        self.norm = norm
        module_list = [nn.Linear(input_size, hidden_size), activation_class()]
        if dropout is not None:
            module_list.append(nn.Dropout(dropout))
        if norm:
            module_list.append(nn.LayerNorm(hidden_size))
        for _ in range(n_hidden_layers):
            module_list.extend([nn.Linear(hidden_size, hidden_size), activation_class()])
            if dropout is not None:
                module_list.append(nn.Dropout(dropout))
            if norm:
                module_list.append(nn.LayerNorm(hidden_size))
        module_list.append(nn.Linear(hidden_size, output_size))
        self.sequential = nn.Sequential(*module_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if False:
            return 10
        return self.sequential(x)