"""
transpose last 2 dimensions of the input
"""
import torch.nn as nn

class TransposeLast(nn.Module):

    def __init__(self, deconstruct_idx=None, tranpose_dim=-2):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.deconstruct_idx = deconstruct_idx
        self.tranpose_dim = tranpose_dim

    def forward(self, x):
        if False:
            i = 10
            return i + 15
        if self.deconstruct_idx is not None:
            x = x[self.deconstruct_idx]
        return x.transpose(self.tranpose_dim, -1)