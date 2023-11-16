import torch
from .subpackage_0 import important_string

class ImportsIndirectlyFromSubPackage(torch.nn.Module):
    key = important_string

    def forward(self, inp):
        if False:
            return 10
        return torch.sum(inp)