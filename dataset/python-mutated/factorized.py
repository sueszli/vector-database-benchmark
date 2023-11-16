from typing import Union
import torch
from torch import nn

class FactorizedLinear(nn.Module):
    """Factorized wrapper for ``nn.Linear``

    Args:
        nn_linear: torch ``nn.Linear`` module
        dim_ratio: dimension ration to use after weights SVD
    """

    def __init__(self, nn_linear: nn.Linear, dim_ratio: Union[int, float]=1.0):
        if False:
            while True:
                i = 10
        super().__init__()
        self.bias = nn.parameter.Parameter(nn_linear.bias.data, requires_grad=True)
        (u, vh) = self._spectral_init(nn_linear.weight.data, dim_ratio=dim_ratio)
        self.u = nn.parameter.Parameter(u, requires_grad=True)
        self.vh = nn.parameter.Parameter(vh, requires_grad=True)
        self.dim_ratio = dim_ratio
        self.in_features = u.size(0)
        self.out_features = vh.size(1)

    @staticmethod
    def _spectral_init(m, dim_ratio: Union[int, float]=1):
        if False:
            print('Hello World!')
        (u, s, vh) = torch.linalg.svd(m, full_matrices=False)
        u = u @ torch.diag(torch.sqrt(s))
        vh = torch.diag(torch.sqrt(s)) @ vh
        if dim_ratio < 1:
            dims = int(u.size(1) * dim_ratio)
            u = u[:, :dims]
            vh = vh[:dims, :]
        return (u, vh)

    def extra_repr(self) -> str:
        if False:
            print('Hello World!')
        'Extra representation log.'
        return f'in_features={self.in_features}, out_features={self.out_features}, bias=True, dim_ratio={self.dim_ratio}'

    def forward(self, x: torch.Tensor):
        if False:
            while True:
                i = 10
        'Forward call.'
        return x @ (self.u @ self.vh).transpose(0, 1) + self.bias
__all__ = ['FactorizedLinear']