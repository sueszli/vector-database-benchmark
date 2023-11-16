"""
MIT License

Copyright (c) 2020 Phil Wang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """An alternate to layer normalization, without mean centering and the learned bias [1]

    References
    ----------
    .. [1] Zhang, Biao, and Rico Sennrich. "Root mean square layer normalization." Advances in Neural Information
           Processing Systems 32 (2019).
    """

    def __init__(self, dim, eps=1e-08):
        if False:
            print('Hello World!')
        super().__init__()
        self.scale = dim ** (-0.5)
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        if False:
            for i in range(10):
                print('nop')
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g

class LayerNormNoBias(nn.LayerNorm):

    def __init__(self, input_size, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(input_size, elementwise_affine=False, **kwargs)

class LayerNorm(nn.LayerNorm):

    def __init__(self, input_size, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(input_size, **kwargs)

class RINorm(nn.Module):

    def __init__(self, input_dim: int, eps=1e-05, affine=True):
        if False:
            for i in range(10):
                print('nop')
        'Reversible Instance Normalization based on [1]\n\n        Parameters\n        ----------\n        input_dim\n            The dimension of the input axis being normalized\n        eps\n            The epsilon value for numerical stability\n        affine\n            Whether to apply an affine transformation after normalization\n\n        References\n        ----------\n        .. [1] Kim et al. "Reversible Instance Normalization for Accurate Time-Series Forecasting against\n                Distribution Shift" International Conference on Learning Representations (2022)\n        '
        super().__init__()
        self.input_dim = input_dim
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(self.input_dim))
            self.affine_bias = nn.Parameter(torch.zeros(self.input_dim))

    def forward(self, x: torch.Tensor):
        if False:
            while True:
                i = 10
        calc_dims = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=calc_dims, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=calc_dims, keepdim=True, unbiased=False) + self.eps).detach()
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def inverse(self, x: torch.Tensor):
        if False:
            for i in range(10):
                print('nop')
        if self.affine:
            x = x - self.affine_bias.view(self.affine_bias.shape + (1,))
            x = x / (self.affine_weight.view(self.affine_weight.shape + (1,)) + self.eps * self.eps)
        x = x * self.stdev.view(self.stdev.shape + (1,))
        x = x + self.mean.view(self.mean.shape + (1,))
        return x