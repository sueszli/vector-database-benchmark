import torch
from torch import nn
from torch.nn.modules.utils import _single
from torch import Tensor

class ConvTBC(torch.nn.Module):
    """1D convolution over an input of shape (time x batch x channel)

    The implementation uses gemm to perform the convolution. This implementation
    is faster than cuDNN for small kernel sizes.
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        if False:
            i = 10
            return i + 15
        super(ConvTBC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _single(kernel_size)
        self.padding = _single(padding)
        self.weight = torch.nn.Parameter(torch.Tensor(self.kernel_size[0], in_channels, out_channels))
        self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        if False:
            for i in range(10):
                print('nop')
        nn.init.xavier_normal_(self.weight)
        nn.init.zeros_(self.bias)

    def conv_tbc(self, input: Tensor):
        if False:
            for i in range(10):
                print('nop')
        return torch.conv_tbc(input.contiguous(), self.weight, self.bias, self.padding[0])

    def forward(self, input: Tensor):
        if False:
            while True:
                i = 10
        return self.conv_tbc(input)

    def __repr__(self):
        if False:
            print('Hello World!')
        s = '{name}({in_channels}, {out_channels}, kernel_size={kernel_size}, padding={padding}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)