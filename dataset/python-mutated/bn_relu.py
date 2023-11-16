import torch
import torch.ao.nn.intrinsic
import torch.ao.nn.intrinsic.qat
import torch.ao.nn.quantized as nnq
__all__ = ['BNReLU2d', 'BNReLU3d']

class BNReLU2d(nnq.BatchNorm2d):
    """
    A BNReLU2d module is a fused module of BatchNorm2d and ReLU

    We adopt the same interface as :class:`torch.ao.nn.quantized.BatchNorm2d`.

    Attributes:
        Same as torch.ao.nn.quantized.BatchNorm2d

    """
    _FLOAT_MODULE = torch.ao.nn.intrinsic.BNReLU2d

    def __init__(self, num_features, eps=1e-05, momentum=0.1, device=None, dtype=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(num_features, eps=eps, momentum=momentum, device=device, dtype=dtype)

    def forward(self, input):
        if False:
            for i in range(10):
                print('nop')
        if len(input.shape) != 4:
            raise ValueError('Input shape must be `(N, C, H, W)`!')
        return torch.ops.quantized.batch_norm2d_relu(input, self.weight, self.bias, self.running_mean, self.running_var, self.eps, self.scale, self.zero_point)

    def _get_name(self):
        if False:
            i = 10
            return i + 15
        return 'QuantizedBNReLU2d'

    @classmethod
    def from_float(cls, mod):
        if False:
            print('Hello World!')
        return super().from_float(mod)

    @classmethod
    def from_reference(cls, bn_relu, output_scale, output_zero_point):
        if False:
            i = 10
            return i + 15
        return super().from_reference(bn_relu[0], output_scale, output_zero_point)

class BNReLU3d(nnq.BatchNorm3d):
    """
    A BNReLU3d module is a fused module of BatchNorm3d and ReLU

    We adopt the same interface as :class:`torch.ao.nn.quantized.BatchNorm3d`.

    Attributes:
        Same as torch.ao.nn.quantized.BatchNorm3d

    """
    _FLOAT_MODULE = torch.ao.nn.intrinsic.BNReLU3d

    def __init__(self, num_features, eps=1e-05, momentum=0.1, device=None, dtype=None):
        if False:
            print('Hello World!')
        super().__init__(num_features, eps=eps, momentum=momentum, device=device, dtype=dtype)

    def forward(self, input):
        if False:
            print('Hello World!')
        if len(input.shape) != 5:
            raise ValueError('Input shape must be `(N, C, D, H, W)`!')
        return torch.ops.quantized.batch_norm3d_relu(input, self.weight, self.bias, self.running_mean, self.running_var, self.eps, self.scale, self.zero_point)

    def _get_name(self):
        if False:
            return 10
        return 'QuantizedBNReLU3d'

    @classmethod
    def from_float(cls, mod):
        if False:
            print('Hello World!')
        return super().from_float(mod)

    @classmethod
    def from_reference(cls, bn_relu, output_scale, output_zero_point):
        if False:
            return 10
        return super().from_reference(bn_relu[0], output_scale, output_zero_point)