import torch
import torch.nn as nn
import torch.ao.nn.intrinsic as nni
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.utils.fusion import fuse_linear_bn_weights
__all__ = ['LinearBn1d']

class LinearBn1d(nn.modules.linear.Linear, nni._FusedModule):
    """
    A LinearBn1d module is a module fused from Linear and BatchNorm1d, attached
    with FakeQuantize modules for weight, used in quantization aware training.

    We combined the interface of :class:`torch.nn.Linear` and
    :class:torch.nn.BatchNorm1d`.

    Similar to :class:`torch.nn.Linear`, with FakeQuantize modules initialized
    to default.

    Attributes:
        freeze_bn:
        weight_fake_quant: fake quant module for weight

    """

    def __init__(self, in_features, out_features, bias=True, eps=1e-05, momentum=0.1, freeze_bn=False, qconfig=None):
        if False:
            return 10
        nn.modules.linear.Linear.__init__(self, in_features, out_features, bias)
        assert qconfig, 'qconfig must be provided for QAT module'
        self.qconfig = qconfig
        self.freeze_bn = freeze_bn if self.training else True
        self.bn = nn.BatchNorm1d(out_features, eps, momentum, True, True)
        self.weight_fake_quant = self.qconfig.weight()
        if bias:
            self.bias = Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_bn_parameters()
        if self.training:
            if freeze_bn:
                self.freeze_bn_stats()
            else:
                self.update_bn_stats()
        else:
            self.freeze_bn_stats()

    def reset_running_stats(self):
        if False:
            return 10
        self.bn.reset_running_stats()

    def reset_bn_parameters(self):
        if False:
            print('Hello World!')
        self.bn.reset_running_stats()
        init.uniform_(self.bn.weight)
        init.zeros_(self.bn.bias)

    def reset_parameters(self):
        if False:
            for i in range(10):
                print('nop')
        super().reset_parameters()

    def update_bn_stats(self):
        if False:
            i = 10
            return i + 15
        self.freeze_bn = False
        self.bn.training = True
        return self

    def freeze_bn_stats(self):
        if False:
            print('Hello World!')
        self.freeze_bn = True
        self.bn.training = False
        return self

    def forward(self, input):
        if False:
            while True:
                i = 10
        assert self.bn.running_var is not None
        running_std = torch.sqrt(self.bn.running_var + self.bn.eps)
        scale_factor = self.bn.weight / running_std
        weight_shape = [1] * len(self.weight.shape)
        weight_shape[0] = -1
        bias_shape = [1] * len(self.weight.shape)
        bias_shape[1] = -1
        scaled_weight = self.weight_fake_quant(self.weight * scale_factor.reshape(weight_shape))
        if self.bias is not None:
            zero_bias = torch.zeros_like(self.bias)
        else:
            zero_bias = torch.zeros(self.out_features, device=scaled_weight.device)
        linear_out = F.linear(input, scaled_weight, zero_bias)
        linear_out_orig = linear_out / scale_factor.reshape(bias_shape)
        if self.bias is not None:
            linear_out_orig = linear_out_orig + self.bias.reshape(bias_shape)
        bn_out = self.bn(linear_out_orig)
        return bn_out

    def train(self, mode=True):
        if False:
            return 10
        "\n        Batchnorm's training behavior is using the self.training flag. Prevent\n        changing it if BN is frozen. This makes sure that calling `model.train()`\n        on a model with a frozen BN will behave properly.\n        "
        self.training = mode
        if not self.freeze_bn:
            for module in self.children():
                module.train(mode)
        return self

    @classmethod
    def from_float(cls, mod):
        if False:
            return 10
        "Create a qat module from a float module or qparams_dict\n\n            Args: `mod' a float module, either produced by torch.ao.quantization\n            utilities or directly from user\n        "
        assert type(mod) == nni.LinearBn1d, 'qat.' + cls.__name__ + '.from_float only works for ' + nni.LinearBn1d.__name__
        assert hasattr(mod, 'qconfig'), 'Input float module must have qconfig defined'
        assert mod.qconfig, 'Input float module must have a valid config'
        qconfig = mod.qconfig
        (linear, bn) = (mod[0], mod[1])
        qat_linearbn = cls(linear.in_features, linear.out_features, linear.bias is not None, bn.eps, bn.momentum, False, qconfig)
        qat_linearbn.weight = linear.weight
        qat_linearbn.bias = linear.bias
        qat_linearbn.bn.weight = bn.weight
        qat_linearbn.bn.bias = bn.bias
        qat_linearbn.bn.running_mean = bn.running_mean
        qat_linearbn.bn.running_var = bn.running_var
        qat_linearbn.bn.num_batches_tracked = bn.num_batches_tracked
        return qat_linearbn

    def to_float(self):
        if False:
            for i in range(10):
                print('nop')
        linear = torch.nn.Linear(self.in_features, self.out_features)
        assert self.bn.running_var is not None and self.bn.running_mean is not None
        (linear.weight, linear.bias) = fuse_linear_bn_weights(self.weight, self.bias, self.bn.running_mean, self.bn.running_var, self.bn.eps, self.bn.weight, self.bn.bias)
        return linear