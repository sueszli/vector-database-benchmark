from __future__ import annotations
import copy
from typing import Optional, Tuple, TypeVar
import torch
__all__ = ['fuse_conv_bn_eval', 'fuse_conv_bn_weights', 'fuse_linear_bn_eval', 'fuse_linear_bn_weights']
ConvT = TypeVar('ConvT', bound='torch.nn.modules.conv._ConvNd')
LinearT = TypeVar('LinearT', bound='torch.nn.Linear')

def fuse_conv_bn_eval(conv: ConvT, bn: torch.nn.modules.batchnorm._BatchNorm, transpose: bool=False) -> ConvT:
    if False:
        return 10
    'Fuse a convolutional module and a BatchNorm module into a single, new convolutional module.\n\n    Args:\n        conv (torch.nn.modules.conv._ConvNd): A convolutional module.\n        bn (torch.nn.modules.batchnorm._BatchNorm): A BatchNorm module.\n        transpose (bool, optional): If True, transpose the convolutional weight. Defaults to False.\n\n    Returns:\n        torch.nn.modules.conv._ConvNd: The fused convolutional module.\n\n    .. note::\n        Both ``conv`` and ``bn`` must be in eval mode, and ``bn`` must have its running buffers computed.\n    '
    assert not (conv.training or bn.training), 'Fusion only for eval!'
    fused_conv = copy.deepcopy(conv)
    assert bn.running_mean is not None and bn.running_var is not None
    (fused_conv.weight, fused_conv.bias) = fuse_conv_bn_weights(fused_conv.weight, fused_conv.bias, bn.running_mean, bn.running_var, bn.eps, bn.weight, bn.bias, transpose)
    return fused_conv

def fuse_conv_bn_weights(conv_w: torch.Tensor, conv_b: Optional[torch.Tensor], bn_rm: torch.Tensor, bn_rv: torch.Tensor, bn_eps: float, bn_w: Optional[torch.Tensor], bn_b: Optional[torch.Tensor], transpose: bool=False) -> Tuple[torch.nn.Parameter, torch.nn.Parameter]:
    if False:
        for i in range(10):
            print('nop')
    'Fuse convolutional module parameters and BatchNorm module parameters into new convolutional module parameters.\n\n    Args:\n        conv_w (torch.Tensor): Convolutional weight.\n        conv_b (Optional[torch.Tensor]): Convolutional bias.\n        bn_rm (torch.Tensor): BatchNorm running mean.\n        bn_rv (torch.Tensor): BatchNorm running variance.\n        bn_eps (float): BatchNorm epsilon.\n        bn_w (Optional[torch.Tensor]): BatchNorm weight.\n        bn_b (Optional[torch.Tensor]): BatchNorm bias.\n        transpose (bool, optional): If True, transpose the conv weight. Defaults to False.\n\n    Returns:\n        Tuple[torch.nn.Parameter, torch.nn.Parameter]: Fused convolutional weight and bias.\n    '
    conv_weight_dtype = conv_w.dtype
    conv_bias_dtype = conv_b.dtype if conv_b is not None else conv_weight_dtype
    if conv_b is None:
        conv_b = torch.zeros_like(bn_rm)
    if bn_w is None:
        bn_w = torch.ones_like(bn_rm)
    if bn_b is None:
        bn_b = torch.zeros_like(bn_rm)
    bn_var_rsqrt = torch.rsqrt(bn_rv + bn_eps)
    if transpose:
        shape = [1, -1] + [1] * (len(conv_w.shape) - 2)
    else:
        shape = [-1, 1] + [1] * (len(conv_w.shape) - 2)
    fused_conv_w = (conv_w * (bn_w * bn_var_rsqrt).reshape(shape)).to(dtype=conv_weight_dtype)
    fused_conv_b = ((conv_b - bn_rm) * bn_var_rsqrt * bn_w + bn_b).to(dtype=conv_bias_dtype)
    return (torch.nn.Parameter(fused_conv_w, conv_w.requires_grad), torch.nn.Parameter(fused_conv_b, conv_b.requires_grad))

def fuse_linear_bn_eval(linear: LinearT, bn: torch.nn.modules.batchnorm._BatchNorm) -> LinearT:
    if False:
        return 10
    'Fuse a linear module and a BatchNorm module into a single, new linear module.\n\n    Args:\n        linear (torch.nn.Linear): A Linear module.\n        bn (torch.nn.modules.batchnorm._BatchNorm): A BatchNorm module.\n\n    Returns:\n        torch.nn.Linear: The fused linear module.\n\n    .. note::\n        Both ``linear`` and ``bn`` must be in eval mode, and ``bn`` must have its running buffers computed.\n    '
    assert not (linear.training or bn.training), 'Fusion only for eval!'
    fused_linear = copy.deepcopy(linear)
    assert bn.running_mean is not None and bn.running_var is not None
    (fused_linear.weight, fused_linear.bias) = fuse_linear_bn_weights(fused_linear.weight, fused_linear.bias, bn.running_mean, bn.running_var, bn.eps, bn.weight, bn.bias)
    return fused_linear

def fuse_linear_bn_weights(linear_w: torch.Tensor, linear_b: Optional[torch.Tensor], bn_rm: torch.Tensor, bn_rv: torch.Tensor, bn_eps: float, bn_w: torch.Tensor, bn_b: torch.Tensor) -> Tuple[torch.nn.Parameter, torch.nn.Parameter]:
    if False:
        for i in range(10):
            print('nop')
    'Fuse linear module parameters and BatchNorm module parameters into new linear module parameters.\n\n    Args:\n        linear_w (torch.Tensor): Linear weight.\n        linear_b (Optional[torch.Tensor]): Linear bias.\n        bn_rm (torch.Tensor): BatchNorm running mean.\n        bn_rv (torch.Tensor): BatchNorm running variance.\n        bn_eps (float): BatchNorm epsilon.\n        bn_w (torch.Tensor): BatchNorm weight.\n        bn_b (torch.Tensor): BatchNorm bias.\n        transpose (bool, optional): If True, transpose the conv weight. Defaults to False.\n\n    Returns:\n        Tuple[torch.nn.Parameter, torch.nn.Parameter]: Fused linear weight and bias.\n    '
    if linear_b is None:
        linear_b = torch.zeros_like(bn_rm)
    bn_scale = bn_w * torch.rsqrt(bn_rv + bn_eps)
    fused_w = linear_w * bn_scale.unsqueeze(-1)
    fused_b = (linear_b - bn_rm) * bn_scale + bn_b
    return (torch.nn.Parameter(fused_w, linear_w.requires_grad), torch.nn.Parameter(fused_b, linear_b.requires_grad))