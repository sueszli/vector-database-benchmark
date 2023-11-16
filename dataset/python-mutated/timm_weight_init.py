import math
import warnings
import torch
from torch.nn.init import _calculate_fan_in_and_fan_out

def _trunc_normal_(tensor, mean, std, a, b):
    if False:
        while True:
            i = 10

    def norm_cdf(x):
        if False:
            print('Hello World!')
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0
    if mean < a - 2 * std or mean > b + 2 * std:
        warnings.warn('mean is more than 2 std from [a, b] in nn.init.trunc_normal_. The distribution of values may be incorrect.', stacklevel=2)
    l = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)
    tensor.uniform_(2 * l - 1, 2 * u - 1)
    tensor.erfinv_()
    tensor.mul_(std * math.sqrt(2.0))
    tensor.add_(mean)
    tensor.clamp_(min=a, max=b)
    return tensor

def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    if False:
        while True:
            i = 10
    'Fills the input Tensor with values drawn from a truncated\n    normal distribution. The values are effectively drawn from the\n    normal distribution :math:`\\mathcal{N}(\\text{mean}, \\text{std}^2)`\n    with values outside :math:`[a, b]` redrawn until they are within\n    the bounds. The method used for generating the random values works\n    best when :math:`a \\leq \\text{mean} \\leq b`.\n\n    NOTE: this impl is similar to the PyTorch trunc_normal_, the bounds [a, b] are\n    applied while sampling the normal with mean/std applied, therefore a, b args\n    should be adjusted to match the range of mean, std args.\n\n    Args:\n        tensor: an n-dimensional `torch.Tensor`\n        mean: the mean of the normal distribution\n        std: the standard deviation of the normal distribution\n        a: the minimum cutoff value\n        b: the maximum cutoff value\n    Examples:\n        >>> w = torch.empty(3, 5)\n        >>> nn.init.trunc_normal_(w)\n    '
    with torch.no_grad():
        return _trunc_normal_(tensor, mean, std, a, b)

def trunc_normal_tf_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    if False:
        for i in range(10):
            print('nop')
    "Fills the input Tensor with values drawn from a truncated\n    normal distribution. The values are effectively drawn from the\n    normal distribution :math:`\\mathcal{N}(\\text{mean}, \\text{std}^2)`\n    with values outside :math:`[a, b]` redrawn until they are within\n    the bounds. The method used for generating the random values works\n    best when :math:`a \\leq \\text{mean} \\leq b`.\n\n    NOTE: this 'tf' variant behaves closer to Tensorflow / JAX impl where the\n    bounds [a, b] are applied when sampling the normal distribution with mean=0, std=1.0\n    and the result is subsquently scaled and shifted by the mean and std args.\n\n    Args:\n        tensor: an n-dimensional `torch.Tensor`\n        mean: the mean of the normal distribution\n        std: the standard deviation of the normal distribution\n        a: the minimum cutoff value\n        b: the maximum cutoff value\n    Examples:\n        >>> w = torch.empty(3, 5)\n        >>> nn.init.trunc_normal_(w)\n    "
    with torch.no_grad():
        _trunc_normal_(tensor, 0, 1.0, a, b)
        tensor.mul_(std).add_(mean)
    return tensor

def variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='normal'):
    if False:
        return 10
    (fan_in, fan_out) = _calculate_fan_in_and_fan_out(tensor)
    if mode == 'fan_in':
        denom = fan_in
    elif mode == 'fan_out':
        denom = fan_out
    elif mode == 'fan_avg':
        denom = (fan_in + fan_out) / 2
    else:
        raise ValueError(f'invalid mode {mode}')
    variance = scale / denom
    if distribution == 'truncated_normal':
        trunc_normal_tf_(tensor, std=math.sqrt(variance) / 0.8796256610342398)
    elif distribution == 'normal':
        with torch.no_grad():
            tensor.normal_(std=math.sqrt(variance))
    elif distribution == 'uniform':
        bound = math.sqrt(3 * variance)
        with torch.no_grad():
            tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f'invalid distribution {distribution}')

def lecun_normal_(tensor):
    if False:
        i = 10
        return i + 15
    variance_scaling_(tensor, mode='fan_in', distribution='truncated_normal')