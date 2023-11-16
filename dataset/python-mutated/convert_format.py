from copy import deepcopy
from .. import functional as F
from ..core import _config
from ..module import Module
from ..tensor import Tensor

def _is_nchw_format(param: Tensor):
    if False:
        for i in range(10):
            print('nop')
    return (param.ndim == 4 or param.ndim == 5) and param.format != 'nhwc'

def convert_tensor_format(x: Tensor, inplace: bool=True):
    if False:
        i = 10
        return i + 15
    'Convert NCHW Tensor to NHWC Tensor.'
    if not _is_nchw_format(x):
        return x
    if x.ndim != 4 and x.ndim != 5:
        raise ValueError('Unsupport tensor ndim {}'.format(x.ndim))
    if x.format != 'nhwc':
        data = x.numpy()
        if inplace:
            x[...] = Tensor(data, format='nhwc')
        else:
            x = Tensor(data, format='nhwc')
    return x

def convert_module_format(module: Module, inplace: bool=True):
    if False:
        i = 10
        return i + 15
    'Convert NCHW Module to NHWC Module.'
    if not inplace:
        module = deepcopy(module)
    for (name, param) in module.named_tensors():
        convert_tensor_format(param, inplace=True)
    return module