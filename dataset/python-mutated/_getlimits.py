import torch
from . import _dtypes

def finfo(dtyp):
    if False:
        for i in range(10):
            print('nop')
    torch_dtype = _dtypes.dtype(dtyp).torch_dtype
    return torch.finfo(torch_dtype)

def iinfo(dtyp):
    if False:
        print('Hello World!')
    torch_dtype = _dtypes.dtype(dtyp).torch_dtype
    return torch.iinfo(torch_dtype)