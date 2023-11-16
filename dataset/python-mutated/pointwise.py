import torch
from model import get_custom_op_library_path
torch.ops.load_library(get_custom_op_library_path())

@torch.library.impl_abstract('custom::cos')
def cos_abstract(x):
    if False:
        while True:
            i = 10
    return torch.empty_like(x)

@torch.library.impl_abstract('custom::tan')
def tan_abstract(x):
    if False:
        print('Hello World!')
    return torch.empty_like(x)