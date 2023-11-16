import torch
from model import get_custom_op_library_path
torch.ops.load_library(get_custom_op_library_path())

@torch.library.impl_abstract('custom::sin')
def sin_abstract(x):
    if False:
        i = 10
        return i + 15
    return torch.empty_like(x)