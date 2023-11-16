import torch
from model import get_custom_op_library_path
torch.ops.load_library(get_custom_op_library_path())

@torch.library.impl_abstract('custom::nonzero')
def nonzero_abstract(x):
    if False:
        for i in range(10):
            print('nop')
    n = x.dim()
    ctx = torch.library.get_ctx()
    nnz = ctx.create_unbacked_symint()
    shape = [nnz, n]
    return x.new_empty(shape, dtype=torch.long)