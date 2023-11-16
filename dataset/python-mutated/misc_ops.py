import torch
from torch.distributed._shard.sharded_tensor import _sharded_op_impl

@_sharded_op_impl(torch._has_compatible_shallow_copy_type)
def tensor_has_compatible_shallow_copy_type(types, args=(), kwargs=None, pg=None):
    if False:
        while True:
            i = 10
    return False