import torch
from torch.utils import _pytree as pytree
from typing import Optional

def _basic_validation(op, args=(), kwargs=None):
    if False:
        print('Hello World!')
    '\n    Common validation across all ops go in here.\n    '
    from torch.distributed._shard.sharded_tensor import ShardedTensor
    if len(args) == 0 and (kwargs is None or len(kwargs) == 0):
        raise ValueError(f" No input for '{op.__name__}'!")
    has_distributed_tensor = False

    def is_distributed_tensor(e):
        if False:
            return 10
        nonlocal has_distributed_tensor
        if isinstance(e, ShardedTensor):
            has_distributed_tensor = True
    pytree.tree_map_(is_distributed_tensor, args)
    pytree.tree_map_(is_distributed_tensor, kwargs)
    if not has_distributed_tensor:
        raise TypeError(f"torch function '{op.__name__}', with args: {args} and kwargs: {kwargs} are called without any distributed tensor!")
    cur_pg: Optional[torch.distributed.ProcessGroup] = None

    def validate_pg(e):
        if False:
            print('Hello World!')
        nonlocal cur_pg
        if isinstance(e, ShardedTensor):
            if cur_pg is not None and e._process_group is not cur_pg:
                raise RuntimeError('All distributed tensors should use the same ProcessGroup if used together in an op.')
            cur_pg = e._process_group
    pytree.tree_map_(validate_pg, args)
    pytree.tree_map_(validate_pg, kwargs)

def _register_default_op(op, decorator):
    if False:
        for i in range(10):
            print('nop')

    @decorator(op)
    def tensor_default_op(types, args=(), kwargs=None, pg=None):
        if False:
            return 10
        '\n        Handles ``__torch_function__`` dispatch for the default tensor ops that\n        behave the same as ``torch.Tensor`` such as ``torch.Tensor.shape`` or\n        ``torch.Tensor.dtype``. We simply lower to the real op call with\n        DisableTorchFunctionSubclass context like ``torch.Tensor.__torch_function__``\n        to avoid recursions.\n        '
        if kwargs is None:
            kwargs = {}
        with torch._C.DisableTorchFunctionSubclass():
            return op(*args, **kwargs)