from typing import Any, List, Tuple
import torch.nn as nn
from torch.distributed.tensor.parallel._data_parallel_utils import _flatten_tensor, _unflatten_tensor
__all__ = []

def _get_submodule_n_params(module: nn.Module, path: str):
    if False:
        while True:
            i = 10
    '\n    Get submodule and the direct path of parameter from the module\n    '
    if '.' in path:
        path_list = path.split('.')
        parent_module_path = '.'.join(path_list[:-1])
        module = module.get_submodule(parent_module_path)
        path = path_list[-1]
    return (module, path)

def _update_module_param(param_list: List[Tuple[nn.Module, str, nn.Parameter]]):
    if False:
        print('Hello World!')
    '\n    Update parameters within the module\n    '
    for item in param_list:
        (parent_module, module_path, t) = item
        assert hasattr(parent_module, module_path)
        delattr(parent_module, module_path)
        setattr(parent_module, module_path, t)

def _reconstruct_dtensor(module: nn.Module, _input: Any):
    if False:
        return 10
    '\n    Recontruct DTensor parameters from local tensors\n    '
    param_list = []
    for (name, t) in module.named_parameters():
        if hasattr(t, '_st_info'):
            dtensor = _unflatten_tensor(t, t._st_info)
            param_list.append((*_get_submodule_n_params(module, name), dtensor))
    _update_module_param(param_list)

def _localize_dtensor(module: nn.Module, *_: Any):
    if False:
        while True:
            i = 10
    '\n    Convert DTensor parameters to local tensors\n    '
    param_list = []
    for (name, param) in module.named_parameters():
        (t, sharding_info) = _flatten_tensor(param)
        if sharding_info is not None:
            t = nn.Parameter(t)
            t._st_info = sharding_info
            param_list.append((*_get_submodule_n_params(module, name), t))
    _update_module_param(param_list)

def _pre_dp_module_transform(module: nn.Module):
    if False:
        for i in range(10):
            print('nop')
    '\n    Enable the composability between Tensor Parallelism (TP) and Data\n    Parallelism(DP) in PyTorch when using DDP. We need to convert Parameters which\n    are DTensors to local tensors before wrapping with data parallelism API.\n    We then register two hooks, one for converting local tensors back to DTensor\n    preforward and one to convert DTensors back to tensors after Forward. By\n    integrating this way, we avoid any special handling of DTensor parameters by DDP\n    and get DTensor\'s gradients propagated back to DP, e.g. gradient buckets of DDP.\n\n    For now, this API only works with ``DistributedDataParallel``. It will later support\n    other DP methods such as FSDP.\n\n    Args:\n        module (:class:`nn.Module`):\n            Module which has been applied TP on.\n\n    Example::\n        >>> # xdoctest: +SKIP("distributed")\n        >>> from torch.distributed.tensor.parallel import parallelize_module, PairwiseParallel\n        >>> from torch.nn.parallel import DistributedDataParallel as DDP\n        >>> from torch.distributed.tensor.parallel.ddp import pre_dp_module_transform\n        >>>\n        >>> # Define the module.\n        >>> m = module(...)\n        >>> parallelize_module(m, PairwiseParallel())\n        >>> m = pre_dp_module_transform(m)\n        >>> m = DDP(m)\n        >>>\n    '
    _localize_dtensor(module, None, None)
    module.register_forward_pre_hook(_reconstruct_dtensor)
    module.register_forward_hook(_localize_dtensor)