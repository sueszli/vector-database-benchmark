import os
from typing import Any, List, Optional
import torch
try:
    import horovod.torch
    _HVD = horovod.torch
except (ModuleNotFoundError, ImportError):
    _HVD = None

def initialize_horovod():
    if False:
        for i in range(10):
            print('nop')
    if not _HVD:
        raise ValueError('Horovod backend specified, but cannot import `horovod.torch`. Install Horovod following the instructions at: https://github.com/horovod/horovod')
    _HVD.init()
    return _HVD

def has_horovodrun():
    if False:
        return 10
    'Returns True if running with `horovodrun` using Gloo or OpenMPI.'
    return 'OMPI_COMM_WORLD_RANK' in os.environ or 'HOROVOD_RANK' in os.environ

def gather_all_tensors(result: torch.Tensor, group: Optional[Any]=None) -> List[torch.Tensor]:
    if False:
        print('Hello World!')
    'Function to gather all tensors from several processes onto a list that is broadcast to all processes.\n\n    Works on tensors that have the same number of dimensions, but where each dimension may differ. In this case\n    tensors are padded, gathered and then trimmed to secure equal workload for all processes.\n\n    :param result: the value to sync\n    :param group: the process group to gather results from (not supported: always uses world)\n\n    :return: list with size equal to the process group where gathered_result[i]\n             corresponds to result tensor from process i\n    '
    if group is not None:
        raise ValueError('Horovod does not support allgather using a subcommunicator at this time. Unset `group`.')
    if _HVD is None or not _HVD.is_initialized():
        return [result]
    if len(result.shape) == 0:
        result = result.reshape(1)
    is_bool = False
    if result.dtype == torch.bool:
        result = result.int()
        is_bool = True
    result = result.unsqueeze(0)
    gathered_result = _HVD.allgather(result)
    gathered_result = list(gathered_result)
    if is_bool:
        gathered_result = [t.bool() for t in gathered_result]
    return gathered_result

def is_distributed_available() -> bool:
    if False:
        i = 10
        return i + 15
    return _HVD is not None and (_HVD.is_initialized() or os.environ.get('HOROVOD_RANK'))