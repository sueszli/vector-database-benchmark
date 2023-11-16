from typing import List, Dict, Tuple, Optional
import torch
from torch import Tensor
from torch.autograd.grad_mode import no_grad
from typing_extensions import TypeAlias

def _get_foreach_kernels_supported_devices() -> List[str]:
    if False:
        i = 10
        return i + 15
    'Return the device type list that supports foreach kernels.'
    return ['cuda', 'xpu', torch._C._get_privateuse1_backend_name()]

def _get_fused_kernels_supported_devices() -> List[str]:
    if False:
        return 10
    'Return the device type list that supports fused kernels in optimizer.'
    return ['cuda', 'xpu', torch._C._get_privateuse1_backend_name()]
TensorListList: TypeAlias = List[List[Optional[Tensor]]]
Indices: TypeAlias = List[int]

@no_grad()
def _group_tensors_by_device_and_dtype(tensorlistlist: TensorListList, with_indices: bool=False) -> Dict[Tuple[torch.device, torch.dtype], Tuple[TensorListList, Indices]]:
    if False:
        print('Hello World!')
    return {(device, getattr(torch, str_dtype)): value for ((device, str_dtype), value) in torch._C._group_tensors_by_device_and_dtype(tensorlistlist, with_indices).items()}

def _has_foreach_support(tensors: List[Tensor], device: torch.device) -> bool:
    if False:
        i = 10
        return i + 15
    if device.type not in set(_get_foreach_kernels_supported_devices() + ['cpu']) or torch.jit.is_scripting():
        return False
    return all((t is None or type(t) == torch.Tensor for t in tensors))