import torch
from typing import Iterable, Optional

def parameters_to_vector(parameters: Iterable[torch.Tensor]) -> torch.Tensor:
    if False:
        for i in range(10):
            print('nop')
    'Flatten an iterable of parameters into a single vector.\n\n    Args:\n        parameters (Iterable[Tensor]): an iterable of Tensors that are the\n            parameters of a model.\n\n    Returns:\n        The parameters represented by a single vector\n    '
    param_device = None
    vec = []
    for param in parameters:
        param_device = _check_param_device(param, param_device)
        vec.append(param.view(-1))
    return torch.cat(vec)

def vector_to_parameters(vec: torch.Tensor, parameters: Iterable[torch.Tensor]) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Copy slices of a vector into an iterable of parameters.\n\n    Args:\n        vec (Tensor): a single vector representing the parameters of a model.\n        parameters (Iterable[Tensor]): an iterable of Tensors that are the\n            parameters of a model.\n    '
    if not isinstance(vec, torch.Tensor):
        raise TypeError(f'expected torch.Tensor, but got: {torch.typename(vec)}')
    param_device = None
    pointer = 0
    for param in parameters:
        param_device = _check_param_device(param, param_device)
        num_param = param.numel()
        param.data = vec[pointer:pointer + num_param].view_as(param).data
        pointer += num_param

def _check_param_device(param: torch.Tensor, old_param_device: Optional[int]) -> int:
    if False:
        while True:
            i = 10
    'Check if the parameters are located on the same device.\n\n    Currently, the conversion between model parameters and single vector form is not supported\n    for multiple allocations, e.g. parameters in different GPUs/PrivateUse1s, or mixture of CPU/GPU/PrivateUse1.\n\n    Args:\n        param ([Tensor]): a Tensor of a parameter of a model\n        old_param_device (int): the device where the first parameter of a\n                                model is allocated.\n\n    Returns:\n        old_param_device (int): report device for the first time\n    '
    support_device_types = ['cuda', torch._C._get_privateuse1_backend_name()]
    if old_param_device is None:
        old_param_device = param.get_device() if param.device.type in support_device_types else -1
    else:
        warn = False
        if param.device.type in support_device_types:
            warn = param.get_device() != old_param_device
        else:
            warn = old_param_device != -1
        if warn:
            raise TypeError('Found two parameters on different devices, this is currently not supported.')
    return old_param_device