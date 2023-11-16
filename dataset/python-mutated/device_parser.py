from typing import List, MutableSequence, Optional, Tuple, Union
import torch
from lightning.fabric.utilities.exceptions import MisconfigurationException
from lightning.fabric.utilities.types import _DEVICE

def _determine_root_gpu_device(gpus: List[_DEVICE]) -> Optional[_DEVICE]:
    if False:
        print('Hello World!')
    '\n    Args:\n        gpus: Non-empty list of ints representing which GPUs to use\n\n    Returns:\n        Designated root GPU device id\n\n    Raises:\n        TypeError:\n            If ``gpus`` is not a list\n        AssertionError:\n            If GPU list is empty\n    '
    if gpus is None:
        return None
    if not isinstance(gpus, list):
        raise TypeError('GPUs should be a list')
    assert len(gpus) > 0, 'GPUs should be a non-empty list'
    return gpus[0]

def _parse_gpu_ids(gpus: Optional[Union[int, str, List[int]]], include_cuda: bool=False, include_mps: bool=False) -> Optional[List[int]]:
    if False:
        for i in range(10):
            print('nop')
    "Parses the GPU IDs given in the format as accepted by the :class:`~lightning.pytorch.trainer.trainer.Trainer`.\n\n    Args:\n        gpus: An int -1 or string '-1' indicate that all available GPUs should be used.\n            A list of unique ints or a string containing a list of comma separated unique integers\n            indicates specific GPUs to use.\n            An int of 0 means that no GPUs should be used.\n            Any int N > 0 indicates that GPUs [0..N) should be used.\n        include_cuda: A boolean value indicating whether to include CUDA devices for GPU parsing.\n        include_mps: A boolean value indicating whether to include MPS devices for GPU parsing.\n\n    Returns:\n        A list of GPUs to be used or ``None`` if no GPUs were requested\n\n    Raises:\n        MisconfigurationException:\n            If no GPUs are available but the value of gpus variable indicates request for GPUs\n\n    .. note::\n        ``include_cuda`` and ``include_mps`` default to ``False`` so that you only\n        have to specify which device type to use and all other devices are not disabled.\n\n    "
    _check_data_type(gpus)
    if gpus is None or (isinstance(gpus, int) and gpus == 0) or str(gpus).strip() in ('0', '[]'):
        return None
    gpus = _normalize_parse_gpu_string_input(gpus)
    gpus = _normalize_parse_gpu_input_to_list(gpus, include_cuda=include_cuda, include_mps=include_mps)
    if not gpus:
        raise MisconfigurationException('GPUs requested but none are available.')
    if torch.distributed.is_available() and torch.distributed.is_torchelastic_launched() and (len(gpus) != 1) and (len(_get_all_available_gpus(include_cuda=include_cuda, include_mps=include_mps)) == 1):
        return gpus
    _check_unique(gpus)
    return _sanitize_gpu_ids(gpus, include_cuda=include_cuda, include_mps=include_mps)

def _normalize_parse_gpu_string_input(s: Union[int, str, List[int]]) -> Union[int, List[int]]:
    if False:
        for i in range(10):
            print('nop')
    if not isinstance(s, str):
        return s
    if s == '-1':
        return -1
    if ',' in s:
        return [int(x.strip()) for x in s.split(',') if len(x) > 0]
    return int(s.strip())

def _sanitize_gpu_ids(gpus: List[int], include_cuda: bool=False, include_mps: bool=False) -> List[int]:
    if False:
        return 10
    'Checks that each of the GPUs in the list is actually available. Raises a MisconfigurationException if any of the\n    GPUs is not available.\n\n    Args:\n        gpus: List of ints corresponding to GPU indices\n\n    Returns:\n        Unmodified gpus variable\n\n    Raises:\n        MisconfigurationException:\n            If machine has fewer available GPUs than requested.\n\n    '
    if sum((include_cuda, include_mps)) == 0:
        raise ValueError('At least one gpu type should be specified!')
    all_available_gpus = _get_all_available_gpus(include_cuda=include_cuda, include_mps=include_mps)
    for gpu in gpus:
        if gpu not in all_available_gpus:
            raise MisconfigurationException(f'You requested gpu: {gpus}\n But your machine only has: {all_available_gpus}')
    return gpus

def _normalize_parse_gpu_input_to_list(gpus: Union[int, List[int], Tuple[int, ...]], include_cuda: bool, include_mps: bool) -> Optional[List[int]]:
    if False:
        print('Hello World!')
    assert gpus is not None
    if isinstance(gpus, (MutableSequence, tuple)):
        return list(gpus)
    if not gpus:
        return None
    if gpus == -1:
        return _get_all_available_gpus(include_cuda=include_cuda, include_mps=include_mps)
    return list(range(gpus))

def _get_all_available_gpus(include_cuda: bool=False, include_mps: bool=False) -> List[int]:
    if False:
        print('Hello World!')
    '\n    Returns:\n        A list of all available GPUs\n    '
    from lightning.fabric.accelerators.cuda import _get_all_visible_cuda_devices
    from lightning.fabric.accelerators.mps import _get_all_available_mps_gpus
    cuda_gpus = _get_all_visible_cuda_devices() if include_cuda else []
    mps_gpus = _get_all_available_mps_gpus() if include_mps else []
    return cuda_gpus + mps_gpus

def _check_unique(device_ids: List[int]) -> None:
    if False:
        for i in range(10):
            print('nop')
    "Checks that the device_ids are unique.\n\n    Args:\n        device_ids: List of ints corresponding to GPUs indices\n\n    Raises:\n        MisconfigurationException:\n            If ``device_ids`` of GPUs aren't unique\n\n    "
    if len(device_ids) != len(set(device_ids)):
        raise MisconfigurationException("Device ID's (GPU) must be unique.")

def _check_data_type(device_ids: object) -> None:
    if False:
        return 10
    "Checks that the device_ids argument is one of the following: int, string, or sequence of integers.\n\n    Args:\n        device_ids: gpus/tpu_cores parameter as passed to the Trainer\n\n    Raises:\n        TypeError:\n            If ``device_ids`` of GPU/TPUs aren't ``int``, ``str`` or sequence of ``int```\n\n    "
    msg = 'Device IDs (GPU/TPU) must be an int, a string, a sequence of ints, but you passed'
    if device_ids is None:
        raise TypeError(f'{msg} None')
    if isinstance(device_ids, (MutableSequence, tuple)):
        for id_ in device_ids:
            id_type = type(id_)
            if id_type is not int:
                raise TypeError(f'{msg} a sequence of {type(id_).__name__}.')
    elif type(device_ids) not in (int, str):
        raise TypeError(f'{msg} {device_ids!r}.')