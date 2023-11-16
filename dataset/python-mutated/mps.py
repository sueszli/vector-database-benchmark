from typing import Any, Dict, List, Optional, Union
import torch
from typing_extensions import override
from lightning.fabric.accelerators import _AcceleratorRegistry
from lightning.fabric.accelerators.mps import MPSAccelerator as _MPSAccelerator
from lightning.fabric.utilities.device_parser import _parse_gpu_ids
from lightning.fabric.utilities.types import _DEVICE
from lightning.pytorch.accelerators.accelerator import Accelerator
from lightning.pytorch.accelerators.cpu import _PSUTIL_AVAILABLE
from lightning.pytorch.utilities.exceptions import MisconfigurationException

class MPSAccelerator(Accelerator):
    """Accelerator for Metal Apple Silicon GPU devices.

    .. warning::  Use of this accelerator beyond import and instantiation is experimental.

    """

    @override
    def setup_device(self, device: torch.device) -> None:
        if False:
            print('Hello World!')
        '\n        Raises:\n            MisconfigurationException:\n                If the selected device is not MPS.\n        '
        if device.type != 'mps':
            raise MisconfigurationException(f'Device should be MPS, got {device} instead.')

    @override
    def get_device_stats(self, device: _DEVICE) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        'Get M1 (cpu + gpu) stats from ``psutil`` package.'
        return get_device_stats()

    @override
    def teardown(self) -> None:
        if False:
            return 10
        pass

    @staticmethod
    @override
    def parse_devices(devices: Union[int, str, List[int]]) -> Optional[List[int]]:
        if False:
            i = 10
            return i + 15
        'Accelerator device parsing logic.'
        return _parse_gpu_ids(devices, include_mps=True)

    @staticmethod
    @override
    def get_parallel_devices(devices: Union[int, str, List[int]]) -> List[torch.device]:
        if False:
            for i in range(10):
                print('nop')
        'Gets parallel devices for the Accelerator.'
        parsed_devices = MPSAccelerator.parse_devices(devices)
        assert parsed_devices is not None
        return [torch.device('mps', i) for i in range(len(parsed_devices))]

    @staticmethod
    @override
    def auto_device_count() -> int:
        if False:
            print('Hello World!')
        'Get the devices when set to auto.'
        return 1

    @staticmethod
    @override
    def is_available() -> bool:
        if False:
            i = 10
            return i + 15
        'MPS is only available on a machine with the ARM-based Apple Silicon processors.'
        return _MPSAccelerator.is_available()

    @classmethod
    @override
    def register_accelerators(cls, accelerator_registry: _AcceleratorRegistry) -> None:
        if False:
            for i in range(10):
                print('nop')
        accelerator_registry.register('mps', cls, description=cls.__name__)
_VM_PERCENT = 'M1_vm_percent'
_PERCENT = 'M1_percent'
_SWAP_PERCENT = 'M1_swap_percent'

def get_device_stats() -> Dict[str, float]:
    if False:
        print('Hello World!')
    if not _PSUTIL_AVAILABLE:
        raise ModuleNotFoundError(f'Fetching MPS device stats requires `psutil` to be installed. {str(_PSUTIL_AVAILABLE)}')
    import psutil
    return {_VM_PERCENT: psutil.virtual_memory().percent, _PERCENT: psutil.cpu_percent(), _SWAP_PERCENT: psutil.swap_memory().percent}