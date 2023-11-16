import os
import logging
from typing import Optional, List, Tuple
from functools import lru_cache
from importlib.util import find_spec
from ray._private.accelerators.accelerator import AcceleratorManager
logger = logging.getLogger(__name__)
HABANA_VISIBLE_DEVICES_ENV_VAR = 'HABANA_VISIBLE_MODULES'
NOSET_HABANA_VISIBLE_MODULES_ENV_VAR = 'RAY_EXPERIMENTAL_NOSET_HABANA_VISIBLE_MODULES'

@lru_cache()
def is_package_present(package_name: str) -> bool:
    if False:
        print('Hello World!')
    try:
        return find_spec(package_name) is not None
    except ModuleNotFoundError:
        return False
HPU_PACKAGE_AVAILABLE = is_package_present('habana_frameworks')

class HPUAcceleratorManager(AcceleratorManager):
    """Intel Habana(HPU) accelerators."""

    @staticmethod
    def get_resource_name() -> str:
        if False:
            for i in range(10):
                print('nop')
        return 'HPU'

    @staticmethod
    def get_visible_accelerator_ids_env_var() -> str:
        if False:
            i = 10
            return i + 15
        return HABANA_VISIBLE_DEVICES_ENV_VAR

    @staticmethod
    def get_current_process_visible_accelerator_ids() -> Optional[List[str]]:
        if False:
            return 10
        hpu_visible_devices = os.environ.get(HPUAcceleratorManager.get_visible_accelerator_ids_env_var(), None)
        if hpu_visible_devices is None:
            return None
        if hpu_visible_devices == '':
            return []
        return list(hpu_visible_devices.split(','))

    @staticmethod
    def get_current_node_num_accelerators() -> int:
        if False:
            for i in range(10):
                print('nop')
        'Attempt to detect the number of HPUs on this machine.\n        Returns:\n            The number of HPUs if any were detected, otherwise 0.\n        '
        if HPU_PACKAGE_AVAILABLE:
            import habana_frameworks.torch.hpu as torch_hpu
            if torch_hpu.is_available():
                return torch_hpu.device_count()
            else:
                logging.info('HPU devices not available')
                return 0
        else:
            return 0

    @staticmethod
    def is_initialized() -> bool:
        if False:
            return 10
        'Attempt to check if HPU backend is initialized.\n        Returns:\n            True if backend initialized else False.\n        '
        if HPU_PACKAGE_AVAILABLE:
            import habana_frameworks.torch.hpu as torch_hpu
            if torch_hpu.is_available() and torch_hpu.is_initialized():
                return True
            else:
                return False
        else:
            return False

    @staticmethod
    def get_current_node_accelerator_type() -> Optional[str]:
        if False:
            return 10
        'Attempt to detect the HPU family type.\n        Returns:\n            The device name (GAUDI, GAUDI2) if detected else None.\n        '
        if HPUAcceleratorManager.is_initialized():
            import habana_frameworks.torch.hpu as torch_hpu
            return f'Intel-{torch_hpu.get_device_name()}'
        else:
            logging.info('HPU type cannot be detected')
            return None

    @staticmethod
    def validate_resource_request_quantity(quantity: float) -> Tuple[bool, Optional[str]]:
        if False:
            return 10
        if isinstance(quantity, float) and (not quantity.is_integer()):
            return (False, f'{HPUAcceleratorManager.get_resource_name()} resource quantity must be whole numbers. The specified quantity {quantity} is invalid.')
        else:
            return (True, None)

    @staticmethod
    def set_current_process_visible_accelerator_ids(visible_hpu_devices: List[str]) -> None:
        if False:
            for i in range(10):
                print('nop')
        if os.environ.get(NOSET_HABANA_VISIBLE_MODULES_ENV_VAR):
            return
        os.environ[HPUAcceleratorManager.get_visible_accelerator_ids_env_var()] = ','.join([str(i) for i in visible_hpu_devices])