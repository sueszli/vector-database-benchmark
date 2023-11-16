import os
import logging
from typing import Optional, List, Tuple
from ray._private.accelerators.accelerator import AcceleratorManager
logger = logging.getLogger(__name__)
ONEAPI_DEVICE_SELECTOR_ENV_VAR = 'ONEAPI_DEVICE_SELECTOR'
NOSET_ONEAPI_DEVICE_SELECTOR_ENV_VAR = 'RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR'
ONEAPI_DEVICE_BACKEND_TYPE = 'level_zero'
ONEAPI_DEVICE_TYPE = 'gpu'

class IntelGPUAcceleratorManager(AcceleratorManager):
    """Intel GPU accelerators."""

    @staticmethod
    def get_resource_name() -> str:
        if False:
            for i in range(10):
                print('nop')
        return 'GPU'

    @staticmethod
    def get_visible_accelerator_ids_env_var() -> str:
        if False:
            return 10
        return ONEAPI_DEVICE_SELECTOR_ENV_VAR

    @staticmethod
    def get_current_process_visible_accelerator_ids() -> Optional[List[str]]:
        if False:
            while True:
                i = 10
        oneapi_visible_devices = os.environ.get(IntelGPUAcceleratorManager.get_visible_accelerator_ids_env_var(), None)
        if oneapi_visible_devices is None:
            return None
        if oneapi_visible_devices == '':
            return []
        if oneapi_visible_devices == 'NoDevFiles':
            return []
        prefix = ONEAPI_DEVICE_BACKEND_TYPE + ':'
        return list(oneapi_visible_devices.split(prefix)[1].split(','))

    @staticmethod
    def get_current_node_num_accelerators() -> int:
        if False:
            for i in range(10):
                print('nop')
        try:
            import dpctl
        except ImportError:
            dpctl = None
        if dpctl is None:
            return 0
        num_gpus = 0
        try:
            dev_info = ONEAPI_DEVICE_BACKEND_TYPE + ':' + ONEAPI_DEVICE_TYPE
            context = dpctl.SyclContext(dev_info)
            num_gpus = context.device_count
        except Exception:
            num_gpus = 0
        return num_gpus

    @staticmethod
    def get_current_node_accelerator_type() -> Optional[str]:
        if False:
            i = 10
            return i + 15
        "Get the name of first Intel GPU. (supposed only one GPU type on a node)\n        Example:\n            name: 'Intel(R) Data Center GPU Max 1550'\n            return name: 'Intel-GPU-Max-1550'\n        Returns:\n            A string representing the name of Intel GPU type.\n        "
        try:
            import dpctl
        except ImportError:
            dpctl = None
        if dpctl is None:
            return None
        accelerator_type = None
        try:
            dev_info = ONEAPI_DEVICE_BACKEND_TYPE + ':' + ONEAPI_DEVICE_TYPE + ':0'
            dev = dpctl.SyclDevice(dev_info)
            accelerator_type = 'Intel-GPU-' + '-'.join(dev.name.split(' ')[-2:])
        except Exception:
            accelerator_type = None
        return accelerator_type

    @staticmethod
    def validate_resource_request_quantity(quantity: float) -> Tuple[bool, Optional[str]]:
        if False:
            i = 10
            return i + 15
        return (True, None)

    @staticmethod
    def set_current_process_visible_accelerator_ids(visible_xpu_devices: List[str]) -> None:
        if False:
            i = 10
            return i + 15
        if os.environ.get(NOSET_ONEAPI_DEVICE_SELECTOR_ENV_VAR):
            return
        prefix = ONEAPI_DEVICE_BACKEND_TYPE + ':'
        os.environ[IntelGPUAcceleratorManager.get_visible_accelerator_ids_env_var()] = prefix + ','.join([str(i) for i in visible_xpu_devices])