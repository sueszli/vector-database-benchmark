import pytest
from typing import List
from unittest.mock import patch
import ray._private.thirdparty.pynvml as pynvml

class DeviceHandleMock(dict):

    def __init__(self, name: str, uuid: str, mig_devices: List['DeviceHandleMock']=None, **kwargs):
        if False:
            return 10
        super().__init__()
        self['name'] = name
        self['uuid'] = uuid
        if mig_devices is not None:
            self['mig_devices'] = mig_devices
        self.update(kwargs)

class PyNVMLMock:

    def __init__(self, mock_data, driver_version='535.104.12'):
        if False:
            for i in range(10):
                print('nop')
        self._mock_data = mock_data
        self.driver_version = driver_version

    def nvmlInit(self):
        if False:
            return 10
        return

    def nvmlShutdown(self):
        if False:
            for i in range(10):
                print('nop')
        return

    def nvmlSystemGetDriverVersion(self):
        if False:
            for i in range(10):
                print('nop')
        return self.driver_version

    def nvmlDeviceGetCount(self):
        if False:
            return 10
        return len(self._mock_data)

    def nvmlDeviceGetHandleByIndex(self, index):
        if False:
            for i in range(10):
                print('nop')
        return self._mock_data[index]

    def nvmlDeviceGetName(self, handle):
        if False:
            for i in range(10):
                print('nop')
        return handle.get('name', '')

    def nvmlDeviceGetMaxMigDeviceCount(self, handle):
        if False:
            print('Hello World!')
        if 'mig_devices' in handle:
            return max(7, len(handle['mig_devices']))
        else:
            raise pynvml.NVMLError_NotSupported

    def nvmlDeviceGetMigDeviceHandleByIndex(self, handle, mig_index):
        if False:
            print('Hello World!')
        try:
            return handle['mig_devices'][mig_index]
        except IndexError:
            raise pynvml.NVMLError_NotFound

    def nvmlDeviceGetUUID(self, handle):
        if False:
            print('Hello World!')
        return handle.get('uuid', '')

    def nvmlDeviceGetComputeInstanceId(self, mig_handle):
        if False:
            i = 10
            return i + 15
        return mig_handle['ci_id']

    def nvmlDeviceGetGpuInstanceId(self, mig_handle):
        if False:
            for i in range(10):
                print('nop')
        return mig_handle['gi_id']

@pytest.fixture
def patch_mock_pynvml(mock_nvml):
    if False:
        return 10
    with patch('ray._private.thirdparty.pynvml.nvmlInit', mock_nvml.nvmlInit), patch('ray._private.thirdparty.pynvml.nvmlShutdown', mock_nvml.nvmlShutdown), patch('ray._private.thirdparty.pynvml.nvmlSystemGetDriverVersion', mock_nvml.nvmlSystemGetDriverVersion), patch('ray._private.thirdparty.pynvml.nvmlDeviceGetCount', mock_nvml.nvmlDeviceGetCount), patch('ray._private.thirdparty.pynvml.nvmlDeviceGetHandleByIndex', mock_nvml.nvmlDeviceGetHandleByIndex), patch('ray._private.thirdparty.pynvml.nvmlDeviceGetName', mock_nvml.nvmlDeviceGetName), patch('ray._private.thirdparty.pynvml.nvmlDeviceGetMaxMigDeviceCount', mock_nvml.nvmlDeviceGetMaxMigDeviceCount), patch('ray._private.thirdparty.pynvml.nvmlDeviceGetMigDeviceHandleByIndex', mock_nvml.nvmlDeviceGetMigDeviceHandleByIndex), patch('ray._private.thirdparty.pynvml.nvmlDeviceGetUUID', mock_nvml.nvmlDeviceGetUUID), patch('ray._private.thirdparty.pynvml.nvmlDeviceGetComputeInstanceId', mock_nvml.nvmlDeviceGetComputeInstanceId), patch('ray._private.thirdparty.pynvml.nvmlDeviceGetGpuInstanceId', mock_nvml.nvmlDeviceGetGpuInstanceId):
        yield