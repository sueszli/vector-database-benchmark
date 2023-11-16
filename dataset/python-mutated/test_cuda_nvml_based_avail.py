import os
import sys
import multiprocessing
import torch
import unittest
from unittest.mock import patch
with patch.dict(os.environ, {'PYTORCH_NVML_BASED_CUDA_CHECK': '1'}):
    from torch.testing._internal.common_utils import parametrize, instantiate_parametrized_tests, run_tests, TestCase, IS_WINDOWS, IS_JETSON, NoTest
    torch.testing._internal.common_utils.remove_device_and_dtype_suffixes = lambda x: x
    TEST_CUDA = torch.cuda.is_available()
    if not TEST_CUDA:
        print('CUDA not available, skipping tests', file=sys.stderr)
        TestCase = NoTest

class TestExtendedCUDAIsAvail(TestCase):
    SUBPROCESS_REMINDER_MSG = '\n REMINDER: Tests defined in test_cuda_nvml_based_avail.py must be run in a process where there CUDA Driver API has not been initialized. Before further debugging, ensure you are either using run_test.py or have added --subprocess to run each test in a different subprocess.'

    def setUp(self):
        if False:
            return 10
        super().setUp()
        torch.cuda.device_count.cache_clear()

    @staticmethod
    def in_bad_fork_test() -> bool:
        if False:
            while True:
                i = 10
        _ = torch.cuda.is_available()
        return torch.cuda._is_in_bad_fork()

    @unittest.skipIf(IS_WINDOWS, 'Needs fork')
    @parametrize('nvml_avail', [True, False])
    @parametrize('avoid_init', ['1', '0', None])
    def test_cuda_is_available(self, avoid_init, nvml_avail):
        if False:
            for i in range(10):
                print('nop')
        if IS_JETSON and nvml_avail and (avoid_init == '1'):
            self.skipTest('Not working for Jetson')
        patch_env = {'PYTORCH_NVML_BASED_CUDA_CHECK': avoid_init} if avoid_init else {}
        with patch.dict(os.environ, **patch_env):
            if nvml_avail:
                _ = torch.cuda.is_available()
            else:
                with patch.object(torch.cuda, '_device_count_nvml', return_value=-1):
                    _ = torch.cuda.is_available()
            with multiprocessing.get_context('fork').Pool(1) as pool:
                in_bad_fork = pool.apply(TestExtendedCUDAIsAvail.in_bad_fork_test)
            if os.getenv('PYTORCH_NVML_BASED_CUDA_CHECK') == '1' and nvml_avail:
                self.assertFalse(in_bad_fork, TestExtendedCUDAIsAvail.SUBPROCESS_REMINDER_MSG)
            else:
                assert in_bad_fork

class TestVisibleDeviceParses(TestCase):

    def test_env_var_parsing(self):
        if False:
            for i in range(10):
                print('nop')

        def _parse_visible_devices(val):
            if False:
                return 10
            from torch.cuda import _parse_visible_devices as _pvd
            with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': val}, clear=True):
                return _pvd()
        self.assertEqual(_parse_visible_devices('1gpu2,2ampere'), [1, 2])
        self.assertEqual(_parse_visible_devices('0, 1, 2, -1, 3'), [0, 1, 2])
        self.assertEqual(_parse_visible_devices('0, 1, 2, 1'), [])
        self.assertEqual(_parse_visible_devices('2, +3, -0, 5'), [2, 3, 0, 5])
        self.assertEqual(_parse_visible_devices('one,two,3,4'), [])
        self.assertEqual(_parse_visible_devices('4,3,two,one'), [4, 3])
        self.assertEqual(_parse_visible_devices('GPU-9e8d35e3'), ['GPU-9e8d35e3'])
        self.assertEqual(_parse_visible_devices('GPU-123, 2'), ['GPU-123'])
        self.assertEqual(_parse_visible_devices('MIG-89c850dc'), ['MIG-89c850dc'])

    def test_partial_uuid_resolver(self):
        if False:
            while True:
                i = 10
        from torch.cuda import _transform_uuid_to_ordinals
        uuids = ['GPU-9942190a-aa31-4ff1-4aa9-c388d80f85f1', 'GPU-9e8d35e3-a134-0fdd-0e01-23811fdbd293', 'GPU-e429a63e-c61c-4795-b757-5132caeb8e70', 'GPU-eee1dfbc-0a0f-6ad8-5ff6-dc942a8b9d98', 'GPU-bbcd6503-5150-4e92-c266-97cc4390d04e', 'GPU-472ea263-58d7-410d-cc82-f7fdece5bd28', 'GPU-e56257c4-947f-6a5b-7ec9-0f45567ccf4e', 'GPU-1c20e77d-1c1a-d9ed-fe37-18b8466a78ad']
        self.assertEqual(_transform_uuid_to_ordinals(['GPU-9e8d35e3'], uuids), [1])
        self.assertEqual(_transform_uuid_to_ordinals(['GPU-e4', 'GPU-9e8d35e3'], uuids), [2, 1])
        self.assertEqual(_transform_uuid_to_ordinals('GPU-9e8d35e3,GPU-1,GPU-47'.split(','), uuids), [1, 7, 5])
        self.assertEqual(_transform_uuid_to_ordinals(['GPU-123', 'GPU-9e8d35e3'], uuids), [])
        self.assertEqual(_transform_uuid_to_ordinals(['GPU-9e8d35e3', 'GPU-123', 'GPU-47'], uuids), [1])
        self.assertEqual(_transform_uuid_to_ordinals(['GPU-9e8d35e3', 'GPU-e', 'GPU-47'], uuids), [1])
        self.assertEqual(_transform_uuid_to_ordinals(['GPU-9e8d35e3', 'GPU-47', 'GPU-9e8'], uuids), [])

    def test_ordinal_parse_visible_devices(self):
        if False:
            return 10

        def _device_count_nvml(val):
            if False:
                i = 10
                return i + 15
            from torch.cuda import _device_count_nvml as _dc
            with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': val}, clear=True):
                return _dc()
        with patch.object(torch.cuda, '_raw_device_count_nvml', return_value=2):
            self.assertEqual(_device_count_nvml('1, 0'), 2)
            self.assertEqual(_device_count_nvml('1, 5, 0'), 1)
instantiate_parametrized_tests(TestExtendedCUDAIsAvail)
if __name__ == '__main__':
    run_tests()