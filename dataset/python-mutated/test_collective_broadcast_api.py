import unittest
from legacy_test.test_collective_api_base import TestDistBase
import paddle
paddle.enable_static()

class TestCollectiveBroadcastAPI(TestDistBase):

    def _setup_config(self):
        if False:
            print('Hello World!')
        pass

    def test_broadcast_nccl(self):
        if False:
            i = 10
            return i + 15
        self.check_with_place('collective_broadcast_api.py', 'broadcast', 'nccl')

    def test_broadcast_nccl_with_comm_context(self):
        if False:
            i = 10
            return i + 15
        dtypes_to_test = ['float16', 'float32', 'float64', 'int32', 'int64', 'int8', 'uint8', 'bool']
        if self._nccl_version >= 21000:
            dtypes_to_test.append('bfloat16')
        for dtype in dtypes_to_test:
            self.check_with_place('collective_broadcast_api.py', 'broadcast', 'nccl', dtype=dtype, need_envs={'USE_COMM_CONTEXT': '1'})

    def test_broadcast_gloo(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_with_place('collective_broadcast_api.py', 'broadcast', 'gloo')

    def test_broadcast_gloo_with_comm_context(self):
        if False:
            while True:
                i = 10
        dtypes_to_test = ['float16', 'float32', 'float64', 'int32', 'int64', 'int8', 'uint8', 'bool']
        for dtype in dtypes_to_test:
            self.check_with_place('collective_broadcast_api.py', 'broadcast', 'gloo', dtype=dtype, need_envs={'USE_COMM_CONTEXT': '1'})

    def test_broadcast_nccl_dygraph(self):
        if False:
            print('Hello World!')
        dtypes_to_test = ['float16', 'float32', 'float64', 'int32', 'int64', 'int8', 'uint8', 'bool']
        if self._nccl_version >= 21000:
            dtypes_to_test.append('bfloat16')
        for dtype in dtypes_to_test:
            self.check_with_place('collective_broadcast_api_dygraph.py', 'broadcast', 'nccl', static_mode='0', dtype=dtype)

    def test_broadcast_gloo_dygraph(self):
        if False:
            i = 10
            return i + 15
        dtypes_to_test = ['float16', 'float32', 'float64', 'int32', 'int64', 'int8', 'uint8', 'bool', 'bfloat16']
        for dtype in dtypes_to_test:
            self.check_with_place('collective_broadcast_api_dygraph.py', 'broadcast', 'gloo', '0', static_mode='0', dtype=dtype)
if __name__ == '__main__':
    unittest.main()