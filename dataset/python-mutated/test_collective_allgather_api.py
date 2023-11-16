import unittest
from test_collective_api_base import TestDistBase
import paddle
paddle.enable_static()

class TestCollectiveAllgatherAPI(TestDistBase):

    def _setup_config(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def test_allgather_nccl(self):
        if False:
            for i in range(10):
                print('nop')
        dtypes_to_test = ['float16', 'float32', 'float64', 'int32', 'int64', 'int8', 'uint8', 'bool']
        for dtype in dtypes_to_test:
            self.check_with_place('collective_allgather_api.py', 'allgather', 'nccl', dtype=dtype)

    def test_allgather_nccl_with_comm_context(self):
        if False:
            while True:
                i = 10
        dtypes_to_test = ['float16', 'float32', 'float64', 'int32', 'int64', 'int8', 'uint8', 'bool']
        if self._nccl_version >= 21000:
            dtypes_to_test.append('bfloat16')
        for dtype in dtypes_to_test:
            self.check_with_place('collective_allgather_api.py', 'allgather', 'nccl', dtype=dtype, need_envs={'USE_COMM_CONTEXT': '1'})

    def test_allgather_nccl_with_new_comm(self):
        if False:
            print('Hello World!')
        dtypes_to_test = ['float16', 'float32', 'float64', 'int32', 'int64', 'int8', 'uint8', 'bool']
        for dtype in dtypes_to_test:
            self.check_with_place('collective_allgather_api.py', 'allgather', 'nccl', dtype=dtype, need_envs={'FLAGS_dynamic_static_unified_comm': '1'})

    def test_allgather_gloo(self):
        if False:
            while True:
                i = 10
        dtypes_to_test = ['float16', 'float32', 'float64', 'int32', 'int64', 'int8', 'uint8', 'bool']
        for dtype in dtypes_to_test:
            self.check_with_place('collective_allgather_api.py', 'allgather', 'gloo', '3', dtype=dtype)

    def test_allgather_gloo_with_comm_context(self):
        if False:
            i = 10
            return i + 15
        dtypes_to_test = ['float16', 'float32', 'float64', 'int32', 'int64', 'int8', 'uint8', 'bool']
        for dtype in dtypes_to_test:
            self.check_with_place('collective_allgather_api.py', 'allgather', 'gloo', '3', dtype=dtype, need_envs={'USE_COMM_CONTEXT': '1'})

    def test_allgather_nccl_dygraph(self):
        if False:
            while True:
                i = 10
        dtypes_to_test = ['float16', 'float32', 'float64', 'int32', 'int64', 'int8', 'uint8', 'bool']
        if self._nccl_version >= 21000:
            dtypes_to_test.append('bfloat16')
        for dtype in dtypes_to_test:
            self.check_with_place('collective_allgather_api_dygraph.py', 'allgather', 'nccl', static_mode='0', dtype=dtype)

    def test_allgather_gloo_dygraph(self):
        if False:
            while True:
                i = 10
        dtypes_to_test = ['float16', 'float32', 'float64', 'int32', 'int64', 'int8', 'uint8', 'bool', 'bfloat16']
        for dtype in dtypes_to_test:
            self.check_with_place('collective_allgather_api_dygraph.py', 'allgather', 'gloo', '3', static_mode='0', dtype=dtype)
if __name__ == '__main__':
    unittest.main()