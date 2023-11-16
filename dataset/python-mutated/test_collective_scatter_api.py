import unittest
from test_collective_api_base import TestDistBase
import paddle
paddle.enable_static()

class TestCollectiveScatterAPI(TestDistBase):

    def _setup_config(self):
        if False:
            print('Hello World!')
        pass

    def test_scatter_gloo(self):
        if False:
            return 10
        self.check_with_place('collective_scatter_api.py', 'scatter', 'gloo', '4')

    def test_scatter_nccl(self):
        if False:
            while True:
                i = 10
        self.check_with_place('collective_scatter_api.py', 'scatter', 'nccl')

    def test_scatter_nccl_with_new_comm(self):
        if False:
            while True:
                i = 10
        dtypes_to_test = ['float16', 'float32', 'float64', 'int32', 'int64']
        for dtype in dtypes_to_test:
            self.check_with_place('collective_scatter_api.py', 'scatter', 'nccl', dtype=dtype, need_envs={'FLAGS_dynamic_static_unified_comm': 'true'})

    def test_scatter_nccl_dygraph(self):
        if False:
            return 10
        dtypes_to_test = ['float16', 'float32', 'float64', 'int32', 'int64', 'int8', 'uint8', 'bool']
        if self._nccl_version >= 21000:
            dtypes_to_test.append('bfloat16')
        for dtype in dtypes_to_test:
            self.check_with_place('collective_scatter_api_dygraph.py', 'scatter', 'nccl', static_mode='0', dtype=dtype)

    def test_scatter_gloo_dygraph(self):
        if False:
            for i in range(10):
                print('nop')
        dtypes_to_test = ['float16', 'float32', 'float64', 'int32', 'int64', 'int8', 'uint8', 'bool', 'bfloat16']
        for dtype in dtypes_to_test:
            self.check_with_place('collective_scatter_api_dygraph.py', 'scatter', 'gloo', '4', static_mode='0', dtype=dtype)
if __name__ == '__main__':
    unittest.main()