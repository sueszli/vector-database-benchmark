import unittest
import legacy_test.test_collective_api_base as test_base

class TestCollectiveReduceScatterAPI(test_base.TestDistBase):

    def _setup_config(self):
        if False:
            i = 10
            return i + 15
        pass

    def test_reduce_scatter_nccl_with_comm_context(self):
        if False:
            print('Hello World!')
        dtypes_to_test = ['float16', 'float32', 'float64', 'int32', 'int64', 'int8', 'uint8', 'bool']
        if self._nccl_version >= 21000:
            dtypes_to_test.append('bfloat16')
        for dtype in dtypes_to_test:
            self.check_with_place('collective_reduce_scatter_api.py', 'reduce_scatter', 'nccl', dtype=dtype, need_envs={'USE_COMM_CONTEXT': '1'})

    def test_reduce_scatter_nccl_with_new_comm(self):
        if False:
            print('Hello World!')
        dtypes_to_test = ['float16', 'float32', 'float64', 'int32', 'int64']
        if self._nccl_version >= 21000:
            dtypes_to_test.append('bfloat16')
        for dtype in dtypes_to_test:
            self.check_with_place('collective_reduce_scatter_api.py', 'reduce_scatter', 'nccl', dtype=dtype, need_envs={'FLAGS_dynamic_static_unified_comm': 'true'})

    def test_reduce_scatter_nccl_dygraph(self):
        if False:
            return 10
        dtypes_to_test = ['float16', 'float32', 'float64', 'int32', 'int64', 'int8', 'uint8', 'bool']
        if self._nccl_version >= 21000:
            dtypes_to_test.append('bfloat16')
        for dtype in dtypes_to_test:
            self.check_with_place('collective_reduce_scatter_api_dygraph.py', 'reduce_scatter', 'nccl', static_mode='0', dtype=dtype)
if __name__ == '__main__':
    unittest.main()