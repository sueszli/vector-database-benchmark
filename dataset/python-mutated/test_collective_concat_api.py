import unittest
from legacy_test.test_collective_api_base import TestDistBase
import paddle
paddle.enable_static()

class TestCollectiveConcatAPI(TestDistBase):

    def _setup_config(self):
        if False:
            i = 10
            return i + 15
        pass

    def test_concat_with_comm_context(self):
        if False:
            print('Hello World!')
        dtypes_to_test = ['float16', 'float32', 'float64', 'int32', 'int64', 'int8', 'uint8', 'bool']
        if self._nccl_version >= 21000:
            dtypes_to_test.append('bfloat16')
        for dtype in dtypes_to_test:
            self.check_with_place('collective_concat_api.py', 'dist_concat', 'nccl', dtype=dtype, need_envs={'USE_COMM_CONTEXT': '1'})

    def test_concat_with_new_comm(self):
        if False:
            print('Hello World!')
        dtypes_to_test = ['float16', 'float32', 'float64', 'int32', 'int64']
        for dtype in dtypes_to_test:
            self.check_with_place('collective_concat_api.py', 'dist_concat', 'nccl', dtype=dtype, need_envs={'FLAGS_dynamic_static_unified_comm': '1'})
if __name__ == '__main__':
    unittest.main()