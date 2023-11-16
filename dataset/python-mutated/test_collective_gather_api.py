import unittest
from legacy_test.test_collective_api_base import TestDistBase
import paddle
paddle.enable_static()

class TestCollectiveGatherAPI(TestDistBase):

    def _setup_config(self):
        if False:
            while True:
                i = 10
        pass

    def test_gather_nccl_dygraph(self):
        if False:
            i = 10
            return i + 15
        dtypes_to_test = ['float16', 'float32', 'float64', 'int32', 'int64', 'int8', 'uint8', 'bool']
        if self._nccl_version >= 21000:
            dtypes_to_test.append('bfloat16')
        for dtype in dtypes_to_test:
            self.check_with_place('collective_gather_api_dygraph.py', 'gather', 'nccl', static_mode='0', dtype=dtype)
if __name__ == '__main__':
    unittest.main()