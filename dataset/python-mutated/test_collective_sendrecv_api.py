import unittest
from legacy_test.test_collective_api_base import TestDistBase
import paddle
paddle.enable_static()

class TestCollectiveSendRecvAPI(TestDistBase):

    def _setup_config(self):
        if False:
            while True:
                i = 10
        pass

    def test_sendrecv_nccl_with_comm_context(self):
        if False:
            i = 10
            return i + 15
        dtypes_to_test = ['float16', 'float32', 'float64', 'int32', 'int64', 'int8', 'uint8', 'bool']
        if self._nccl_version >= 21000:
            dtypes_to_test.append('bfloat16')
        for dtype in dtypes_to_test:
            if paddle.base.core.is_compiled_with_cuda():
                self.check_with_place('collective_sendrecv_api.py', 'sendrecv', 'nccl', dtype=dtype, need_envs={'USE_COMM_CONTEXT': '1'})

    def test_sendrecv_nccl_dygraph(self):
        if False:
            while True:
                i = 10
        dtypes_to_test = ['float16', 'float32', 'float64', 'int32', 'int64', 'int8', 'uint8', 'bool']
        if self._nccl_version >= 21000:
            dtypes_to_test.append('bfloat16')
        for dtype in dtypes_to_test:
            self.check_with_place('collective_sendrecv_api_dygraph.py', 'sendrecv', 'nccl', static_mode='0', dtype=dtype)
if __name__ == '__main__':
    unittest.main()