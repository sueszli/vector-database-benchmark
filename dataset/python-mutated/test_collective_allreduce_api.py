import unittest
from test_collective_api_base import TestDistBase
import paddle
paddle.enable_static()
import paddle.distributed as dist

class TestCollectiveAllreduceAPI(TestDistBase):

    def _setup_config(self):
        if False:
            while True:
                i = 10
        pass

    def test_allreduce_nccl(self):
        if False:
            while True:
                i = 10
        if paddle.base.core.is_compiled_with_cuda():
            self.check_with_place('collective_allreduce_api.py', 'allreduce', 'nccl')

    def test_allreduce_nccl_with_comm_context(self):
        if False:
            return 10
        dtypes_to_test = ['float16', 'float32', 'float64', 'int32', 'int64', 'int8', 'uint8', 'bool']
        red_types_to_test = [dist.ReduceOp.SUM]
        if self._nccl_version >= 21000:
            dtypes_to_test.append('bfloat16')
        for dtype in dtypes_to_test:
            for red_type in red_types_to_test:
                self.check_with_place('collective_allreduce_api.py', 'allreduce', 'nccl', dtype=dtype, reduce_type=red_type, need_envs={'USE_COMM_CONTEXT': '1'})

    def test_allreduce_nccl_with_new_comm(self):
        if False:
            i = 10
            return i + 15
        dtypes_to_test = ['float16', 'float32', 'float64', 'int32', 'int64']
        red_types_to_test = [dist.ReduceOp.SUM]
        if self._nccl_version >= 21000:
            dtypes_to_test.append('bfloat16')
        for dtype in dtypes_to_test:
            for red_type in red_types_to_test:
                self.check_with_place('collective_allreduce_api.py', 'allreduce', 'nccl', dtype=dtype, reduce_type=red_type, need_envs={'FLAGS_dynamic_static_unified_comm': '1'})

    def test_allreduce_bkcl(self):
        if False:
            while True:
                i = 10
        if paddle.base.core.is_compiled_with_xpu():
            self.check_with_place('collective_allreduce_api.py', 'allreduce', 'bkcl')

    def test_allreduce_gloo(self):
        if False:
            print('Hello World!')
        self.check_with_place('collective_allreduce_api.py', 'allreduce', 'gloo', '2')

    def test_allreduce_gloo_with_comm_context(self):
        if False:
            return 10
        dtypes_to_test = ['float16', 'float32', 'float64', 'int32', 'int64', 'int8', 'uint8', 'bool']
        red_types_to_test = [dist.ReduceOp.SUM]
        for dtype in dtypes_to_test:
            for red_type in red_types_to_test:
                self.check_with_place('collective_allreduce_api.py', 'allreduce', 'gloo', '2', dtype=dtype, reduce_type=red_type, need_envs={'USE_COMM_CONTEXT': '1'})

    def test_allreduce_nccl_dygraph(self):
        if False:
            print('Hello World!')
        dtypes_to_test = ['float16', 'float32', 'float64', 'int32', 'int64', 'int8', 'uint8', 'bool']
        if self._nccl_version >= 21000:
            dtypes_to_test.append('bfloat16')
        for dtype in dtypes_to_test:
            self.check_with_place('collective_allreduce_api_dygraph.py', 'allreduce', 'nccl', static_mode='0', dtype=dtype)

    def test_allreduce_gloo_dygraph(self):
        if False:
            for i in range(10):
                print('nop')
        dtypes_to_test = ['float16', 'float32', 'float64', 'int32', 'int64', 'int8', 'uint8', 'bool', 'bfloat16']
        for dtype in dtypes_to_test:
            self.check_with_place('collective_allreduce_api_dygraph.py', 'allreduce', 'gloo', '2', static_mode='0', dtype=dtype)
if __name__ == '__main__':
    unittest.main()