import os
import unittest
from legacy_test.test_dist_base import TestDistBase
import paddle
paddle.enable_static()
flag_name = os.path.splitext(__file__)[0]

class TestStaticModelParallel(TestDistBase):

    def _setup_config(self):
        if False:
            return 10
        self._sync_mode = True
        self._use_reduce = False
        self._use_reader_alloc = False
        self._nccl_comm_num = 1
        self._pipeline_mode = True

    def test_dist_static_model_parallel(self):
        if False:
            i = 10
            return i + 15
        from paddle import base
        if base.core.is_compiled_with_cuda():
            self.check_with_place('static_model_parallel_by_row.py', delta=1e-05, check_error_log=True, log_name=flag_name)

    def test_dist_static_model_parallel2(self):
        if False:
            while True:
                i = 10
        from paddle import base
        if base.core.is_compiled_with_cuda():
            self.check_with_place('static_model_parallel_by_col.py', delta=1e-05, check_error_log=True, log_name=flag_name)

    def test_dist_static_model_parallel3(self):
        if False:
            for i in range(10):
                print('nop')
        from paddle import base
        if base.core.is_compiled_with_cuda():
            self.check_with_place('static_model_parallel_embedding.py', delta=1e-05, check_error_log=True, log_name=flag_name)
if __name__ == '__main__':
    unittest.main()