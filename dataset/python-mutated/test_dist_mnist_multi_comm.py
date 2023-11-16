import os
import unittest
from test_dist_base import TestDistBase
import paddle
paddle.enable_static()
flag_name = os.path.splitext(__file__)[0]

class TestDistMnistNCCL2MultiNCCLComm(TestDistBase):

    def _setup_config(self):
        if False:
            i = 10
            return i + 15
        self._sync_mode = True
        self._use_reduce = False
        self._use_reader_alloc = False
        self._nccl2_mode = True
        self._nccl_comm_num = 3

    def test_dist_train(self):
        if False:
            return 10
        from paddle import base
        if base.core.is_compiled_with_cuda():
            self.check_with_place('dist_mnist.py', delta=1e-05, check_error_log=True, log_name=flag_name)
if __name__ == '__main__':
    unittest.main()