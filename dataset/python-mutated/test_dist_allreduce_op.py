import unittest
from test_dist_base import TestDistBase
import paddle
paddle.enable_static()

class TestDistMnistNCCL2(TestDistBase):

    def _setup_config(self):
        if False:
            while True:
                i = 10
        self._sync_mode = True
        self._use_reduce = False
        self._use_reader_alloc = False
        self._nccl2_mode = True
        self._nccl2_reduce_layer = True

    def test_dist_train(self):
        if False:
            return 10
        from paddle import base
        if base.core.is_compiled_with_cuda():
            self.check_with_place('dist_allreduce_op.py', delta=1e-05, check_error_log=True)
if __name__ == '__main__':
    unittest.main()