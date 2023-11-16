import unittest
from test_dist_base import TestDistBase

class TestDistMnist2x2FP16AllReduce(TestDistBase):

    def _setup_config(self):
        if False:
            for i in range(10):
                print('nop')
        self._sync_mode = True
        self._use_reduce = False
        self._nccl2_mode = True
        self._nccl2_reduce_layer = True

    def test_dist_train(self):
        if False:
            for i in range(10):
                print('nop')
        from paddle import base
        if base.core.is_compiled_with_cuda():
            self.check_with_place('dist_mnist_fp16_allreduce.py', delta=1e-05, check_error_log=True)
if __name__ == '__main__':
    unittest.main()