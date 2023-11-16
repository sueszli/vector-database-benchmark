import unittest
from test_dist_base import TestDistBase
import paddle
paddle.enable_static()

class TestDistMnistLocalSGDFleetApi(TestDistBase):

    def _setup_config(self):
        if False:
            return 10
        self._sync_mode = True
        self._use_reduce = False
        self._use_reader_alloc = False
        self._nccl2_mode = True
        self._use_fleet_api = True
        self._use_local_sgd = True

    def test_dist_train(self):
        if False:
            i = 10
            return i + 15
        from paddle import base
        if base.core.is_compiled_with_cuda():
            self.check_with_place('dist_mnist.py', delta=1e-05)

class TestDistMnistGradAllReduceFleetApi(TestDistBase):

    def _setup_config(self):
        if False:
            print('Hello World!')
        self._sync_mode = True
        self._use_reduce = False
        self._use_reader_alloc = False
        self._nccl2_mode = True
        self._use_fleet_api = True
        self._ut4grad_allreduce = True

    def test_dist_train(self):
        if False:
            i = 10
            return i + 15
        from paddle import base
        if base.core.is_compiled_with_cuda():
            self.check_with_place('dist_mnist.py', delta=1e-05)
if __name__ == '__main__':
    unittest.main()