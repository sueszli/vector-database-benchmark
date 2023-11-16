import os
import unittest
from legacy_test.test_dist_base import TestDistBase
from paddle import base
flag_name = os.path.splitext(__file__)[0]

class TestParallelDygraphMnist(TestDistBase):

    def _setup_config(self):
        if False:
            print('Hello World!')
        self._sync_mode = False
        self._nccl2_mode = True
        self._dygraph = True

    def test_mnist(self):
        if False:
            return 10
        if base.core.is_compiled_with_cuda():
            self.check_with_place('parallel_dygraph_sync_batch_norm.py', delta=1e-05, check_error_log=True, log_name=flag_name)
if __name__ == '__main__':
    unittest.main()