import os
import unittest
from test_dist_base import TestDistBase
import paddle
paddle.enable_static()
flag_name = os.path.splitext(__file__)[0]

class TestFleetMetaOptimizerAllReduceFusePrecision(TestDistBase):

    def _setup_config(self):
        if False:
            for i in range(10):
                print('nop')
        self._sync_mode = True
        self._use_reduce = False
        self._use_reader_alloc = False
        self._nccl2_mode = True
        self._nccl2_reduce_layer = True
        self._use_fleet_api = True
        self._use_fleet_api_20 = True

    def test_dist_train(self):
        if False:
            for i in range(10):
                print('nop')
        from paddle import base
        if base.core.is_compiled_with_cuda():
            self.check_with_place('dist_fleet_raw_program_optimizer_fuse_allreduce.py', delta=1e-05, check_error_log=True, log_name=flag_name)
if __name__ == '__main__':
    unittest.main()