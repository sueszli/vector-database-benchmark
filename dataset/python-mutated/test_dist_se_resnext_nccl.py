import os
import unittest
from test_dist_base import TestDistBase
import paddle
paddle.enable_static()
flag_name = os.path.splitext(__file__)[0]

class TestDistSeResneXtNCCL(TestDistBase):

    def _setup_config(self):
        if False:
            i = 10
            return i + 15
        self._sync_mode = True
        self._use_reader_alloc = False
        self._nccl2_mode = True

    def test_dist_train(self):
        if False:
            i = 10
            return i + 15
        from paddle import base
        if base.core.is_compiled_with_cuda():
            self.check_with_place('dist_se_resnext.py', delta=1e-05, check_error_log=True, log_name=flag_name)

class TestDistSeResneXtNCCLMP(TestDistBase):

    def _setup_config(self):
        if False:
            while True:
                i = 10
        self._sync_mode = True
        self._use_reader_alloc = False
        self._nccl2_mode = True
        self._mp_mode = True

    def test_dist_train(self):
        if False:
            return 10
        from paddle import base
        if base.core.is_compiled_with_cuda():
            self.check_with_place('dist_se_resnext.py', delta=1e-05, check_error_log=True, need_envs={'NCCL_P2P_DISABLE': '1'}, log_name=flag_name)
if __name__ == '__main__':
    unittest.main()