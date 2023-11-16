import os
import unittest
from legacy_test.test_dist_base import TestDistBase
flag_name = os.path.splitext(__file__)[0]

class TestDistSeResnetNCCL2DGC(TestDistBase):

    def _setup_config(self):
        if False:
            i = 10
            return i + 15
        self._sync_mode = True
        self._use_reduce = False
        self._use_reader_alloc = False
        self._nccl2_mode = True
        self._use_dgc = True

    @unittest.skip(reason='Skip unstable ci')
    def test_dist_train(self):
        if False:
            while True:
                i = 10
        from paddle import base
        if base.core.is_compiled_with_cuda():
            self.check_with_place(os.path.abspath('../../dist_se_resnext.py'), delta=30, check_error_log=True, log_name=flag_name)
if __name__ == '__main__':
    unittest.main()