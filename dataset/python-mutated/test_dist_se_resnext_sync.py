import os
import unittest
from test_dist_base import TestDistBase
flag_name = os.path.splitext(__file__)[0]

class TestDistSeResneXt2x2(TestDistBase):

    def _setup_config(self):
        if False:
            while True:
                i = 10
        self._sync_mode = True
        self._use_reader_alloc = False

    @unittest.skip(reason='Skip unstable ci')
    def test_dist_train(self):
        if False:
            while True:
                i = 10
        self.check_with_place('dist_se_resnext.py', delta=1e-07, check_error_log=True, log_name=flag_name)
if __name__ == '__main__':
    unittest.main()