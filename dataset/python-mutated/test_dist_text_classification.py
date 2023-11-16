import os
import unittest
from test_dist_base import TestDistBase
flag_name = os.path.splitext(__file__)[0]

class TestDistTextClassification2x2(TestDistBase):

    def _setup_config(self):
        if False:
            while True:
                i = 10
        self._sync_mode = True
        self._enforce_place = 'CPU'

    def test_text_classification(self):
        if False:
            return 10
        self.check_with_place('dist_text_classification.py', delta=1e-06, check_error_log=True, log_name=flag_name)

class TestDistTextClassification2x2Async(TestDistBase):

    def _setup_config(self):
        if False:
            print('Hello World!')
        self._sync_mode = False
        self._enforce_place = 'CPU'

    def test_se_resnext(self):
        if False:
            while True:
                i = 10
        self.check_with_place('dist_text_classification.py', delta=100, check_error_log=True, log_name=flag_name)
if __name__ == '__main__':
    unittest.main()