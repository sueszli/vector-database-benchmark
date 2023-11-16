import os
import unittest
from test_dist_base import TestDistBase
flag_name = os.path.splitext(__file__)[0]

class TestDistW2V2x2(TestDistBase):

    def _setup_config(self):
        if False:
            for i in range(10):
                print('nop')
        self._sync_mode = True
        self._enforce_place = 'CPU'

    def test_dist_train(self):
        if False:
            i = 10
            return i + 15
        self.check_with_place('dist_word2vec.py', delta=0.0001, check_error_log=True, log_name=flag_name)

class TestDistW2V2x2WithMemOpt(TestDistBase):

    def _setup_config(self):
        if False:
            print('Hello World!')
        self._sync_mode = True
        self._mem_opt = True
        self._enforce_place = 'CPU'

    def test_dist_train(self):
        if False:
            while True:
                i = 10
        self.check_with_place('dist_word2vec.py', delta=0.0001, check_error_log=True, log_name=flag_name)

class TestDistW2V2x2Async(TestDistBase):

    def _setup_config(self):
        if False:
            print('Hello World!')
        self._sync_mode = False
        self._enforce_place = 'CPU'

    def test_dist_train(self):
        if False:
            return 10
        self.check_with_place('dist_word2vec.py', delta=100, check_error_log=True, log_name=flag_name)
if __name__ == '__main__':
    unittest.main()