import os
import unittest
from test_dist_base import TestDistBase
flag_name = os.path.splitext(__file__)[0]

class TestDistMnist2x2(TestDistBase):

    def _setup_config(self):
        if False:
            print('Hello World!')
        self._sync_mode = True
        self._use_reduce = False

    def test_dist_train(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_with_place('dist_mnist.py', delta=1e-05, check_error_log=True, log_name=flag_name)

class TestDistMnist2x2WithMemopt(TestDistBase):

    def _setup_config(self):
        if False:
            for i in range(10):
                print('nop')
        self._sync_mode = True
        self._mem_opt = True

    def test_dist_train(self):
        if False:
            return 10
        self.check_with_place('dist_mnist.py', delta=1e-05, check_error_log=True, log_name=flag_name)

class TestDistMnistAsync(TestDistBase):

    def _setup_config(self):
        if False:
            print('Hello World!')
        self._sync_mode = False
        self._use_reduce = False

    def test_dist_train(self):
        if False:
            while True:
                i = 10
        self.check_with_place('dist_mnist.py', delta=200, check_error_log=True, log_name=flag_name)

class TestDistMnistDcAsgd(TestDistBase):

    def _setup_config(self):
        if False:
            while True:
                i = 10
        self._sync_mode = False
        self._dc_asgd = True

    def test_se_resnext(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_with_place('dist_mnist.py', delta=200, check_error_log=True, log_name=flag_name)
if __name__ == '__main__':
    unittest.main()