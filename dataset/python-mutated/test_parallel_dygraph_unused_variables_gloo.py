import os
import unittest
from legacy_test.test_dist_base import TestDistBase
flag_name = os.path.splitext(__file__)[0]

class TestParallelDygraphUnusedVar_GLOO(TestDistBase):

    def _setup_config(self):
        if False:
            while True:
                i = 10
        self._sync_mode = False
        self._gloo_mode = True
        self._dygraph = True

    def test_net(self):
        if False:
            while True:
                i = 10
        self.check_with_place('parallel_dygraph_unused_variables.py', delta=1e-05, check_error_log=True, log_name=flag_name)

class TestParallelDygraphNoVar_GLOO(TestDistBase):

    def _setup_config(self):
        if False:
            while True:
                i = 10
        self._sync_mode = False
        self._gloo_mode = True
        self._dygraph = True

    def test_net(self):
        if False:
            print('Hello World!')
        self.check_with_place('parallel_dygraph_none_var.py', delta=1e-05, check_error_log=True, log_name=flag_name)

class TestParallelDygraphSharedUnusedVariables_GLOO(TestDistBase):

    def _setup_config(self):
        if False:
            i = 10
            return i + 15
        self._sync_mode = False
        self._gloo_mode = True
        self._dygraph = True

    def test_mnist(self):
        if False:
            while True:
                i = 10
        self.check_with_place('parallel_dygraph_shared_unused_var.py', delta=1e-05, check_error_log=True, log_name=flag_name)
if __name__ == '__main__':
    unittest.main()