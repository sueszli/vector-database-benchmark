import os
import unittest
from test_dist_base import TestDistBase
from paddle import base
flag_name = os.path.splitext(__file__)[0]

class TestParallelDygraphTransformer_GLOO(TestDistBase):

    def _setup_config(self):
        if False:
            return 10
        self._sync_mode = False
        self._gloo_mode = True
        self._dygraph = True

    def test_transformer(self):
        if False:
            i = 10
            return i + 15
        self.check_with_place('parallel_dygraph_transformer.py', delta=1e-05, check_error_log=True, log_name=flag_name)

class TestParallelDygraphTransformerAccGrad_GLOO(TestDistBase):

    def _setup_config(self):
        if False:
            print('Hello World!')
        self._sync_mode = False
        self._gloo_mode = True
        self._dygraph = True
        self._accumulate_gradient = True
        self._find_unused_parameters = False

    def test_transformer(self):
        if False:
            i = 10
            return i + 15
        if base.core.is_compiled_with_cuda():
            self.check_with_place('parallel_dygraph_transformer.py', delta=1e-05, check_error_log=True, log_name=flag_name)
if __name__ == '__main__':
    unittest.main()