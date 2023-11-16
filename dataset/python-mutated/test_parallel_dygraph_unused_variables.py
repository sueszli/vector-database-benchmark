import os
import sys
import unittest
sys.path.append('../../legacy_test')
from parallel_dygraph_unused_variables import TestSparseEmbeddingUnusedVars
from spawn_runner_base import TestDistSpawnRunner
from test_dist_base import TestDistBase
from paddle import base
flag_name = os.path.splitext(__file__)[0]

class TestParallelDygraphUnusedVar(TestDistBase):

    def _setup_config(self):
        if False:
            while True:
                i = 10
        self._sync_mode = False
        self._nccl2_mode = True
        self._dygraph = True

    def test_net(self):
        if False:
            while True:
                i = 10
        if base.core.is_compiled_with_cuda():
            self.check_with_place(os.path.abspath('../../legacy_test/parallel_dygraph_unused_variables.py'), delta=1e-05, check_error_log=True, log_name=flag_name)

class TestFleetDygraphUnusedVar(TestParallelDygraphUnusedVar):

    def _setup_config(self):
        if False:
            return 10
        self._sync_mode = False
        self._nccl2_mode = True
        self._dygraph = True
        self._use_fleet_api = True

class TestSparseEmbeddingUnusedVarsSpawn(TestDistSpawnRunner):

    def test_mnist_with_spawn(self):
        if False:
            print('Hello World!')
        if base.core.is_compiled_with_cuda():
            self.check_dist_result_with_spawn(test_class=TestSparseEmbeddingUnusedVars, delta=1e-05)

class TestParallelDygraphNoVar(TestDistBase):

    def _setup_config(self):
        if False:
            while True:
                i = 10
        self._sync_mode = False
        self._nccl2_mode = True
        self._dygraph = True

    def test_net(self):
        if False:
            while True:
                i = 10
        if base.core.is_compiled_with_cuda():
            self.check_with_place(os.path.abspath('../../legacy_test/parallel_dygraph_none_var.py'), delta=1e-05, check_error_log=True, log_name=flag_name)

class TestParallelDygraphSharedUnusedVariables(TestDistBase):

    def _setup_config(self):
        if False:
            for i in range(10):
                print('nop')
        self._sync_mode = False
        self._nccl2_mode = True
        self._dygraph = True

    def test_mnist(self):
        if False:
            while True:
                i = 10
        if base.core.is_compiled_with_cuda():
            self.check_with_place(os.path.abspath('../../legacy_test/parallel_dygraph_shared_unused_var.py'), delta=1e-05, check_error_log=True, log_name=flag_name)
if __name__ == '__main__':
    unittest.main()