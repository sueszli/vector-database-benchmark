import os
import unittest
from legacy_test.spawn_runner_base import TestDistSpawnRunner
from legacy_test.test_dist_base import TestDistBase
from parallel_dygraph_no_sync import TestNoSync
from parallel_dygraph_no_sync_control_flow import TestNoSyncControlFlow
from parallel_dygraph_no_sync_unused_params import TestNoSyncUnusedParam
from paddle import base
flag_name = os.path.splitext(__file__)[0]

class TestParallelDygraphNoSync(TestDistBase):

    def _setup_config(self):
        if False:
            i = 10
            return i + 15
        self._sync_mode = False
        self._nccl2_mode = True
        self._dygraph = True
        self._find_unused_parameters = False

    def test_no_sync(self):
        if False:
            return 10
        if base.core.is_compiled_with_cuda():
            self.check_with_place('parallel_dygraph_no_sync.py', delta=1e-05, check_error_log=True, log_name=flag_name)

class TestParallelDygraphNoSyncUnusedParam(TestDistBase):

    def _setup_config(self):
        if False:
            return 10
        self._sync_mode = False
        self._nccl2_mode = True
        self._dygraph = True
        self._find_unused_parameters = True

    def test_no_sync_ununsed_param(self):
        if False:
            for i in range(10):
                print('nop')
        if base.core.is_compiled_with_cuda():
            self.check_with_place('parallel_dygraph_no_sync_unused_params.py', delta=1e-05, check_error_log=True, log_name=flag_name)

class TestParallelDygraphNoSyncControlFlow(TestDistBase):

    def _setup_config(self):
        if False:
            return 10
        self._sync_mode = False
        self._nccl2_mode = True
        self._dygraph = True
        self._find_unused_parameters = True

    def test_no_sync_control_flow(self):
        if False:
            for i in range(10):
                print('nop')
        if base.core.is_compiled_with_cuda():
            self.check_with_place('parallel_dygraph_no_sync_control_flow.py', delta=1e-05, check_error_log=True, log_name=flag_name)

class TestParallelDygraphNoSyncSpawn(TestDistSpawnRunner):

    def test_no_sync_with_spawn(self):
        if False:
            for i in range(10):
                print('nop')
        if base.core.is_compiled_with_cuda():
            self.check_dist_result_with_spawn(test_class=TestNoSync, delta=1e-05)

class TestParallelDygraphNoSyncUnusedParamSpawn(TestDistSpawnRunner):

    def _args_config(self, args):
        if False:
            print('Hello World!')
        args.find_unused_parameters = True

    def test_no_sync_with_spawn(self):
        if False:
            i = 10
            return i + 15
        if base.core.is_compiled_with_cuda():
            self.check_dist_result_with_spawn(test_class=TestNoSyncUnusedParam, delta=1e-05)

class TestParallelDygraphNoSyncControlFlowSpawn(TestDistSpawnRunner):

    def _args_config(self, args):
        if False:
            i = 10
            return i + 15
        args.find_unused_parameters = True

    def test_no_sync_with_spawn(self):
        if False:
            i = 10
            return i + 15
        if base.core.is_compiled_with_cuda():
            self.check_dist_result_with_spawn(test_class=TestNoSyncControlFlow, delta=1e-05)
if __name__ == '__main__':
    unittest.main()