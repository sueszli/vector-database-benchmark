import os
import unittest
from parallel_dygraph_se_resnext import TestSeResNeXt
from spawn_runner_base import TestDistSpawnRunner
from test_dist_base import TestDistBase
from paddle import base
flag_name = os.path.splitext(__file__)[0]

class TestParallelDygraphSeResNeXt(TestDistBase):

    def _setup_config(self):
        if False:
            while True:
                i = 10
        self._sync_mode = False
        self._nccl2_mode = True
        self._dygraph = True

    def test_se_resnext(self):
        if False:
            for i in range(10):
                print('nop')
        if base.core.is_compiled_with_cuda():
            self.check_with_place('parallel_dygraph_se_resnext.py', delta=0.01, check_error_log=True, log_name=flag_name)

class TestParallelDygraphSeResNeXtSpawn(TestDistSpawnRunner):

    def test_se_resnext_with_spawn(self):
        if False:
            i = 10
            return i + 15
        if base.core.is_compiled_with_cuda():
            self.check_dist_result_with_spawn(test_class=TestSeResNeXt, delta=0.01)
if __name__ == '__main__':
    unittest.main()