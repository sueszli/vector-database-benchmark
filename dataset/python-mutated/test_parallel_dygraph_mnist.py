import os
import unittest
from legacy_test.parallel_dygraph_mnist import TestMnist
from legacy_test.spawn_runner_base import TestDistSpawnRunner
from legacy_test.test_dist_base import TestDistBase
from paddle import base
flag_name = os.path.splitext(__file__)[0]

class TestParallelDygraphMnist(TestDistBase):

    def _setup_config(self):
        if False:
            return 10
        self._sync_mode = False
        self._nccl2_mode = True
        self._dygraph = True
        self._find_unused_parameters = True

    def test_mnist(self):
        if False:
            i = 10
            return i + 15
        if base.core.is_compiled_with_cuda():
            self.check_with_place(os.path.abspath('../../legacy_test/parallel_dygraph_mnist.py'), delta=1e-05, check_error_log=True, log_name=flag_name)

class TestParallelDygraphMnistXPU(TestDistBase):

    def _setup_config(self):
        if False:
            print('Hello World!')
        self._sync_mode = False
        self._bkcl_mode = True
        self._dygraph = True
        self._enforce_place = 'XPU'

    def test_mnist_xpu(self):
        if False:
            while True:
                i = 10
        if base.core.is_compiled_with_xpu():
            self.check_with_place(os.path.abspath('../../legacy_test/parallel_dygraph_mnist.py'), delta=0.0001, check_error_log=True, log_name=flag_name)

class TestParallelDygraphMnistSpawn(TestDistSpawnRunner):

    def test_mnist_with_spawn(self):
        if False:
            for i in range(10):
                print('nop')
        if base.core.is_compiled_with_cuda():
            self.check_dist_result_with_spawn(test_class=TestMnist, delta=1e-05)

class TestParallelDygraphMnistAccGrad(TestDistBase):

    def _setup_config(self):
        if False:
            print('Hello World!')
        self._sync_mode = False
        self._nccl2_mode = True
        self._dygraph = True
        self._use_fleet_api = True
        self._accumulate_gradient = True
        self._find_unused_parameters = False

    def test_mnist(self):
        if False:
            while True:
                i = 10
        if base.core.is_compiled_with_cuda():
            self.check_with_place(os.path.abspath('../../legacy_test/parallel_dygraph_mnist.py'), delta=1e-05, check_error_log=True, log_name=flag_name)

class TestFleetDygraphMnistXPU(TestDistBase):

    def _setup_config(self):
        if False:
            return 10
        self._sync_mode = False
        self._bkcl_mode = True
        self._dygraph = True
        self._enforce_place = 'XPU'
        self._use_fleet_api = True

    def test_mnist(self):
        if False:
            i = 10
            return i + 15
        if base.core.is_compiled_with_xpu():
            self.check_with_place(os.path.abspath('../../legacy_test/parallel_dygraph_mnist.py'), delta=0.0001, check_error_log=True, log_name=flag_name)
if __name__ == '__main__':
    unittest.main()