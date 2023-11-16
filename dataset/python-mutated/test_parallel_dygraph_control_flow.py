import os
import unittest
from legacy_test.test_dist_base import TestDistBase
from paddle import base
flag_name = os.path.splitext(__file__)[0]

class TestDygraphControlFlowSame(TestDistBase):

    def _setup_config(self):
        if False:
            print('Hello World!')
        self._sync_mode = False
        self._nccl2_mode = True
        self._dygraph = True
        self._find_unused_parameters = True

    def test_net(self):
        if False:
            print('Hello World!')
        if base.core.is_compiled_with_cuda():
            self.check_with_place('parallel_dygraph_control_flow_same.py', delta=1e-05, check_error_log=True, log_name=flag_name)

class TestFleetDygraphControlFlowSame(TestDygraphControlFlowSame):

    def _setup_config(self):
        if False:
            print('Hello World!')
        self._sync_mode = False
        self._nccl2_mode = True
        self._dygraph = True
        self._use_fleet_api = True
        self._find_unused_parameters = True

class TestFleetDygraphControlFlowSameAccGrad(TestDygraphControlFlowSame):

    def _setup_config(self):
        if False:
            return 10
        self._sync_mode = False
        self._nccl2_mode = True
        self._dygraph = True
        self._accumulate_gradient = True
        self._find_unused_parameters = True

class TestDygraphControlFlowDiff(TestDistBase):

    def _setup_config(self):
        if False:
            i = 10
            return i + 15
        self._sync_mode = False
        self._nccl2_mode = True
        self._dygraph = True
        self._find_unused_parameters = True

    def test_net(self):
        if False:
            for i in range(10):
                print('nop')
        if base.core.is_compiled_with_cuda():
            self.check_with_place('parallel_dygraph_control_flow_different.py', delta=1e-05, check_error_log=True, log_name=flag_name)

class TestFleetDygraphControlFlowDiff(TestDygraphControlFlowDiff):

    def _setup_config(self):
        if False:
            while True:
                i = 10
        self._sync_mode = False
        self._nccl2_mode = True
        self._dygraph = True
        self._use_fleet_api = True
        self._find_unused_parameters = True

class TestFleetDygraphControlFlowDiffAccGrad(TestDygraphControlFlowDiff):

    def _setup_config(self):
        if False:
            while True:
                i = 10
        self._sync_mode = False
        self._nccl2_mode = True
        self._dygraph = True
        self._accumulate_gradient = True
        self._find_unused_parameters = True
if __name__ == '__main__':
    unittest.main()