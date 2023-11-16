import os
import unittest
from legacy_test.test_dist_base import TestDistBase
from paddle import base
flag_name = os.path.splitext(__file__)[0]

class TestDistMnistGradMerge(TestDistBase):

    def _setup_config(self):
        if False:
            i = 10
            return i + 15
        self._sync_mode = True
        self._use_reduce = False
        self._nccl2_mode = True
        self._nccl2_reduce_layer = True

    def test_dist_train(self):
        if False:
            return 10
        if base.core.is_compiled_with_cuda():
            self.check_with_place('dist_mnist_gradient_merge.py', delta=1e-05, check_error_log=True, log_name=flag_name)

class TestDistMnistGradMergeNoFuse(TestDistBase):

    def _setup_config(self):
        if False:
            return 10
        self._sync_mode = True
        self._use_reduce = False
        self._nccl2_mode = True
        self._fuse_all_reduce = False

    def test_dist_train(self):
        if False:
            for i in range(10):
                print('nop')
        if base.core.is_compiled_with_cuda():
            self.check_with_place('dist_mnist_gradient_merge.py', delta=1e-05, check_error_log=True, log_name=flag_name + '_no_fuse')

class TestDistMnistGradMergeRawOptimizerBase(TestDistBase):

    def _setup_config(self):
        if False:
            while True:
                i = 10
        self._use_reader_alloc = False
        self._nccl2_mode = True
        self._use_fleet_api = True
        self._use_fleet_api_20 = True

    def enable_avg(self):
        if False:
            i = 10
            return i + 15
        return False

    def test_dist_train(self):
        if False:
            return 10
        if base.core.is_compiled_with_cuda():
            avg = str(self.enable_avg())
            log_name = flag_name + '_raw_optimizer_gm_avg_' + avg
            self.check_with_place('dist_mnist_gradient_merge_raw_optimizer.py', delta=1e-05, check_error_log=True, log_name=log_name, need_envs={'FLAGS_apply_pass_to_program': '1', 'enable_gm_avg': avg})

class TestDistMnistGradMergeRawOptimizerAvg(TestDistMnistGradMergeRawOptimizerBase):

    def enable_avg(self):
        if False:
            i = 10
            return i + 15
        return True
if __name__ == '__main__':
    unittest.main()