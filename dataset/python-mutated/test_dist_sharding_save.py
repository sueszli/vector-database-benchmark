import os
import shutil
import unittest
from test_dist_base import TestDistBase
import paddle
paddle.enable_static()

class TestDistMnistFleetSave(TestDistBase):

    def _setup_config(self):
        if False:
            return 10
        self._sync_mode = True
        self._use_reduce = False
        self._use_reader_alloc = False
        self._nccl2_mode = True
        self._use_fleet_api = True
        self._sharding_save = True
        self._enforce_place = 'GPU'

    def _rm_temp_files(self, dirname):
        if False:
            return 10
        shutil.rmtree(dirname)

    def _test_saved_files(self, dirname):
        if False:
            while True:
                i = 10
        sharding_save_files = sorted(os.listdir(dirname))
        check_files = ['fc_0.b_0', 'fc_0.b_0_velocity_0', 'fc_0.w_0', 'fc_0.w_0_velocity_0', 'fc_1.b_0', 'fc_1.b_0_velocity_0', 'fc_1.w_0', 'fc_1.w_0_velocity_0', 'fc_2.b_0', 'fc_2.b_0_velocity_0', 'fc_2.w_0', 'fc_2.w_0_velocity_0', 'learning_rate_0']
        if sharding_save_files != check_files:
            self._rm_temp_files(dirname)
            raise ValueError('Test Failed.')
        self._rm_temp_files(dirname)
        return True

    def check_with_place(self, model_file, delta=0.001, check_error_log=True, need_envs={}, log_name=''):
        if False:
            for i in range(10):
                print('nop')
        required_envs = self._get_required_envs(check_error_log, need_envs)
        (tr0_losses, tr1_losses) = self._run_cluster_nccl2(model_file, required_envs, update_method='nccl2', check_error_log=check_error_log, log_name=log_name)
        dirname = './ut_sharding_save_model'
        self._test_saved_files(dirname)

    def test_dist_train(self):
        if False:
            for i in range(10):
                print('nop')
        from paddle import base
        if base.core.is_compiled_with_cuda():
            self.check_with_place('dist_sharding_save.py', delta=1e-05)
if __name__ == '__main__':
    unittest.main()