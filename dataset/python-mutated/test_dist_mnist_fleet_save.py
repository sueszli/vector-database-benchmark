import os
import shutil
import unittest
from test_dist_base import TestDistBase
import paddle
paddle.enable_static()

class TestDistMnistFleetSave(TestDistBase):

    def _setup_config(self):
        if False:
            print('Hello World!')
        self._sync_mode = True
        self._use_reduce = False
        self._use_reader_alloc = False
        self._nccl2_mode = True
        self._use_fleet_api = True
        self._save_model = True

    def _rm_temp_files(self, dirname):
        if False:
            while True:
                i = 10
        base_model_path = os.path.join(dirname, 'base_persistables')
        fleet_model_path = os.path.join(dirname, 'fleet_persistables')
        base_infer_path = os.path.join(dirname, 'base_infer')
        fleet_infer_path = os.path.join(dirname, 'fleet_infer')
        base_model_path_2 = os.path.join(dirname, 'base_persistables_2')
        fleet_model_path_2 = os.path.join(dirname, 'fleet_persistables_2')
        base_infer_path_2 = os.path.join(dirname, 'base_infer_2')
        fleet_infer_path_2 = os.path.join(dirname, 'fleet_infer_2')
        shutil.rmtree(base_model_path)
        shutil.rmtree(fleet_model_path)
        shutil.rmtree(base_infer_path)
        shutil.rmtree(fleet_infer_path)
        shutil.rmtree(base_model_path_2)
        shutil.rmtree(fleet_model_path_2)
        shutil.rmtree(base_infer_path_2)
        shutil.rmtree(fleet_infer_path_2)

    def _test_saved_files(self, dirname):
        if False:
            while True:
                i = 10
        base_model_path = os.path.join(dirname, 'base_persistables')
        base_persistables = sorted(os.listdir(base_model_path))
        fleet_model_path = os.path.join(dirname, 'fleet_persistables')
        fleet_persistables = sorted(os.listdir(fleet_model_path))
        base_infer_path = os.path.join(dirname, 'base_infer')
        base_infer_files = sorted(os.listdir(base_infer_path))
        fleet_infer_path = os.path.join(dirname, 'fleet_infer')
        fleet_infer_files = sorted(os.listdir(fleet_infer_path))
        if len(base_persistables) != len(fleet_persistables):
            self._rm_temp_files(dirname)
            raise ValueError('Test Failed.')
        for i in range(len(base_persistables)):
            if base_persistables[i] != fleet_persistables[i]:
                self._rm_temp_files(dirname)
                raise ValueError('Test Failed.')
        if len(base_infer_files) != len(fleet_infer_files):
            self._rm_temp_files(dirname)
            raise ValueError('Test Failed.')
        for i in range(len(base_infer_files)):
            if base_infer_files[i] != fleet_infer_files[i]:
                self._rm_temp_files(dirname)
                raise ValueError('Test Failed.')
        self._rm_temp_files(dirname)
        return True

    def check_with_place(self, model_file, delta=0.001, check_error_log=False, need_envs={}, log_name=''):
        if False:
            for i in range(10):
                print('nop')
        required_envs = self._get_required_envs(check_error_log, need_envs)
        (tr0_losses, tr1_losses) = self._run_cluster_nccl2(model_file, required_envs, update_method='nccl2', check_error_log=check_error_log, log_name=log_name)
        dirname = '/tmp'
        self._test_saved_files(dirname)

    def test_dist_train(self):
        if False:
            while True:
                i = 10
        from paddle import base
        if base.core.is_compiled_with_cuda():
            self.check_with_place('dist_mnist.py', delta=1e-05)
if __name__ == '__main__':
    unittest.main()