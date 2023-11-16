import os
import shutil
import tempfile
import unittest
from test_dist_fleet_base import TestFleetBase

class TestDistCtrInfer(TestFleetBase):

    def _setup_config(self):
        if False:
            print('Hello World!')
        self._mode = 'async'
        self._reader = 'pyreader'

    def check_with_place(self, model_file, delta=0.001, check_error_log=False, need_envs={}):
        if False:
            return 10
        required_envs = {'PATH': os.getenv('PATH', ''), 'PYTHONPATH': os.getenv('PYTHONPATH', ''), 'LD_LIBRARY_PATH': os.getenv('LD_LIBRARY_PATH', ''), 'FLAGS_rpc_deadline': '30000', 'http_proxy': '', 'FLAGS_communicator_send_queue_size': '2', 'FLAGS_communicator_max_merge_var_num': '2', 'CPU_NUM': '2', 'LOG_DIRNAME': '/tmp', 'LOG_PREFIX': self.__class__.__name__}
        required_envs.update(need_envs)
        if check_error_log:
            required_envs['GLOG_v'] = '3'
            required_envs['GLOG_logtostderr'] = '1'
        (tr0_losses, tr1_losses) = self._run_cluster(model_file, required_envs)

    def test_dist_infer(self):
        if False:
            i = 10
            return i + 15
        model_dirname = tempfile.mkdtemp()
        self.check_with_place('dist_fleet_ctr.py', delta=1e-05, check_error_log=False, need_envs={'SAVE_DIRNAME': model_dirname})
        self._need_test = 1
        self._model_dir = model_dirname
        self.check_with_place('dist_fleet_ctr.py', delta=1e-05, check_error_log=False)
        shutil.rmtree(model_dirname)

class TestDistCtrTrainInfer(TestFleetBase):

    def _setup_config(self):
        if False:
            i = 10
            return i + 15
        self._mode = 'async'
        self._reader = 'pyreader'
        self._need_test = 1

    def check_with_place(self, model_file, delta=0.001, check_error_log=False, need_envs={}):
        if False:
            while True:
                i = 10
        required_envs = {'PATH': os.getenv('PATH', ''), 'PYTHONPATH': os.getenv('PYTHONPATH', ''), 'LD_LIBRARY_PATH': os.getenv('LD_LIBRARY_PATH', ''), 'FLAGS_rpc_deadline': '30000', 'http_proxy': '', 'FLAGS_communicator_send_queue_size': '2', 'FLAGS_communicator_max_merge_var_num': '2', 'CPU_NUM': '2', 'LOG_DIRNAME': '/tmp', 'LOG_PREFIX': self.__class__.__name__}
        required_envs.update(need_envs)
        if check_error_log:
            required_envs['GLOG_v'] = '3'
            required_envs['GLOG_logtostderr'] = '1'
        (tr0_losses, tr1_losses) = self._run_cluster(model_file, required_envs)

    def test_dist_train_infer(self):
        if False:
            while True:
                i = 10
        self.check_with_place('dist_fleet_ctr.py', delta=1e-05, check_error_log=False)
if __name__ == '__main__':
    unittest.main()