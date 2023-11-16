import os
import tempfile
import unittest
from test_dist_fleet_base import TestFleetBase

@unittest.skip(reason='Skip unstable ut, need paddle sync mode fix')
class TestDistMnistSync2x2(TestFleetBase):

    def _setup_config(self):
        if False:
            i = 10
            return i + 15
        self._mode = 'sync'
        self._reader = 'pyreader'
        self._need_test = 1

    def check_with_place(self, model_file, delta=0.001, check_error_log=False, need_envs={}):
        if False:
            while True:
                i = 10
        required_envs = {'PATH': os.getenv('PATH', ''), 'PYTHONPATH': os.getenv('PYTHONPATH', ''), 'LD_LIBRARY_PATH': os.getenv('LD_LIBRARY_PATH', ''), 'FLAGS_rpc_deadline': '5000', 'http_proxy': '', 'CPU_NUM': '2', 'LOG_DIRNAME': '/tmp', 'LOG_PREFIX': self.__class__.__name__}
        required_envs.update(need_envs)
        if check_error_log:
            required_envs['GLOG_v'] = '3'
            required_envs['GLOG_logtostderr'] = '1'
        (tr0_losses, tr1_losses) = self._run_cluster(model_file, required_envs)

    def test_dist_train(self):
        if False:
            print('Hello World!')
        self.check_with_place('dist_fleet_ctr.py', delta=1e-05, check_error_log=False)

class TestDistMnistAsyncDataset2x2(TestFleetBase):

    def _setup_config(self):
        if False:
            print('Hello World!')
        self._mode = 'async'
        self._reader = 'dataset'

    def check_with_place(self, model_file, delta=0.001, check_error_log=False, need_envs={}):
        if False:
            for i in range(10):
                print('nop')
        required_envs = {'PATH': os.getenv('PATH', ''), 'PYTHONPATH': os.getenv('PYTHONPATH', ''), 'LD_LIBRARY_PATH': os.getenv('LD_LIBRARY_PATH', ''), 'FLAGS_rpc_deadline': '5000', 'http_proxy': '', 'SAVE_MODEL': '1', 'dump_param': 'concat_0.tmp_0', 'dump_fields': 'dnn-fc-3.tmp_0,dnn-fc-3.tmp_0@GRAD', 'dump_fields_path': tempfile.mkdtemp(), 'Debug': '1', 'LOG_DIRNAME': '/tmp', 'LOG_PREFIX': self.__class__.__name__}
        required_envs.update(need_envs)
        if check_error_log:
            required_envs['GLOG_v'] = '3'
            required_envs['GLOG_logtostderr'] = '1'
        (tr0_losses, tr1_losses) = self._run_cluster(model_file, required_envs)

    def test_dist_train(self):
        if False:
            print('Hello World!')
        self.check_with_place('dist_fleet_ctr.py', delta=1e-05, check_error_log=False)
if __name__ == '__main__':
    unittest.main()