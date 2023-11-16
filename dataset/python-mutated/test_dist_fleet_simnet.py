import os
import unittest
from test_dist_fleet_base import TestFleetBase
import paddle
paddle.enable_static()

class TestDistSimnetASync2x2(TestFleetBase):

    def _setup_config(self):
        if False:
            return 10
        self._mode = 'async'
        self._reader = 'pyreader'

    def check_with_place(self, model_file, delta=0.001, check_error_log=False, need_envs={}):
        if False:
            while True:
                i = 10
        required_envs = {'PATH': os.getenv('PATH', ''), 'PYTHONPATH': os.getenv('PYTHONPATH', ''), 'LD_LIBRARY_PATH': os.getenv('LD_LIBRARY_PATH', ''), 'FLAGS_rpc_deadline': '5000', 'http_proxy': '', 'CPU_NUM': '2'}
        required_envs.update(need_envs)
        if check_error_log:
            required_envs['GLOG_v'] = '3'
            required_envs['GLOG_logtostderr'] = '1'
        (tr0_losses, tr1_losses) = self._run_cluster(model_file, required_envs)

    def test_dist_train(self):
        if False:
            print('Hello World!')
        self.check_with_place('dist_fleet_simnet_bow.py', delta=1e-05, check_error_log=True)
if __name__ == '__main__':
    unittest.main()