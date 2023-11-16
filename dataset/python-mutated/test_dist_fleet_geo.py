import os
os.environ['WITH_DISTRIBUTE'] = 'ON'
import unittest
from dist_fleet_simnet_bow import train_network
from test_dist_fleet_base import TestFleetBase
import paddle
from paddle.distributed import fleet
from paddle.distributed.fleet.base import role_maker
paddle.enable_static()

class TestDistGeoCtr_2x2(TestFleetBase):

    def _setup_config(self):
        if False:
            i = 10
            return i + 15
        self._mode = 'geo'
        self._reader = 'pyreader'
        self._geo_sgd_need_push_nums = 5

    def check_with_place(self, model_file, delta=0.001, check_error_log=False, need_envs={}):
        if False:
            for i in range(10):
                print('nop')
        required_envs = {'PATH': os.getenv('PATH', ''), 'PYTHONPATH': os.getenv('PYTHONPATH', ''), 'LD_LIBRARY_PATH': os.getenv('LD_LIBRARY_PATH', ''), 'FLAGS_rpc_deadline': '5000', 'http_proxy': '', 'LOG_DIRNAME': '/tmp', 'LOG_PREFIX': self.__class__.__name__}
        required_envs.update(need_envs)
        if check_error_log:
            required_envs['GLOG_v'] = '4'
            required_envs['GLOG_logtostderr'] = '1'
        (tr0_losses, tr1_losses) = self._run_cluster(model_file, required_envs)

    def test_dist_train(self):
        if False:
            while True:
                i = 10
        self.check_with_place('dist_fleet_ctr.py', delta=1e-05, check_error_log=False)

class TestGeoSgdTranspiler(unittest.TestCase):

    def test_pserver(self):
        if False:
            print('Hello World!')
        role = role_maker.UserDefinedRoleMaker(current_id=0, role=role_maker.Role.SERVER, worker_num=2, server_endpoints=['127.0.0.1:36011', '127.0.0.1:36012'])
        fleet.init(role)
        batch_size = 128
        is_sparse = True
        is_distribute = False
        strategy = paddle.distributed.fleet.DistributedStrategy()
        strategy.a_sync = True
        strategy.a_sync_configs = {'k_steps': 100, 'launch_barrier': False}
        (avg_cost, _, _, _) = train_network(batch_size, is_distribute, is_sparse)
        optimizer = paddle.optimizer.SGD(0.1)
        optimizer = fleet.distributed_optimizer(optimizer, strategy)
        optimizer.minimize(avg_cost)
if __name__ == '__main__':
    unittest.main()