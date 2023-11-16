import unittest
from test_dist_base import TestDistBase
import paddle
paddle.enable_static()

class TestDistMnistNCCL2FleetApi(TestDistBase):

    def _setup_config(self):
        if False:
            for i in range(10):
                print('nop')
        self._sync_mode = True
        self._use_reduce = False
        self._use_reader_alloc = False
        self._nccl2_mode = True
        self._use_fleet_api = True
        self._sync_batch_norm = True

    def test_dist_train(self):
        if False:
            for i in range(10):
                print('nop')
        from paddle import base
        if base.core.is_compiled_with_cuda():
            self.check_with_place('dist_mnist.py', delta=1e-05, check_error_log=True, need_envs={'FLAGS_allreduce_record_one_event': '1'})

class FleetCollectiveTest(unittest.TestCase):

    def test_open_sync_batch_norm(self):
        if False:
            i = 10
            return i + 15
        from paddle import base
        from paddle.incubate.distributed.fleet import role_maker
        from paddle.incubate.distributed.fleet.collective import DistributedStrategy, fleet
        if not base.core.is_compiled_with_cuda():
            return
        data = paddle.static.data(name='X', shape=[-1, 1], dtype='float32')
        hidden = paddle.static.nn.fc(x=data, size=10)
        loss = paddle.mean(hidden)
        optimizer = paddle.optimizer.Adam()
        role = role_maker.UserDefinedCollectiveRoleMaker(0, ['127.0.0.1:6170'])
        fleet.init(role)
        dist_strategy = DistributedStrategy()
        dist_strategy.sync_batch_norm = True
        dist_optimizer = fleet.distributed_optimizer(optimizer, strategy=dist_strategy)
        dist_optimizer.minimize(loss)
        self.assertEqual(dist_strategy.exec_strategy.num_threads, 1)
if __name__ == '__main__':
    unittest.main()