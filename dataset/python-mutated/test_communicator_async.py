import os
import time
import unittest
import paddle
paddle.enable_static()
from paddle.distributed import fleet
from paddle.distributed.fleet.base import role_maker

class TestCommunicator(unittest.TestCase):

    def net(self):
        if False:
            return 10
        x = paddle.static.data(name='x', shape=[-1, 1], dtype='float32')
        y = paddle.static.data(name='y', shape=[-1, 1], dtype='float32')
        cost = paddle.nn.functional.square_error_cost(input=x, label=y)
        avg_cost = paddle.mean(cost)
        return avg_cost

    def test_communicator_async(self):
        if False:
            for i in range(10):
                print('nop')
        role = role_maker.UserDefinedRoleMaker(current_id=0, role=role_maker.Role.WORKER, worker_num=2, server_endpoints=['127.0.0.1:6001', '127.0.0.1:6002'])
        fleet.init(role)
        avg_cost = self.net()
        optimizer = paddle.optimizer.SGD(0.01)
        strategy = paddle.distributed.fleet.DistributedStrategy()
        strategy.a_sync = True
        strategy.a_sync_configs = {'launch_barrier': False}
        optimizer = fleet.distributed_optimizer(optimizer, strategy)
        optimizer.minimize(avg_cost)
        os.environ['TEST_MODE'] = '1'
        fleet.init_worker()
        time.sleep(10)
        fleet.stop_worker()
if __name__ == '__main__':
    unittest.main()