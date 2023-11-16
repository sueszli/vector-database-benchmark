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
            for i in range(10):
                print('nop')
        x = paddle.static.data(name='x', shape=[-1, 1], dtype='float32')
        y = paddle.static.data(name='y', shape=[-1, 1], dtype='float32')
        cost = paddle.nn.functional.square_error_cost(input=x, label=y)
        avg_cost = paddle.mean(cost)
        return avg_cost

    def test_communicator_sync(self):
        if False:
            for i in range(10):
                print('nop')
        os.environ['TRAINING_ROLE'] = 'TRAINER'
        os.environ['PADDLE_PSERVER_NUMS'] = '2'
        os.environ['PADDLE_TRAINERS_NUM'] = '2'
        os.environ['POD_IP'] = '127.0.0.1'
        os.environ['PADDLE_PORT'] = '36001'
        os.environ['PADDLE_TRAINER_ID'] = '0'
        os.environ['PADDLE_TRAINERS_NUM'] = '2'
        os.environ['PADDLE_PSERVERS_IP_PORT_LIST'] = '127.0.0.1:36001,127.0.0.2:36001'
        fleet.init(role_maker.PaddleCloudRoleMaker())
        avg_cost = self.net()
        optimizer = paddle.optimizer.SGD(0.01)
        strategy = paddle.distributed.fleet.DistributedStrategy()
        strategy.a_sync = False
        strategy.a_sync_configs = {'launch_barrier': False}
        optimizer = fleet.distributed_optimizer(optimizer, strategy)
        optimizer.minimize(avg_cost)
        os.environ['TEST_MODE'] = '1'
        fleet.init_worker()
        time.sleep(10)
        fleet.stop_worker()
if __name__ == '__main__':
    unittest.main()