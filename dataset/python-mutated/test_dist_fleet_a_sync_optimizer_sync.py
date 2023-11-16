import os
import unittest
import paddle
from paddle.distributed import fleet
from paddle.distributed.fleet.base import role_maker
paddle.enable_static()

class TestFleetGradientMergeMetaOptimizer(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        os.environ['PADDLE_PSERVER_NUMS'] = '2'
        os.environ['PADDLE_TRAINERS_NUM'] = '2'
        os.environ['POD_IP'] = '127.0.0.1'
        os.environ['PADDLE_PORT'] = '6007'
        os.environ['TRAINING_ROLE'] = 'TRAINER'
        os.environ['PADDLE_TRAINER_ID'] = '0'
        os.environ['PADDLE_TRAINERS_NUM'] = '2'
        os.environ['PADDLE_PSERVERS_IP_PORT_LIST'] = '127.0.0.1:36001,127.0.0.2:36001'

    def test_gradient_merge_optimizer(self):
        if False:
            print('Hello World!')
        fleet.init(role_maker.PaddleCloudRoleMaker())
        x = paddle.static.data(name='x', shape=[-1, 1], dtype='float32')
        y = paddle.static.data(name='y', shape=[-1, 1], dtype='float32')
        cost = paddle.nn.functional.square_error_cost(input=x, label=y)
        avg_cost = paddle.mean(cost)
        strategy = paddle.distributed.fleet.DistributedStrategy()
        strategy.a_sync = False
        strategy.a_sync_configs = {'launch_barrier': False}
        optimizer = paddle.optimizer.SGD(learning_rate=0.01)
        optimizer = fleet.distributed_optimizer(optimizer, strategy=strategy)
        optimizer.minimize(avg_cost)
        prog = paddle.base.default_main_program()
        self.assertEqual(prog.global_block().ops[-1].type, 'send_barrier')
        sends = 0
        sgds = 0
        for op in prog.global_block().ops:
            if op.type == 'send':
                sends += 1
            if op.type == 'sgd':
                sgds += 1
        self.assertEqual(sends, 0)
        self.assertEqual(sgds, 0)
if __name__ == '__main__':
    unittest.main()