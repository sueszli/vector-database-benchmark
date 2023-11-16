import os
import unittest
import paddle
from paddle.distributed.fleet.base import role_maker
paddle.enable_static()

class TestFleetGradientMergeMetaOptimizer(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        os.environ['PADDLE_PSERVER_NUMS'] = '2'
        os.environ['PADDLE_TRAINERS_NUM'] = '2'
        os.environ['POD_IP'] = '127.0.0.1'
        os.environ['PADDLE_PORT'] = '36001'
        os.environ['PADDLE_TRAINER_ID'] = '0'
        os.environ['PADDLE_TRAINERS_NUM'] = '2'
        os.environ['PADDLE_PSERVERS_IP_PORT_LIST'] = '127.0.0.1:36001,127.0.0.2:36001'

    def test_a_sync_optimizer1(self):
        if False:
            print('Hello World!')
        os.environ['TRAINING_ROLE'] = 'TRAINER'
        from paddle.distributed import fleet
        main_program = paddle.base.Program()
        startup_program = paddle.base.Program()
        paddle.base.framework.switch_main_program(main_program)
        paddle.base.framework.switch_startup_program(startup_program)
        fleet.init(role_maker.PaddleCloudRoleMaker())
        input_x = paddle.static.data(name='x', shape=[-1, 32], dtype='float32')
        input_y = paddle.static.data(name='y', shape=[-1, 1], dtype='int64')
        fc_1 = paddle.static.nn.fc(x=input_x, size=64, activation='tanh')
        fc_2 = paddle.static.nn.fc(x=fc_1, size=64, activation='tanh')
        prediction = paddle.static.nn.fc(x=[fc_2], size=2, activation='softmax')
        cost = paddle.nn.functional.cross_entropy(input=prediction, label=input_y, reduction='none', use_softmax=False)
        avg_cost = paddle.mean(x=cost)
        os.environ['FLAGS_LAUNCH_BARRIER'] = '0'
        strategy = paddle.distributed.fleet.DistributedStrategy()
        strategy.auto = True
        optimizer = paddle.optimizer.Adam(learning_rate=0.01)
        optimizer = fleet.distributed_optimizer(optimizer, strategy=strategy)
        optimizer.minimize(avg_cost)
        self.assertTrue(fleet._final_strategy().a_sync)
        a_sync_configs = fleet._final_strategy().a_sync_configs
        self.assertTrue(a_sync_configs['k_steps'] == 0)
if __name__ == '__main__':
    unittest.main()