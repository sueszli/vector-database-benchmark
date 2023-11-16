import os
import unittest
import paddle
from paddle import base
from paddle.distributed import fleet
from paddle.distributed.fleet.base import role_maker
paddle.enable_static()

class TestFleetLambMetaOptimizer(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        os.environ['PADDLE_TRAINER_ID'] = '1'
        os.environ['PADDLE_TRAINER_ENDPOINTS'] = '127.0.0.1:36001,127.0.0.1:36002'

    def net(self, main_prog, startup_prog):
        if False:
            for i in range(10):
                print('nop')
        with base.program_guard(main_prog, startup_prog):
            with base.unique_name.guard():
                input_x = paddle.static.data(name='x', shape=[-1, 32], dtype='float32')
                input_y = paddle.static.data(name='y', shape=[-1, 1], dtype='int64')
                fc_1 = paddle.static.nn.fc(x=input_x, size=64, activation='tanh')
                fc_2 = paddle.static.nn.fc(x=fc_1, size=256, activation='tanh')
                prediction = paddle.static.nn.fc(x=[fc_2], size=2, activation='softmax')
                cost = paddle.nn.functional.cross_entropy(input=prediction, label=input_y, reduction='none', use_softmax=False)
                avg_cost = paddle.mean(x=cost)
                strategy = paddle.distributed.fleet.DistributedStrategy()
                strategy.lamb = True
                strategy.lamb_configs = {'lamb_weight_decay': 0.01, 'exclude_from_weight_decay': []}
        return (avg_cost, strategy)

    def test_lamb_optimizer(self):
        if False:
            i = 10
            return i + 15
        role = role_maker.PaddleCloudRoleMaker(is_collective=True)
        fleet.init(role)
        startup_prog = base.Program()
        train_prog = base.Program()
        (avg_cost, strategy) = self.net(train_prog, startup_prog)
        optimizer = paddle.optimizer.Adam(learning_rate=0.01)
        optimizer = fleet.distributed_optimizer(optimizer, strategy=strategy)
        optimizer.minimize(avg_cost)
        ops = [op.type for op in avg_cost.block.ops]
        self.assertIn('lamb', ops)

    def test_lamb_not_apply_with_momentum(self):
        if False:
            i = 10
            return i + 15
        role = role_maker.PaddleCloudRoleMaker(is_collective=True)
        fleet.init(role)
        startup_prog = base.Program()
        train_prog = base.Program()
        (avg_cost, strategy) = self.net(train_prog, startup_prog)
        optimizer = paddle.optimizer.Momentum(learning_rate=0.1, momentum=0.9)
        optimizer = fleet.distributed_optimizer(optimizer, strategy=strategy)
        optimizer.minimize(avg_cost)
        ops = [op.type for op in avg_cost.block.ops]
        self.assertNotIn('lamb', ops)

    def test_lamb_exclude_fn(self):
        if False:
            i = 10
            return i + 15
        role = role_maker.PaddleCloudRoleMaker(is_collective=True)
        fleet.init(role)
        startup_prog = base.Program()
        train_prog = base.Program()
        (avg_cost, strategy) = self.net(train_prog, startup_prog)
        optimizer = paddle.optimizer.Adam(learning_rate=0.01)
        strategy.lamb_configs = {'lamb_weight_decay': 0.01, 'exclude_from_weight_decay': ['.b_0']}
        optimizer = fleet.distributed_optimizer(optimizer, strategy=strategy)
        optimizer.minimize(avg_cost)
        ops_without_wd = [op for op in avg_cost.block.ops if op.type == 'lamb' and op.attr('op_role_var')[0].endswith('.b_0')]
        for op in ops_without_wd:
            self.assertEqual(op.attr('weight_decay'), 0)

    def test_lamb_apply_with_amp(self):
        if False:
            while True:
                i = 10
        role = role_maker.PaddleCloudRoleMaker(is_collective=True)
        fleet.init(role)
        input_x = paddle.static.data(name='x', shape=[-1, 32], dtype='float32')
        input_y = paddle.static.data(name='y', shape=[-1, 1], dtype='int64')
        fc_1 = paddle.static.nn.fc(x=input_x, size=64, activation='tanh')
        fc_2 = paddle.static.nn.fc(x=fc_1, size=64, activation='tanh')
        prediction = paddle.static.nn.fc(x=[fc_2], size=2, activation='softmax')
        cost = paddle.nn.functional.cross_entropy(input=prediction, label=input_y, reduction='none', use_softmax=False)
        avg_cost = paddle.mean(x=cost)
        strategy = paddle.distributed.fleet.DistributedStrategy()
        strategy.amp = True
        strategy.amp_configs = {'init_loss_scaling': 32768, 'decr_every_n_nan_or_inf': 2, 'incr_every_n_steps': 1000, 'incr_ratio': 2.0, 'use_dynamic_loss_scaling': True, 'decr_ratio': 0.5, 'custom_white_list': ['softmax'], 'custom_black_list': ['tanh']}
        strategy.lamb = True
        strategy.lamb_configs = {'lamb_weight_decay': 0.01, 'exclude_from_weight_decay': []}
        optimizer = paddle.optimizer.Adam(learning_rate=0.01)
        optimizer = fleet.distributed_optimizer(optimizer, strategy=strategy)
        optimizer.minimize(avg_cost)
        ops = [op.type for op in avg_cost.block.ops]
        self.assertIn('lamb', ops)
        self.assertIn('cast', ops)
        self.assertIn('check_finite_and_unscale', ops)
if __name__ == '__main__':
    unittest.main()