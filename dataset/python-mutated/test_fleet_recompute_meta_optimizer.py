import unittest
from fleet_meta_optimizer_base import TestFleetMetaOptimizer
import paddle
from paddle import base
from paddle.distributed.fleet.meta_optimizers import RecomputeOptimizer
paddle.enable_static()

class TestFleetRecomputeMetaOptimizer(TestFleetMetaOptimizer):

    def test_recompute_optimizer_backward(self):
        if False:
            while True:
                i = 10
        'test recompute optimizer backward'
        (train_prog, startup_prog) = (base.Program(), base.Program())
        (avg_cost, strategy) = self.net(train_prog, startup_prog)
        self.set_strategy(strategy, 'recompute')
        opt = paddle.optimizer.Momentum(learning_rate=0.001, momentum=0.9)
        opt = RecomputeOptimizer(opt)
        opt.user_defined_strategy = strategy
        params_grads = opt.backward(avg_cost, startup_prog)
        outs = [op.output('Out')[0] for op in avg_cost.block.ops if op.type == 'mul']
        self.assertIn('subprog', ''.join(outs))

    def test_recompute_optimizer_backward_gradients(self):
        if False:
            print('Hello World!')
        'test recompute optimizer backward + gradients'
        (train_prog, startup_prog) = (base.Program(), base.Program())
        (avg_cost, strategy) = self.net(train_prog, startup_prog)
        self.set_strategy(strategy, 'recompute')
        opt = paddle.optimizer.Momentum(learning_rate=0.001, momentum=0.9)
        opt = RecomputeOptimizer(opt)
        opt.user_defined_strategy = strategy
        params_grads = opt.backward(avg_cost, startup_prog)
        with base.program_guard(train_prog, startup_prog):
            opt.apply_gradients(params_grads)
        outs = [op.output('Out')[0] for op in avg_cost.block.ops if op.type == 'mul']
        self.assertIn('subprog', ''.join(outs))

    def test_recompute_optimizer_backward_optimize(self):
        if False:
            i = 10
            return i + 15
        'test recompute optimizer backward + optimize'
        (train_prog, startup_prog) = (base.Program(), base.Program())
        (avg_cost, strategy) = self.net(train_prog, startup_prog)
        self.set_strategy(strategy, 'recompute')
        opt = paddle.optimizer.Momentum(learning_rate=0.001, momentum=0.9)
        opt = RecomputeOptimizer(opt)
        opt.user_defined_strategy = strategy
        params_grads = opt.backward(avg_cost, startup_prog)
        opt.apply_optimize(avg_cost, startup_prog, params_grads)
        outs = [op.output('Out')[0] for op in avg_cost.block.ops if op.type == 'mul']
        self.assertIn('subprog', ''.join(outs))

    def test_recompute_optimizer(self):
        if False:
            return 10
        (train_prog, startup_prog) = (base.Program(), base.Program())
        (avg_cost, strategy) = self.net(train_prog, startup_prog)
        self.set_strategy(strategy, 'recompute')
        self.optimizer(avg_cost, strategy, train_prog, startup_prog)
        outs = [op.output('Out')[0] for op in avg_cost.block.ops if op.type == 'mul']
        self.assertIn('subprog', ''.join(outs))

    def test_recompute_lars_optimizer(self):
        if False:
            while True:
                i = 10
        (train_prog, startup_prog) = (base.Program(), base.Program())
        (avg_cost, strategy) = self.net(train_prog, startup_prog)
        self.set_strategy(strategy, 'recompute')
        self.set_strategy(strategy, 'lars')
        self.optimizer(avg_cost, strategy, train_prog, startup_prog)
        ops = [op.type for op in avg_cost.block.ops]
        outs = [op.output('Out')[0] for op in avg_cost.block.ops if op.type == 'mul']
        self.assertIn('subprog', ''.join(outs))
        self.assertIn('lars_momentum', ops)

    def test_recompute_lamb_optimizer(self):
        if False:
            for i in range(10):
                print('nop')
        (train_prog, startup_prog) = (base.Program(), base.Program())
        (avg_cost, strategy) = self.net(train_prog, startup_prog)
        self.set_strategy(strategy, 'recompute')
        self.set_strategy(strategy, 'lamb')
        self.optimizer(avg_cost, strategy, train_prog, startup_prog, 'adam')
        ops = [op.type for op in avg_cost.block.ops]
        outs = [op.output('Out')[0] for op in avg_cost.block.ops if op.type == 'mul']
        self.assertIn('subprog', ''.join(outs))
        self.assertIn('lamb', ops)

    def test_recompute_offload(self):
        if False:
            print('Hello World!')
        (train_prog, startup_prog) = (base.Program(), base.Program())
        (avg_cost, strategy) = self.net(train_prog, startup_prog)
        self.set_strategy(strategy, 'recompute-offload')
        self.optimizer(avg_cost, strategy, train_prog, startup_prog)
        ops = [op.type for op in avg_cost.block.ops]
        outs = [op.output('Out')[0] for op in avg_cost.block.ops if op.type == 'memcpy']
        self.assertIn('memcpy', ops)
        self.assertIn('@Pinned', ''.join(outs))
        self.assertIn('@Fetch', ''.join(outs))
if __name__ == '__main__':
    unittest.main()