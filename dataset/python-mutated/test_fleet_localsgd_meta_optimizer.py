import unittest
from fleet_meta_optimizer_base import TestFleetMetaOptimizer
import paddle
from paddle import base
paddle.enable_static()

class TestFleetLocalSGDMetaOptimizer(TestFleetMetaOptimizer):

    def test_localsgd_optimizer(self):
        if False:
            return 10
        (train_prog, startup_prog) = (base.Program(), base.Program())
        (avg_cost, strategy) = self.net(train_prog, startup_prog)
        self.set_strategy(strategy, 'localsgd')
        self.optimizer(avg_cost, strategy, train_prog, startup_prog)
        ops = [op.type for op in avg_cost.block.ops]
        outs = [''.join(op.output('Out')) for op in avg_cost.block.ops if op.type == 'conditional_block']
        self.assertIn('conditional_block', ops)
        self.assertIn('@SNAPSHOT', ''.join(outs))

    def test_localsgd_amp_optimizer(self):
        if False:
            i = 10
            return i + 15
        (train_prog, startup_prog) = (base.Program(), base.Program())
        (avg_cost, strategy) = self.net(train_prog, startup_prog)
        self.set_strategy(strategy, 'localsgd')
        self.set_strategy(strategy, 'amp')
        self.optimizer(avg_cost, strategy, train_prog, startup_prog)
        ops = [op.type for op in avg_cost.block.ops]
        outs = [''.join(op.output('Out')) for op in avg_cost.block.ops if op.type == 'conditional_block']
        self.assertIn('conditional_block', ops)
        self.assertIn('@SNAPSHOT', ''.join(outs))
        self.assertIn('cast', ops)
        self.assertIn('check_finite_and_unscale', ops)

class TestFleetAdaptiveLocalSGDMetaOptimizer(TestFleetMetaOptimizer):

    def test_adaptive_localsgd_optimizer(self):
        if False:
            while True:
                i = 10
        (train_prog, startup_prog) = (base.Program(), base.Program())
        (avg_cost, strategy) = self.net(train_prog, startup_prog)
        self.set_strategy(strategy, 'adaptive_localsgd')
        self.optimizer(avg_cost, strategy, train_prog, startup_prog)
        ops = [op.type for op in avg_cost.block.ops]
        outs = [''.join(op.output('Out')) for op in avg_cost.block.ops if op.type == 'conditional_block']
        self.assertIn('conditional_block', ops)
        self.assertIn('@SNAPSHOT', ''.join(outs))

    def test_localsgd_amp_optimizer(self):
        if False:
            for i in range(10):
                print('nop')
        (train_prog, startup_prog) = (base.Program(), base.Program())
        (avg_cost, strategy) = self.net(train_prog, startup_prog)
        self.set_strategy(strategy, 'adaptive_localsgd')
        self.set_strategy(strategy, 'amp')
        self.optimizer(avg_cost, strategy, train_prog, startup_prog)
        ops = [op.type for op in avg_cost.block.ops]
        outs = [''.join(op.output('Out')) for op in avg_cost.block.ops if op.type == 'conditional_block']
        self.assertIn('conditional_block', ops)
        self.assertIn('@SNAPSHOT', ''.join(outs))
        self.assertIn('cast', ops)
        self.assertIn('check_finite_and_unscale', ops)
if __name__ == '__main__':
    unittest.main()