import unittest
from fleet_meta_optimizer_base import TestFleetMetaOptimizer
import paddle
paddle.enable_static()

class TestFleetGradientMergeMetaOptimizer(TestFleetMetaOptimizer):

    def test_gradient_merge_optimizer(self):
        if False:
            i = 10
            return i + 15
        (train_prog, startup_prog) = (paddle.base.Program(), paddle.base.Program())
        (avg_cost, strategy) = self.net(train_prog, startup_prog)
        self.set_strategy(strategy, 'gradient_merge')
        self.optimizer(avg_cost, strategy, train_prog, startup_prog)
        vars = [x.name for x in train_prog.list_vars()]
        self.assertIn('@GradientMerge', ''.join(vars))

    def test_recom_gm_optimizer(self):
        if False:
            i = 10
            return i + 15
        (train_prog, startup_prog) = (paddle.base.Program(), paddle.base.Program())
        (avg_cost, strategy) = self.net(train_prog, startup_prog)
        self.set_strategy(strategy, 'gradient_merge')
        self.set_strategy(strategy, 'recompute')
        self.optimizer(avg_cost, strategy, train_prog, startup_prog)
        vars = [x.name for x in train_prog.list_vars()]
        self.assertIn('@GradientMerge', ''.join(vars))
        self.assertIn('subprog', ''.join(vars))

    def test_gm_amp_optimizer(self):
        if False:
            while True:
                i = 10
        (train_prog, startup_prog) = (paddle.base.Program(), paddle.base.Program())
        (avg_cost, strategy) = self.net(train_prog, startup_prog)
        self.set_strategy(strategy, 'gradient_merge')
        self.set_strategy(strategy, 'amp')
        self.optimizer(avg_cost, strategy, train_prog, startup_prog)
        vars = [x.name for x in train_prog.list_vars()]
        self.assertIn('@GradientMerge', ''.join(vars))
        self.assertIn('cast', ''.join(vars))

    def test_gm_pure_fp16_optimizer(self):
        if False:
            for i in range(10):
                print('nop')
        (train_prog, startup_prog) = (paddle.base.Program(), paddle.base.Program())
        (avg_cost, strategy) = self.net(train_prog, startup_prog)
        self.set_strategy(strategy, 'gradient_merge')
        self.set_strategy(strategy, 'pure_fp16')
        self.optimizer(avg_cost, strategy, train_prog, startup_prog)
        print(train_prog)
        params = train_prog.all_parameters()
        for param in train_prog.all_parameters():
            self.assertEqual(param.dtype, paddle.base.core.VarDesc.VarType.FP16)
        vars = [x.name for x in train_prog.list_vars()]
        self.assertIn('@GradientMerge', ''.join(vars))
        self.assertIn('cast', ''.join(vars))
if __name__ == '__main__':
    unittest.main()