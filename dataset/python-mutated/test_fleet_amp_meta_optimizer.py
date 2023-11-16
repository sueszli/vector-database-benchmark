import unittest
from fleet_meta_optimizer_base import TestFleetMetaOptimizer
import paddle
from paddle import base
from paddle.distributed import fleet
from paddle.distributed.fleet.base import role_maker
from paddle.distributed.fleet.meta_optimizers import AMPOptimizer
paddle.enable_static()

class TestFleetAMPOptimizer(TestFleetMetaOptimizer):

    def test_amp_optimizer_backward(self):
        if False:
            while True:
                i = 10
        'test amp optimizer backward'
        (train_prog, startup_prog) = (base.Program(), base.Program())
        (avg_cost, strategy) = self.net(train_prog, startup_prog)
        opt = paddle.optimizer.Momentum(learning_rate=0.001, momentum=0.9)
        opt = AMPOptimizer(opt)
        self.set_strategy(strategy, 'amp')
        role = role_maker.PaddleCloudRoleMaker(is_collective=True)
        opt._set_basic_info(avg_cost, role, opt, strategy)
        params_grads = opt.backward(avg_cost, startup_prog)
        ops = [op.type for op in avg_cost.block.ops]
        self.assertIn('cast', ops)
        self.assertNotIn('check_finite_and_unscale', ops)

    def test_amp_optimizer_backward_gradients(self):
        if False:
            return 10
        'test amp optimizer backward + gradients'
        (train_prog, startup_prog) = (base.Program(), base.Program())
        (avg_cost, strategy) = self.net(train_prog, startup_prog)
        opt = paddle.optimizer.Momentum(learning_rate=0.001, momentum=0.9)
        opt = AMPOptimizer(opt)
        self.set_strategy(strategy, 'amp')
        role = role_maker.PaddleCloudRoleMaker(is_collective=True)
        opt._set_basic_info(avg_cost, role, opt, strategy)
        params_grads = opt.backward(avg_cost, startup_prog)
        with base.program_guard(train_prog, startup_prog):
            opt.apply_gradients(params_grads)
        ops = [op.type for op in avg_cost.block.ops]
        self.assertIn('cast', ops)
        self.assertIn('check_finite_and_unscale', ops)

    def test_amp_optimizer_backward_optimize(self):
        if False:
            i = 10
            return i + 15
        'test amp optimizer backward + optimizer'
        (train_prog, startup_prog) = (base.Program(), base.Program())
        (avg_cost, strategy) = self.net(train_prog, startup_prog)
        opt = paddle.optimizer.Momentum(learning_rate=0.001, momentum=0.9)
        opt = AMPOptimizer(opt)
        self.set_strategy(strategy, 'amp')
        role = role_maker.PaddleCloudRoleMaker(is_collective=True)
        opt._set_basic_info(avg_cost, role, opt, strategy)
        params_grads = opt.backward(avg_cost, startup_prog)
        opt.apply_optimize(avg_cost, startup_prog, params_grads)
        ops = [op.type for op in avg_cost.block.ops]
        self.assertIn('cast', ops)
        self.assertIn('check_finite_and_unscale', ops)

    def test_amp_optimizer(self):
        if False:
            return 10
        'test amp'
        (train_prog, startup_prog) = (base.Program(), base.Program())
        (avg_cost, strategy) = self.net(train_prog, startup_prog)
        self.set_strategy(strategy, 'amp')
        self.optimizer(avg_cost, strategy, train_prog, startup_prog)
        ops = [op.type for op in avg_cost.block.ops]
        self.assertIn('cast', ops)
        self.assertIn('check_finite_and_unscale', ops)

    def test_pure_fp16_optimizer(self):
        if False:
            i = 10
            return i + 15
        'test pure fp16'
        (train_prog, startup_prog) = (base.Program(), base.Program())
        (avg_cost, strategy) = self.net(train_prog, startup_prog)
        self.set_strategy(strategy, 'pure_fp16')
        self.optimizer(avg_cost, strategy, train_prog, startup_prog)
        params = train_prog.all_parameters()
        for param in train_prog.all_parameters():
            self.assertEqual(param.dtype, base.core.VarDesc.VarType.FP16)
        ops = [op.type for op in avg_cost.block.ops]
        self.assertIn('cast', ops)
        self.assertIn('check_finite_and_unscale', ops)

    def test_amp_distributed_optimizer(self):
        if False:
            i = 10
            return i + 15
        'test amp when distributed'
        (train_prog, startup_prog) = (base.Program(), base.Program())
        (avg_cost, strategy) = self.net(train_prog, startup_prog)
        self.set_strategy(strategy, 'amp')
        self.optimizer(avg_cost, strategy, train_prog, startup_prog)
        ops = [op.type for op in avg_cost.block.ops]
        self.assertIn('cast', ops)
        self.assertIn('check_finite_and_unscale', ops)
        check_count = 0
        for name in ops:
            if name == 'check_finite_and_unscale':
                check_count += 1
        self.assertEqual(check_count, len(train_prog.all_parameters()))

    def test_amp_recompute_optimizer(self):
        if False:
            for i in range(10):
                print('nop')
        'test amp + recompute'
        (train_prog, startup_prog) = (base.Program(), base.Program())
        (avg_cost, strategy) = self.net(train_prog, startup_prog)
        self.set_strategy(strategy, 'amp')
        self.set_strategy(strategy, 'recompute')
        self.optimizer(avg_cost, strategy, train_prog, startup_prog)
        strategy = fleet._final_strategy()
        ops = [op.type for op in avg_cost.block.ops]
        outs = [op.output('Out')[0] for op in avg_cost.block.ops if op.type == 'mul']
        self.assertIn('cast', ops)
        self.assertIn('check_finite_and_unscale', ops)
        self.assertIn('subprog', ''.join(outs))

    def test_amp_recompute_lars_optimizer(self):
        if False:
            return 10
        'test amp + recompute'
        (train_prog, startup_prog) = (base.Program(), base.Program())
        (avg_cost, strategy) = self.net(train_prog, startup_prog)
        self.set_strategy(strategy, 'amp')
        self.set_strategy(strategy, 'recompute')
        self.set_strategy(strategy, 'lars')
        self.optimizer(avg_cost, strategy, train_prog, startup_prog)
        strategy = fleet._final_strategy()
        ops = [op.type for op in avg_cost.block.ops]
        outs = [op.output('Out')[0] for op in avg_cost.block.ops if op.type == 'mul']
        self.assertIn('cast', ops)
        self.assertIn('check_finite_and_unscale', ops)
        self.assertIn('subprog', ''.join(outs))
        self.assertIn('lars_momentum', ops)

    def test_amp_recompute_lamb_optimizer(self):
        if False:
            print('Hello World!')
        (train_prog, startup_prog) = (base.Program(), base.Program())
        (avg_cost, strategy) = self.net(train_prog, startup_prog)
        self.set_strategy(strategy, 'amp')
        self.set_strategy(strategy, 'recompute')
        self.set_strategy(strategy, 'lamb')
        self.optimizer(avg_cost, strategy, train_prog, startup_prog, 'adam')
        applied_meta_list = fleet._get_applied_meta_list()
        applied_graph_list = fleet._get_applied_graph_list()
        print(applied_meta_list, applied_graph_list)
        self.assertEqual(len(applied_meta_list), 4)
        ops = [op.type for op in avg_cost.block.ops]
        outs = [op.output('Out')[0] for op in avg_cost.block.ops if op.type == 'mul']
        self.assertIn('cast', ops)
        self.assertIn('check_finite_and_unscale', ops)
        self.assertIn('subprog', ''.join(outs))
        self.assertIn('lamb', ops)
if __name__ == '__main__':
    unittest.main()