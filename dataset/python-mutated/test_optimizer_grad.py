import unittest
from collections import defaultdict
import numpy as np
import paddle
from paddle import base
from paddle.base.backward import _append_grad_suffix_
paddle.enable_static()
np.random.seed(10)
SHAPE = [16, 10]

class SimpleNetWithCond:
    """
    Build net with conditional Block and useless layers.
    """

    def __init__(self, test_optimizer, param_lr=1.0, y_no_grad=False):
        if False:
            i = 10
            return i + 15
        self.optimizer = test_optimizer
        self.param_lr = param_lr
        self.shape = SHAPE
        self.y_no_grad = y_no_grad
        self._init_param()

    def _init_param(self):
        if False:
            i = 10
            return i + 15
        self.x = np.ones(self.shape).astype('float32')
        self.y = np.ones(self.shape).astype('float32') * 2.0
        self.z = np.ones(self.shape).astype('float32') * 3.0

    def _calc_gradient(self, cond_i):
        if False:
            while True:
                i = 10
        '\n        Calculate grads of params\n        '
        grads = []
        d_out_val = np.ones_like(self.x).astype('float32') / np.prod(self.shape)
        grads.append(d_out_val)
        if cond_i > 1:
            (y_grad_ratio, z_grad_ratio) = (0 if self.y_no_grad else 3, 1)
        else:
            (y_grad_ratio, z_grad_ratio) = (3, 0)
        if not self.y_no_grad:
            grads.append(d_out_val * y_grad_ratio)
        grads.append(d_out_val * z_grad_ratio)
        return grads

    def build_net(self, cond_i, use_bf16=False):
        if False:
            return 10
        '\n        pseudo code:\n            sum_xy = x + y\n            sub_yz = y - z\n            if i > 1:\n                internal = y + z\n                sum_cond = internal + z\n            else:\n                sum_cond = y + z\n            sum_all = sum_xy + sum_yz + sum_cond\n            mean_out = mean(sum_all)\n            optimizer.minimize(mean_out)\n        '
        param_x = paddle.create_parameter(dtype='float32', shape=self.shape, attr=base.ParamAttr(learning_rate=self.param_lr, name='param_x'), default_initializer=paddle.nn.initializer.Assign(self.x))
        param_y = paddle.create_parameter(dtype='float32', shape=self.shape, attr=base.ParamAttr(learning_rate=self.param_lr, name='param_y'), default_initializer=paddle.nn.initializer.Assign(self.y))
        param_z = paddle.create_parameter(dtype='float32', shape=self.shape, attr=base.ParamAttr(learning_rate=self.param_lr, name='param_z'), default_initializer=paddle.nn.initializer.Assign(self.z))
        sum_xy = paddle.add(param_x, param_y, name='sum_xy')
        sub_yz = paddle.subtract(param_y, param_z, name='sub_yz')
        useless = paddle.static.nn.fc(param_x, size=1, name='fc_useless')

        def cond_true():
            if False:
                for i in range(10):
                    print('nop')
            cond_yz = paddle.add(param_y, param_z, name='sum_cond_yz')
            param_y.stop_gradient = self.y_no_grad
            cond_res = paddle.add(cond_yz, param_z, name='sum_cond_true')
            cond_useless = paddle.multiply(param_x, param_y)
            return cond_res

        def cond_false():
            if False:
                return 10
            cond_res = paddle.add(param_y, param_z, name='sum_cond_false')
            cond_useless = paddle.multiply(param_z, param_z)
            return cond_res
        cond_i = paddle.assign(np.array([cond_i], dtype='float32'))
        sum_cond = paddle.static.nn.cond(cond_i > 1.0, cond_true, cond_false)
        sum_all = paddle.add_n([sum_xy, sub_yz, sum_cond])
        mean_out = paddle.mean(sum_all)
        if use_bf16:
            from paddle.static import amp
            self.optimizer = amp.bf16.decorate_bf16(self.optimizer, amp_lists=amp.bf16.AutoMixedPrecisionListsBF16(custom_fp32_list={'elementwise_add'}), use_bf16_guard=False, use_pure_bf16=True)
        self.optimizer.minimize(mean_out)
        fetch_list = ['param_x', 'param_z'] if self.y_no_grad else ['param_x', 'param_y', 'param_z']
        fetch_list += [_append_grad_suffix_(param) for param in fetch_list]
        return (fetch_list, self.optimizer)

class TestOptimizer(unittest.TestCase):
    """
    TestOptimizer BaseClass to be inherited to test other Optimizer.
    And only need to implement two functions:
        setUp(): to set config info of optimizer, including Optimizer and its hyper-parameter.
        _apply_gradient(): to implement the way of updating grad.
    """

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self._init_config()
        self.optimizer = paddle.optimizer.SGD(learning_rate=0.001)
        self.attr = {}

    def _init_config(self):
        if False:
            while True:
                i = 10
        self.NetClass = SimpleNetWithCond
        self.param_lr = [1.0, 2.0]
        self.cond_i = [0.1, 3]
        self.y_no_grad = [True, False]

    def test_optimizer(self):
        if False:
            print('Hello World!')
        self._check_grads()

    def _apply_gradient(self, param, grad, name):
        if False:
            while True:
                i = 10
        '\n        The way of updating grad in optimizer.(such as SGD)\n        This method should be override.\n        '
        return param - self.attr['lr'] * grad

    def _apply_optimize(self, net, grads):
        if False:
            for i in range(10):
                print('nop')
        '\n        apply to update all params in the net.\n        '
        net.x = self._apply_gradient(net.x, grads[0], 'x')
        if len(grads) == 2:
            net.z = self._apply_gradient(net.z, grads[1], 'z')
            res = [net.x, net.z]
        else:
            net.y = self._apply_gradient(net.y, grads[1], 'y')
            net.z = self._apply_gradient(net.z, grads[2], 'z')
            res = [net.x, net.y, net.z]
        return res

    def _init_param_attr(self):
        if False:
            for i in range(10):
                print('nop')
        self.param_attr = {}
        for key in ['x', 'y', 'z']:
            self.param_attr[key] = self.attr.copy()

    def _check_grads(self, use_bf16=False):
        if False:
            while True:
                i = 10
        '\n        main logic code to check the validity of apply_optimize.\n        '
        places = [base.CPUPlace()]
        if base.core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for place in places:
            for param_lr in self.param_lr:
                for cond_i in self.cond_i:
                    for y_no_grad in self.y_no_grad:
                        self.attr['lr'] = param_lr * self.optimizer._learning_rate
                        self._init_param_attr()
                        main_program = base.Program()
                        init_program = base.Program()
                        with base.program_guard(main_program, init_program):
                            self.optimizer._accumulators = defaultdict(lambda : {})
                            test_net = self.NetClass(self.optimizer, param_lr, y_no_grad)
                            (fetch_list, decorated_optimizer) = test_net.build_net(cond_i, use_bf16)
                            if use_bf16:
                                self.optimizer = decorated_optimizer
                            exe = base.Executor(place)
                            exe.run(init_program)
                            if use_bf16:
                                self.optimizer.amp_init(exe.place)
                            for batch_i in range(2):
                                res = exe.run(main_program, fetch_list=fetch_list)
                                gt_grads = test_net._calc_gradient(cond_i)
                                gt_params = self._apply_optimize(test_net, gt_grads)
                                param_grads = gt_params + gt_grads
                                for i in range(len(res)):
                                    np.testing.assert_allclose(res[i], param_grads[i])

@unittest.skipIf(not base.core.supports_bfloat16(), 'place does not support BF16 evaluation')
class TestSGDOptimizer(TestOptimizer):

    def test_optimizer_multiblock_except(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaisesRegex(ValueError, 'var param_y not in this block'):
            self._check_grads(use_bf16=True)
if __name__ == '__main__':
    unittest.main()