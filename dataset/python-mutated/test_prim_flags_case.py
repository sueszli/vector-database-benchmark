import os
import unittest
import paddle
import paddle.nn.functional as F
from paddle.base import core

def apply_to_static(net, use_cinn):
    if False:
        i = 10
        return i + 15
    build_strategy = paddle.static.BuildStrategy()
    build_strategy.build_cinn_pass = use_cinn
    return paddle.jit.to_static(net, build_strategy=build_strategy, full_graph=True)

class PrimeNet(paddle.nn.Layer):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()

    def forward(self, x):
        if False:
            for i in range(10):
                print('nop')
        out = F.softmax(x)
        res = paddle.exp(out)
        return res

class TestPrimForwardAndBackward(unittest.TestCase):
    """
    Test PrimeNet with @to_static + prim forward + prim backward + cinn v.s Dygraph
    """

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        paddle.seed(2022)
        self.x = paddle.randn([2, 4])
        self.x.stop_gradient = False
        self.flag = None

    def reset_env_flag(self):
        if False:
            i = 10
            return i + 15
        os.environ['FLAGS_prim_backward'] = 'False'
        os.environ['FLAGS_prim_forward'] = 'False'
        if os.getenv('FLAGS_prim_all'):
            del os.environ['FLAGS_prim_all']
        core.check_and_set_prim_all_enabled()

    def train(self, use_cinn):
        if False:
            return 10
        net = PrimeNet()
        net = apply_to_static(net, use_cinn)
        out = net(self.x)
        loss = paddle.mean(out)
        loss.backward()
        self.check_prim(net)

    def check_prim(self, net):
        if False:
            i = 10
            return i + 15
        ops = [op.type for op in net.forward.program_cache.last()[-1][-1].train_program.block(0).ops]
        if self.flag in ['prim_all', 'cinn_prim_all']:
            self.assertTrue('softmax' not in ops)
            self.assertTrue('exp_grad' not in ops)
        elif self.flag in ['prim_forward', 'cinn_prim_forward']:
            self.assertTrue('softmax' not in ops)
            self.assertTrue('exp_grad' in ops)
        elif self.flag in ['prim_backward', 'cinn_prim_backward']:
            self.assertTrue('softmax' in ops)
            self.assertTrue('exp_grad' not in ops)
        elif self.flag == 'cinn':
            self.assertTrue('softmax' in ops)
            self.assertTrue('exp_grad' in ops)
        else:
            raise TypeError

    def test_cinn_prim_all(self):
        if False:
            return 10
        'cinn + prim forward + prim backward'
        self.reset_env_flag()
        os.environ['FLAGS_prim_all'] = 'True'
        self.flag = 'cinn_prim_all'
        _ = self.train(use_cinn=True)

    def test_prim_all(self):
        if False:
            for i in range(10):
                print('nop')
        'prim forward + prim backward'
        self.reset_env_flag()
        os.environ['FLAGS_prim_all'] = 'True'
        self.flag = 'prim_all'
        _ = self.train(use_cinn=False)

    def test_cinn_prim_forward(self):
        if False:
            while True:
                i = 10
        'cinn + prim forward'
        self.reset_env_flag()
        os.environ['FLAGS_prim_forward'] = 'True'
        self.flag = 'cinn_prim_forward'
        _ = self.train(use_cinn=True)

    def test_prim_forward(self):
        if False:
            i = 10
            return i + 15
        'only prim forward'
        self.reset_env_flag()
        os.environ['FLAGS_prim_forward'] = 'True'
        self.flag = 'prim_forward'
        _ = self.train(use_cinn=False)

    def test_cinn_prim_backward(self):
        if False:
            return 10
        'cinn + prim_backward'
        self.reset_env_flag()
        os.environ['FLAGS_prim_backward'] = 'True'
        self.flag = 'cinn_prim_backward'
        _ = self.train(use_cinn=True)

    def test_prim_backward(self):
        if False:
            for i in range(10):
                print('nop')
        'only prim backward'
        self.reset_env_flag()
        os.environ['FLAGS_prim_backward'] = 'True'
        self.flag = 'prim_backward'
        _ = self.train(use_cinn=False)

    def test_cinn(self):
        if False:
            return 10
        'only cinn'
        self.reset_env_flag()
        self.flag = 'cinn'
        _ = self.train(use_cinn=True)
if __name__ == '__main__':
    unittest.main()