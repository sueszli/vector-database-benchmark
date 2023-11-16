import unittest
import numpy as np
from dygraph_to_static_utils_new import Dy2StTestBase, test_ast_only, test_legacy_and_pir_exe_and_pir_api
import paddle
import paddle.nn.functional as F
from paddle.base import core

def apply_to_static(net, use_cinn):
    if False:
        print('Hello World!')
    build_strategy = paddle.static.BuildStrategy()
    build_strategy.build_cinn_pass = use_cinn
    return paddle.jit.to_static(net, build_strategy=build_strategy)

class PrimeNet(paddle.nn.Layer):

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__()
        self.fc = paddle.nn.Linear(4, 4)

    def forward(self, x):
        if False:
            print('Hello World!')
        y = self.fc(x)
        out = F.softmax(y)
        return out

class TestPrimForward(Dy2StTestBase):
    """
    This case only tests prim_forward + to_static + cinn. Thus we need to
    set this flag as False to avoid prim_backward.
    core.set_prim_backward(False)
    """

    def setUp(self):
        if False:
            while True:
                i = 10
        paddle.seed(2022)
        self.x = paddle.randn([2, 4])
        self.x.stop_gradient = False

    def train(self, use_prim):
        if False:
            while True:
                i = 10
        paddle.seed(2022)
        net = PrimeNet()
        sgd = paddle.optimizer.SGD(learning_rate=0.1, parameters=net.parameters())
        core._set_prim_forward_enabled(use_prim)
        if use_prim:
            net = apply_to_static(net, use_prim)
        res = []
        for _ in range(10):
            out = net(self.x)
            loss = paddle.mean(out)
            loss.backward()
            sgd.step()
            sgd.clear_grad()
            res.append(out.numpy())
        self.check_prim(net, use_prim)
        return res

    def check_prim(self, net, use_prim):
        if False:
            print('Hello World!')
        if not use_prim:
            return
        fwd_ops = [op.type for op in net.forward.get_concrete_program(self.x)[1].train_program.block(0).ops]
        self.assertTrue('softmax' not in fwd_ops)

    @test_ast_only
    def test_cinn_prim_forward(self):
        if False:
            for i in range(10):
                print('nop')
        dy_res = self.train(use_prim=False)
        cinn_res = self.train(use_prim=True)
        for i in range(len(dy_res)):
            np.testing.assert_allclose(cinn_res[i], dy_res[i], rtol=1e-07, atol=1e-07)

class TestPrimForwardAndBackward(Dy2StTestBase):
    """
    Test PrimeNet with @to_static + prim forward + prim backward + cinn v.s Dygraph
    """

    def setUp(self):
        if False:
            while True:
                i = 10
        paddle.seed(2022)
        self.x = paddle.randn([2, 4])
        self.x.stop_gradient = False

    def train(self, use_prim):
        if False:
            for i in range(10):
                print('nop')
        paddle.seed(2022)
        net = PrimeNet()
        sgd = paddle.optimizer.SGD(learning_rate=0.1, parameters=net.parameters())
        core._set_prim_all_enabled(use_prim)
        if use_prim:
            net = apply_to_static(net, use_prim)
        res = []
        for _ in range(10):
            out = net(self.x)
            loss = paddle.mean(out)
            loss.backward()
            sgd.step()
            sgd.clear_grad()
            res.append(out.numpy())
        self.check_prim(net, use_prim)
        return res

    def check_prim(self, net, use_prim):
        if False:
            return 10
        if not use_prim:
            return
        fwd_ops = [op.type for op in net.forward.get_concrete_program(self.x)[1].train_program.block(0).ops]
        all_ops = [op.type for op in net.forward.program_cache.last()[-1][-1].train_program.block(0).ops]
        self.assertTrue('softmax' not in fwd_ops)
        for op in all_ops:
            if op != 'matmul_v2_grad':
                self.assertTrue('_grad' not in op)

    @test_ast_only
    def test_cinn_prim(self):
        if False:
            return 10
        dy_res = self.train(use_prim=False)
        cinn_res = self.train(use_prim=True)
        for i in range(len(dy_res)):
            np.testing.assert_allclose(cinn_res[i], dy_res[i], rtol=1e-06, atol=1e-06)

class TestBackend(Dy2StTestBase):

    @test_legacy_and_pir_exe_and_pir_api
    def test_backend(self):
        if False:
            i = 10
            return i + 15
        x = paddle.randn([2, 4])
        out1 = self.forward(x, 'CINN')
        out2 = self.forward(x, None)
        np.testing.assert_allclose(out1, out2, rtol=1e-06)

    def forward(self, x, backend=None):
        if False:
            for i in range(10):
                print('nop')
        paddle.seed(2022)
        net = PrimeNet()
        net = paddle.jit.to_static(net, backend=backend)
        out = net(x)
        return out
if __name__ == '__main__':
    unittest.main()