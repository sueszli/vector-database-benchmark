import unittest
import numpy as np
from dygraph_to_static_utils_new import Dy2StTestBase, test_ast_only
import paddle
import paddle.nn.functional as F
from paddle.base import core
TOLERANCE = {'float16': {'rtol': 0.01, 'atol': 0.01}, 'float32': {'rtol': 1e-05, 'atol': 1e-05}, 'float64': {'rtol': 1e-13, 'atol': 1e-13}}

def generate_data(dtype='float32'):
    if False:
        for i in range(10):
            print('nop')
    np_data1 = np.random.random([2, 64]).astype(dtype)
    np_data2 = np.random.random([64]).astype(dtype)
    np_data3 = np.random.random([64]).astype(dtype)
    return (np_data1, np_data2, np_data3)

def apply_to_static(net, use_cinn):
    if False:
        return 10
    build_strategy = paddle.static.BuildStrategy()
    build_strategy.build_cinn_pass = use_cinn
    return paddle.jit.to_static(net, build_strategy=build_strategy)

class PrimeNet(paddle.nn.Layer):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.fc = paddle.nn.Linear(64, 64)

    def forward(self, x, w, b):
        if False:
            while True:
                i = 10
        n_shape = x.shape[1:]
        out = F.layer_norm(x, n_shape, w, b)
        return out[0]

class TestPrimForward(Dy2StTestBase):
    """
    This case only tests prim_forward + to_static + cinn. Thus we need to
    set this flag as False to avoid prim_backward.
    core.set_prim_backward(False)
    """

    def setUp(self):
        if False:
            print('Hello World!')
        self.x = None
        self.w = None
        self.b = None
        self.dtypes = ['float16', 'float32']

    def train(self, use_prim):
        if False:
            for i in range(10):
                print('nop')
        net = PrimeNet()
        sgd = paddle.optimizer.SGD(learning_rate=0.1, parameters=net.parameters())
        core._set_prim_forward_enabled(use_prim)
        core._add_skip_comp_ops('sqrt')
        if use_prim:
            net = apply_to_static(net, use_prim)
        out = net(self.x, self.w, self.b)
        loss = paddle.mean(out)
        loss.backward()
        sgd.step()
        sgd.clear_grad()
        self.check_prim(net, use_prim)
        return out.numpy()

    def check_prim(self, net, use_prim):
        if False:
            print('Hello World!')
        if not use_prim:
            return
        fwd_ops = [op.type for op in net.forward.get_concrete_program(self.x, self.w, self.b)[1].train_program.block(0).ops]
        self.assertTrue('layer_norm' not in fwd_ops)

    @test_ast_only
    def test_cinn_prim_forward(self):
        if False:
            i = 10
            return i + 15
        for dtype in self.dtypes:
            if paddle.device.get_device() == 'cpu':
                print('need pass this case')
                continue
            (x_n, w_n, b_n) = generate_data(dtype)
            self.x = paddle.to_tensor(x_n)
            self.w = paddle.to_tensor(w_n)
            self.b = paddle.to_tensor(b_n)
            self.x.stop_gradient = False
            dy_res = self.train(use_prim=False)
            cinn_res = self.train(use_prim=True)
            np.testing.assert_allclose(cinn_res, dy_res, rtol=TOLERANCE[dtype]['rtol'], atol=TOLERANCE[dtype]['atol'])

class TestPrimForwardAndBackward(Dy2StTestBase):
    """
    Test PrimeNet with @to_static + prim forward + prim backward + cinn v.s Dygraph
    """

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.x = None
        self.w = None
        self.b = None
        self.dtypes = ['float16', 'float32']

    def train(self, use_prim):
        if False:
            print('Hello World!')
        net = PrimeNet()
        sgd = paddle.optimizer.SGD(learning_rate=0.1, parameters=net.parameters())
        core._set_prim_all_enabled(use_prim)
        core._add_skip_comp_ops('sqrt')
        if use_prim:
            net = apply_to_static(net, use_prim)
        out = net(self.x, self.w, self.b)
        loss = paddle.mean(out)
        loss.backward()
        sgd.step()
        sgd.clear_grad()
        self.check_prim(net, use_prim)
        return out.numpy()

    def check_prim(self, net, use_prim):
        if False:
            return 10
        if not use_prim:
            return
        fwd_ops = [op.type for op in net.forward.get_concrete_program(self.x, self.w, self.b)[1].train_program.block(0).ops]
        self.assertTrue('layer_norm' not in fwd_ops)

    @test_ast_only
    def test_cinn_prim(self):
        if False:
            while True:
                i = 10
        for dtype in self.dtypes:
            if paddle.device.get_device() == 'cpu':
                print('need pass this case')
                continue
            (x_n, w_n, b_n) = generate_data(dtype)
            self.x = paddle.to_tensor(x_n)
            self.w = paddle.to_tensor(w_n)
            self.b = paddle.to_tensor(b_n)
            self.x.stop_gradient = False
            dy_res = self.train(use_prim=False)
            cinn_res = self.train(use_prim=True)
            np.testing.assert_allclose(cinn_res, dy_res, rtol=TOLERANCE[dtype]['rtol'], atol=TOLERANCE[dtype]['atol'])
if __name__ == '__main__':
    unittest.main()