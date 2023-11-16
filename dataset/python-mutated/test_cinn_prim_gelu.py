import unittest
import numpy as np
from dygraph_to_static_utils_new import Dy2StTestBase, test_ast_only
import paddle
import paddle.nn.functional as F
from paddle.base import core
TOLERANCE = {'float16': {'rtol': 0.001, 'atol': 0.001}, 'float32': {'rtol': 1e-06, 'atol': 1e-06}, 'float64': {'rtol': 1e-15, 'atol': 1e-15}}
approximate_conds = [True, False]

def apply_to_static(net, use_cinn):
    if False:
        print('Hello World!')
    build_strategy = paddle.static.BuildStrategy()
    build_strategy.build_cinn_pass = use_cinn
    return paddle.jit.to_static(net, build_strategy=build_strategy)

def generate_data(shape, dtype='float32'):
    if False:
        i = 10
        return i + 15
    np_data = np.random.random(shape).astype(dtype)
    return np_data

class PrimeNet(paddle.nn.Layer):

    def __init__(self, approximate):
        if False:
            return 10
        super().__init__()
        self.fc = paddle.nn.Linear(4, 4)
        self.approximate = approximate

    def forward(self, x):
        if False:
            i = 10
            return i + 15
        out = F.gelu(x, approximate=self.approximate)
        return out

class TestPrimForwardAndBackward(Dy2StTestBase):
    """
    Test PrimeNet with @to_static + prim forward + prim backward + cinn v.s Dygraph
    """

    def setUp(self):
        if False:
            i = 10
            return i + 15
        paddle.seed(2022)
        self.shapes = [[2, 4], [64, 16, 4]]
        self.dtypes = ['float16', 'float32']

    def train(self, use_prim, data):
        if False:
            print('Hello World!')
        for approximate in approximate_conds:
            return self._train(use_prim, approximate, data)

    def _train(self, use_prim, approximate, data):
        if False:
            i = 10
            return i + 15
        paddle.seed(2022)
        net = PrimeNet(approximate)
        sgd = paddle.optimizer.SGD(learning_rate=0.1, parameters=net.parameters())
        core._set_prim_all_enabled(use_prim)
        if use_prim:
            net = apply_to_static(net, use_prim)
        res = []
        self.x = data
        for _ in range(10):
            out = net(data)
            loss = paddle.mean(out)
            loss.backward()
            sgd.step()
            sgd.clear_grad()
            res.append(out.numpy())
        self.check_prim(net, use_prim)
        return res

    def check_prim(self, net, use_prim):
        if False:
            i = 10
            return i + 15
        if not use_prim:
            return
        fwd_ops = [op.type for op in net.forward.get_concrete_program(self.x)[1].train_program.block(0).ops]
        self.assertTrue('gelu' not in fwd_ops)

    @test_ast_only
    def test_cinn_prim(self):
        if False:
            return 10
        for shape in self.shapes:
            for dtype in self.dtypes:
                if paddle.device.get_device() == 'cpu' and dtype == 'float16':
                    print('need pass this case')
                    continue
                data = generate_data(shape, dtype)
                data_t = paddle.to_tensor(data)
                data_t.stop_gradient = False
                dy_res = self.train(use_prim=False, data=data_t)
                cinn_res = self.train(use_prim=True, data=data_t)
                for i in range(len(dy_res)):
                    np.testing.assert_allclose(cinn_res[i], dy_res[i], rtol=TOLERANCE[dtype]['rtol'], atol=TOLERANCE[dtype]['atol'])
if __name__ == '__main__':
    unittest.main()