import unittest
import numpy as np
from dygraph_to_static_utils_new import Dy2StTestBase, test_legacy_and_pir
import paddle

class Net(paddle.nn.Layer):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.relu = paddle.nn.functional.relu
        self.fc = paddle.nn.Linear(4, 4)

    def forward(self, x):
        if False:
            return 10
        y = paddle.full_like(x, 1.0)
        y.stop_gradient = False
        z = self.fc(x) * y
        out = y + z
        out = self.relu(out)
        return out

def apply_to_static(net, use_cinn):
    if False:
        i = 10
        return i + 15
    build_strategy = paddle.static.BuildStrategy()
    build_strategy.build_cinn_pass = use_cinn
    return paddle.jit.to_static(net, build_strategy=build_strategy)

class TestCINN(Dy2StTestBase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.x = paddle.randn([2, 4])
        self.x.stop_gradient = False

    def train(self, use_cinn):
        if False:
            print('Hello World!')
        paddle.seed(2022)
        net = Net()
        sgd = paddle.optimizer.SGD(learning_rate=0.1, parameters=net.parameters())
        if use_cinn:
            net = apply_to_static(net, use_cinn)
        res = []
        for step in range(10):
            out = net(self.x)
            loss = paddle.mean(out)
            loss.backward()
            sgd.step()
            sgd.clear_grad()
            res.append(out.numpy())
            if use_cinn and paddle.device.is_compiled_with_cinn():
                self.assertTrue(paddle.framework.core.is_run_with_cinn(), msg='The test was not running with CINN! Please check.')
            else:
                self.assertFalse(paddle.framework.core.is_run_with_cinn(), msg='The test should not running with CINN when the whl package was not compiled with CINN! Please check.')
        return res

    @test_legacy_and_pir
    def test_cinn(self):
        if False:
            i = 10
            return i + 15
        dy_res = self.train(use_cinn=False)
        cinn_res = self.train(use_cinn=True)
        for i in range(len(dy_res)):
            np.testing.assert_array_equal(cinn_res[i], dy_res[i])
if __name__ == '__main__':
    unittest.main()