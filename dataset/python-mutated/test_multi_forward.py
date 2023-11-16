import unittest
from dygraph_to_static_utils_new import Dy2StTestBase, test_legacy_and_pir_exe_and_pir_api
import paddle

class MyLayer(paddle.nn.Layer):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.linear = paddle.nn.Linear(1, 1)

    def forward(self, x):
        if False:
            return 10
        return self.linear(x)

class TestBackward(Dy2StTestBase):

    @test_legacy_and_pir_exe_and_pir_api
    def test_order_0(self):
        if False:
            print('Hello World!')
        '\n        loss = 1 * w * 1 + 2 * w * 2\n        delta_w = 5\n        '
        model = paddle.jit.to_static(function=MyLayer(), input_spec=[paddle.static.InputSpec(shape=[None, None], dtype=paddle.float32)])
        model.clear_gradients()
        inp = paddle.ones([1, 1])
        out1 = model(inp * 1)
        out2 = model(inp * 2)
        loss = out2 * 2 + out1 * 1
        loss.backward()
        self.assertEqual(model.linear.weight.grad, 5)

    @test_legacy_and_pir_exe_and_pir_api
    def test_order_1(self):
        if False:
            return 10
        '\n        loss = 2 * w * 2  + 1 * w * 1\n        delta_w = 5\n        '
        model = paddle.jit.to_static(function=MyLayer(), input_spec=[paddle.static.InputSpec(shape=[None, None], dtype=paddle.float32)])
        model.clear_gradients()
        inp = paddle.ones([1, 1])
        out1 = model(inp * 1)
        out2 = model(inp * 2)
        loss = out1 * 1 + out2 * 2
        loss.backward()
        self.assertEqual(model.linear.weight.grad, 5)
if __name__ == '__main__':
    unittest.main()