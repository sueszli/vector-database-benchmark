import unittest
import numpy as np
from dygraph_to_static_utils_new import Dy2StTestBase, test_legacy_and_pir_exe_and_pir_api
import paddle
SEED = 2020

class Pool2D(paddle.nn.Layer):

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__()
        self.pool2d = paddle.nn.AvgPool2D(kernel_size=2, stride=1)

    def forward(self, x):
        if False:
            print('Hello World!')

        def get_result(x):
            if False:
                i = 10
                return i + 15
            return self.pool2d(x)
        pre = get_result(x)
        return pre

class Linear(paddle.nn.Layer):

    def __init__(self, input_dim=10, output_dim=5):
        if False:
            return 10
        super().__init__()
        self.fc = paddle.nn.Linear(input_dim, output_dim, weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(value=0.99)), bias_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(value=0.5)))
        self.act = paddle.nn.ReLU()

    def forward(self, x):
        if False:
            return 10
        pre = self.fc(x)
        pre = self.act(pre)
        loss = paddle.mean(pre)
        return (pre, loss)

class TestPool2D(Dy2StTestBase):

    def setUp(self):
        if False:
            return 10
        self.dygraph_class = Pool2D
        self.data = np.random.random((1, 2, 4, 4)).astype('float32')

    def train(self, to_static=False):
        if False:
            return 10
        paddle.jit.enable_to_static(to_static)
        dy_layer = paddle.jit.to_static(self.dygraph_class())
        x = paddle.to_tensor(self.data)
        prediction = dy_layer(x)
        if isinstance(prediction, (list, tuple)):
            prediction = prediction[0]
        return prediction.numpy()

    def train_static(self):
        if False:
            print('Hello World!')
        return self.train(to_static=True)

    def train_dygraph(self):
        if False:
            print('Hello World!')
        return self.train(to_static=False)

    @test_legacy_and_pir_exe_and_pir_api
    def test_to_static(self):
        if False:
            for i in range(10):
                print('nop')
        dygraph_res = self.train_dygraph()
        static_res = self.train_static()
        np.testing.assert_allclose(dygraph_res, static_res, rtol=1e-05)

class TestLinear(TestPool2D):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.dygraph_class = Linear
        self.data = np.random.random((4, 10)).astype('float32')
if __name__ == '__main__':
    unittest.main()