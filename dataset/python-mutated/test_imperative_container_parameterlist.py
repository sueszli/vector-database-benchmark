import unittest
import numpy as np
import paddle
from paddle import _legacy_C_ops, base

class MyLayer(paddle.nn.Layer):

    def __init__(self, num_stacked_param, use_base_api):
        if False:
            while True:
                i = 10
        super().__init__()
        self.params = self.paddle_imperative_ParameterList(num_stacked_param)

    def paddle_imperative_ParameterList(self, num_stacked_param):
        if False:
            for i in range(10):
                print('nop')
        return paddle.nn.ParameterList([paddle.create_parameter(shape=[2, 2], dtype='float32')] * num_stacked_param)

    def forward(self, x):
        if False:
            print('Hello World!')
        for (i, p) in enumerate(self.params):
            x = _legacy_C_ops.mul(x, p)
        return x

class TestImperativeContainerParameterList(unittest.TestCase):

    def paramter_list(self, use_base_api):
        if False:
            while True:
                i = 10
        data_np = np.random.uniform(-1, 1, [5, 2]).astype('float32')
        with base.dygraph.guard():
            x = base.dygraph.to_variable(data_np)
            num_stacked_param = 4
            model = MyLayer(num_stacked_param, use_base_api)
            self.assertEqual(len(model.params), num_stacked_param)
            res = model(x)
            self.assertListEqual(res.shape, [5, 2])
            loss = paddle.mean(res)
            loss.backward()
            model.params[num_stacked_param - 1] = paddle.create_parameter(shape=[2, 3], dtype='float32')
            res = model(x)
            self.assertListEqual(res.shape, [5, 3])
            model.params.append(paddle.create_parameter(shape=[3, 4], dtype='float32'))
            self.assertEqual(len(model.params), num_stacked_param + 1)
            res = model(x)
            self.assertListEqual(res.shape, [5, 4])
            loss = paddle.mean(res)
            loss.backward()

    def test_paramter_list(self):
        if False:
            while True:
                i = 10
        self.paramter_list(False)
        self.paramter_list(True)
if __name__ == '__main__':
    unittest.main()