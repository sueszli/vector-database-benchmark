import unittest
import numpy as np
from test_imperative_base import new_program_scope
import paddle
from paddle import base

class MLP(paddle.nn.Layer):

    def __init__(self, input_size):
        if False:
            print('Hello World!')
        super().__init__()
        self._linear1 = paddle.nn.Linear(input_size, 3, weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(value=0.1)), bias_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(value=0.1)))
        self._linear2 = paddle.nn.Linear(3, 4, weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(value=0.1)), bias_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(value=0.1)))

    def forward(self, inputs):
        if False:
            print('Hello World!')
        x = self._linear1(inputs)
        x = self._linear2(x)
        x = paddle.sum(x)
        return x

class TestDygraphFramework(unittest.TestCase):

    def test_dygraph_backward(self):
        if False:
            return 10
        with new_program_scope():
            mlp = MLP(input_size=2)
            var_inp = paddle.static.data('input', shape=[2, 2], dtype='float32')
            out = mlp(var_inp)
            try:
                out.backward()
                raise AssertionError('backward should not be usable in static graph mode')
            except AssertionError as e:
                self.assertTrue(e is not None)

    def test_dygraph_to_string(self):
        if False:
            return 10
        np_inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        with base.dygraph.guard():
            var_inp = base.dygraph.to_variable(np_inp)
            print(str(var_inp))