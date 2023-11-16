import unittest
from unittest import TestCase
import numpy as np
import paddle
from paddle import base
from paddle.base.wrapped_decorator import wrap_decorator

def _dygraph_guard_(func):
    if False:
        i = 10
        return i + 15

    def __impl__(*args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if base.in_dygraph_mode():
            return func(*args, **kwargs)
        else:
            with base.dygraph.guard():
                return func(*args, **kwargs)
    return __impl__
dygraph_guard = wrap_decorator(_dygraph_guard_)

class TestDygraphClearGradient(TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.input_shape = [10, 2]

    @dygraph_guard
    def test_tensor_method_clear_gradient_case1(self):
        if False:
            while True:
                i = 10
        input = paddle.uniform(self.input_shape)
        linear = paddle.nn.Linear(2, 3)
        out = linear(input)
        out.backward()
        if not base.framework.in_dygraph_mode():
            linear.weight.clear_gradient()
        else:
            linear.weight._zero_grads()
        gradient_actual = linear.weight.grad
        gradient_expected = np.zeros([2, 3]).astype('float64')
        self.assertTrue(np.allclose(gradient_actual.numpy(), gradient_expected))

    @dygraph_guard
    def test_tensor_method_clear_gradient_case2(self):
        if False:
            i = 10
            return i + 15
        input = paddle.uniform(self.input_shape)
        linear = paddle.nn.Linear(2, 3)
        out = linear(input)
        out.backward()
        linear.weight.clear_gradient(False)
        if not base.framework.in_dygraph_mode():
            self.assertTrue(linear.weight._is_gradient_set_empty())
        else:
            self.assertIsNone(linear.weight.grad)
        if not base.framework.in_dygraph_mode():
            linear.weight._gradient_set_empty(False)
            self.assertFalse(linear.weight._is_gradient_set_empty())
        gradient_actual = linear.weight.grad
        print(gradient_actual)
        self.assertTrue(np.empty(gradient_actual))
if __name__ == '__main__':
    unittest.main()