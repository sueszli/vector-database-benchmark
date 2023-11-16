import unittest
import numpy as np
from dygraph_to_static_utils_new import Dy2StTestBase, test_ast_only
import paddle
from paddle import base

@paddle.jit.to_static(full_graph=True)
def dygraph_decorated_func(x):
    if False:
        i = 10
        return i + 15
    x = base.dygraph.to_variable(x)
    if paddle.mean(x) > 0:
        x_v = x - 1
    else:
        x_v = x + 1
    return x_v

@paddle.jit.to_static(full_graph=True)
def jit_decorated_func(x):
    if False:
        return 10
    x = base.dygraph.to_variable(x)
    if paddle.mean(x) > 0:
        x_v = x - 1
    else:
        x_v = x + 1
    return x_v

@paddle.jit.to_static(full_graph=True)
def decorated_call_decorated(x):
    if False:
        return 10
    return jit_decorated_func(x)

class DoubleDecorated:

    @classmethod
    @paddle.jit.to_static(full_graph=True)
    def double_decorated_func1(self, x):
        if False:
            while True:
                i = 10
        return dygraph_decorated_func(x)

    @classmethod
    @paddle.jit.to_static(full_graph=True)
    def double_decorated_func2(self, x):
        if False:
            for i in range(10):
                print('nop')
        return jit_decorated_func(x)

class TestFullNameDecorator(Dy2StTestBase):

    @test_ast_only
    def test_run_success(self):
        if False:
            for i in range(10):
                print('nop')
        x = np.ones([1, 2]).astype('float32')
        answer = np.zeros([1, 2]).astype('float32')
        with base.dygraph.guard():
            np.testing.assert_allclose(dygraph_decorated_func(x).numpy(), answer, rtol=1e-05)
            np.testing.assert_allclose(jit_decorated_func(x).numpy(), answer, rtol=1e-05)
            np.testing.assert_allclose(decorated_call_decorated(x).numpy(), answer, rtol=1e-05)
            with self.assertRaises((NotImplementedError, TypeError)):
                DoubleDecorated().double_decorated_func1(x)
            with self.assertRaises((NotImplementedError, TypeError)):
                DoubleDecorated().double_decorated_func2(x)
if __name__ == '__main__':
    unittest.main()