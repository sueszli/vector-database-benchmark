import unittest
from chainer import function_node
from chainer import testing

def dummy():
    if False:
        for i in range(10):
            print('nop')
    pass

class TestNoNumpyFunction(unittest.TestCase):

    def test_no_numpy_function(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(ValueError):
            testing.unary_math_function_unittest(dummy)

class DummyLinear(function_node.FunctionNode):

    @property
    def label(self):
        if False:
            i = 10
            return i + 15
        return 'dummy_linear'

    def forward(self, x):
        if False:
            return 10
        return (x[0],)

    def backward(self, indexes, gy):
        if False:
            i = 10
            return i + 15
        return (gy[0],)

def dummy_linear(x):
    if False:
        for i in range(10):
            print('nop')
    return DummyLinear().apply((x,))[0]

@testing.unary_math_function_unittest(dummy_linear, func_expected=lambda x, dtype: x)
class TestIsLinear(unittest.TestCase):
    pass
testing.run_module(__name__, __file__)