import unittest
from test_case_base import TestCaseBase
import paddle
from paddle import nn
from paddle.jit.sot import symbolic_translate

class A:

    def __init__(self, vals):
        if False:
            while True:
                i = 10
        vals.append(1)

def foo(x, y):
    if False:
        for i in range(10):
            print('nop')
    out = nn.Softmax()(paddle.to_tensor([x, y], dtype='float32'))
    return out

def foo2(x, y):
    if False:
        for i in range(10):
            print('nop')
    t = nn.Softmax()
    out1 = t(paddle.to_tensor([x, y], dtype='float32'))
    out2 = t(paddle.to_tensor([x, y], dtype='float32'))
    return out1 + out2

def error_foo(x):
    if False:
        return 10
    t = nn.Linear(10, 10)
    return t(x)

def bar(x):
    if False:
        while True:
            i = 10
    a = A(x)
    t = paddle.to_tensor(x)
    return t.mean()

class TestInit(TestCaseBase):

    def test_init_paddle_layer(self):
        if False:
            print('Hello World!')
        self.assert_results(foo, 1, 2)
        self.assert_results(foo2, 1, 2)

    def test_init_python_object(self):
        if False:
            i = 10
            return i + 15
        sot_output = symbolic_translate(bar)([1.0, 2.0])
        dyn_output = bar([1.0, 2.0])
        self.assert_nest_match(sot_output, dyn_output)

    def test_error(self):
        if False:
            i = 10
            return i + 15

        def run():
            if False:
                for i in range(10):
                    print('nop')
            inputs = paddle.randn((10, 10))
            symbolic_translate(error_foo)(inputs)
        self.assertRaises(paddle.jit.sot.utils.exceptions.InnerError, run)
if __name__ == '__main__':
    unittest.main()