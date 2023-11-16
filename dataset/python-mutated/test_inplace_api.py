import unittest
from test_case_base import TestCaseBase
import paddle
from paddle.jit.sot import symbolic_translate

def simple(x, y):
    if False:
        for i in range(10):
            print('nop')
    x[0] = 3.0
    z = [y]
    y[1] = 5.0
    return x[0] + x[1] + z[0][1] + y[0] + y[1]

def inplace_in_if(x, y, z):
    if False:
        i = 10
        return i + 15
    if z:
        x[0] = 3.0
        z = [y]
        y[1] = 5.0
        ret = x[0] + x[1] + z[0][1] + y[0] + y[1]
        return ret
    else:
        return None

def inplace_in_if_fallback(x, y, z):
    if False:
        print('Hello World!')
    if z > 0:
        x[0] = 3.0
        z = [y]
        y[1] = 5.0
        ret = x[0] + x[1] + z[0][1] + y[0] + y[1]
        return ret
    else:
        return None

def inplace_in_loop(x, y):
    if False:
        return 10
    ret = 0
    for i in range(10):
        x[0] = 1
        z = [y]
        y[1] = 2 * i + 1
        ret += x[0] + x[1] + z[0][1] + y[0] + y[1]
    return ret

def inplace_in_loop_fallback(x, y, it):
    if False:
        print('Hello World!')
    ret = 0
    for i in it:
        x[0] = 1
        z = [y]
        y[1] = 2 * i + 1
        ret += x[0] + x[1] + z[0][1] + y[0] + y[1]
    return ret

def inplace_case_0(x):
    if False:
        while True:
            i = 10
    x[:] = 1.0
    return x

def inplace_case_1(x):
    if False:
        return 10
    x[0][0, 0::2] = 1.0
    return x

def inplace_case_2(x):
    if False:
        return 10
    t = x[0]
    t[:, 0::2] = t[:, 0::2] * 0
    t[:, 1::2] = t[:, 1::2] + 2
    return x

class TestExecutor(TestCaseBase):

    def test_case(self):
        if False:
            return 10
        self.assert_results(inplace_case_0, paddle.randn((1, 4)))
        self.assert_results(inplace_case_1, [paddle.randn((1, 4))])
        self.assert_results(inplace_case_2, [paddle.randn((1, 4))])

    def test_backward(self):
        if False:
            return 10

        @symbolic_translate
        def func(x):
            if False:
                while True:
                    i = 10
            m = x * 2
            n = x * 3
            y = m
            y[:] = n
            return y
        x = paddle.ones((1, 4)) * 4
        x.stop_gradient = False
        y = func(x)
        y.sum().backward()
        assert (x.grad.numpy() == 3).all()

    def test_simple(self):
        if False:
            while True:
                i = 10
        self.assert_results(simple, paddle.to_tensor([1.0, 2.0]), paddle.to_tensor([3.0, 4.0]))

    def test_if(self):
        if False:
            while True:
                i = 10
        self.assert_results(inplace_in_if, paddle.to_tensor([1.0, 2.0]), paddle.to_tensor([3.0, 4.0]), True)
        self.assert_results(inplace_in_if_fallback, paddle.to_tensor([1.0, 2.0]), paddle.to_tensor([3.0, 4.0]), paddle.to_tensor(1))

    def test_loop(self):
        if False:
            return 10
        self.assert_results(inplace_in_loop, paddle.to_tensor([1.0, 2.0]), paddle.to_tensor([3.0, 4.0]))
        a = range(10)
        sym_output = symbolic_translate(inplace_in_loop_fallback)(paddle.to_tensor([1.0, 2.0]), paddle.to_tensor([3.0, 4.0]), iter(a))
        paddle_output = inplace_in_loop_fallback(paddle.to_tensor([1.0, 2.0]), paddle.to_tensor([3.0, 4.0]), iter(a))
        self.assert_nest_match(sym_output, paddle_output)
if __name__ == '__main__':
    unittest.main()