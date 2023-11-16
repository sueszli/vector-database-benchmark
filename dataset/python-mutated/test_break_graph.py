import unittest
import numpy as np
from test_case_base import TestCaseBase
import paddle
from paddle.jit.sot.utils.paddle_api_config import add_break_graph_apis

def ifelse_func(x, y):
    if False:
        while True:
            i = 10
    if x > 0:
        y = y + 1
    else:
        y = y + 2
    return y

class TestIfElse(TestCaseBase):

    def test_simple(self):
        if False:
            i = 10
            return i + 15
        x = paddle.to_tensor([1.0])
        y = paddle.to_tensor([2.0])
        self.assert_results(ifelse_func, x, y)

def multi_output(x: paddle.Tensor):
    if False:
        i = 10
        return i + 15
    m = x + 1
    if x > 0:
        return m
    else:
        return 2 * m

class TestExecutor(TestCaseBase):

    def test_simple(self):
        if False:
            i = 10
            return i + 15
        x = paddle.to_tensor(2)
        self.assert_results(multi_output, x)
        x = paddle.to_tensor(-2)
        self.assert_results(multi_output, x)

def print_break_graph(x, y):
    if False:
        return 10
    z = x + y
    print(x, z)
    out = y * z * 2
    return out

class TestPrint(TestCaseBase):

    def test_simple(self):
        if False:
            for i in range(10):
                print('nop')
        x = paddle.to_tensor(2)
        y = paddle.to_tensor(3)
        self.assert_results(print_break_graph, x, y)

def to_tensor_break_graph(x, y):
    if False:
        for i in range(10):
            print('nop')
    z = x + y
    out = y * paddle.to_tensor(2) * z
    return out

class TestToTensor(TestCaseBase):

    def test_simple(self):
        if False:
            i = 10
            return i + 15
        add_break_graph_apis([paddle.to_tensor])
        x = paddle.to_tensor(2)
        y = paddle.to_tensor(3)
        self.assert_results(to_tensor_break_graph, x, y)

def tensor_clear_gradient(x):
    if False:
        print('Hello World!')
    x = paddle.to_tensor(x)
    x.clear_gradient()
    return x

class TestBreakGraphInResumeFn(TestCaseBase):

    def test_simple(self):
        if False:
            for i in range(10):
                print('nop')
        x = paddle.to_tensor(2)
        self.assert_results(tensor_clear_gradient, x)

def inner_fn(a, b, c, d):
    if False:
        while True:
            i = 10
    return a + b * c - d

def multi_stack_args(a, b, c):
    if False:
        for i in range(10):
            print('nop')
    out = inner_fn(a, b, c, paddle.to_tensor(4))
    return out

class TestMultiStackArgs(TestCaseBase):

    def test_simple(self):
        if False:
            i = 10
            return i + 15
        a = paddle.to_tensor(1)
        b = paddle.to_tensor(2)
        c = paddle.to_tensor(3)
        self.assert_results(multi_stack_args, a, b, c)

def break_graph_in_call_method(x):
    if False:
        for i in range(10):
            print('nop')
    out = paddle.nn.functional.relu(paddle.to_tensor([4.0]))
    return x + out

def numpy_break_graph():
    if False:
        i = 10
        return i + 15
    a = paddle.to_tensor([1, 2])
    b = np.sum(a.numpy())
    print(b)
    return b

class TestBreakGraphInCallMethod(TestCaseBase):

    def test_simple(self):
        if False:
            return 10
        x = paddle.to_tensor([1.0])
        break_graph_in_call_method(x)
        x = paddle.to_tensor([2.0])
        break_graph_in_call_method(x)
        x = paddle.to_tensor([3.0])
        self.assert_results(break_graph_in_call_method, x)

    def test_numpy(self):
        if False:
            print('Hello World!')
        self.assert_results(numpy_break_graph)

def test_break_graph_repeat(x):
    if False:
        i = 10
        return i + 15
    out = paddle.to_tensor(paddle.to_tensor(paddle.to_tensor(paddle.to_tensor([1.0]))))
    return x + out

class TestBreakGraphRepeat(TestCaseBase):

    def test_simple(self):
        if False:
            print('Hello World!')
        x = paddle.to_tensor([1.0])
        test_break_graph_repeat(x)
        x = paddle.to_tensor([2.0])
        test_break_graph_repeat(x)
        x = paddle.to_tensor([3.0])
        self.assert_results(test_break_graph_repeat, x)

def break_graph_resume_pass_null(x, y):
    if False:
        return 10
    return paddle.add(x, y[0:50] if y is not None else None)

class TestBreakGraphResumePassNull(TestCaseBase):

    def test_break_graph_resume_pass_null(self):
        if False:
            for i in range(10):
                print('nop')
        x = paddle.rand([50, 50], dtype=paddle.float32)
        y = paddle.rand([100, 50], dtype=paddle.float32)
        self.assert_results(break_graph_resume_pass_null, x, y)
if __name__ == '__main__':
    unittest.main()