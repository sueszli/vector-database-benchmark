import unittest
from test_case_base import TestCaseBase
import paddle

def tensor_method_call_1(x: paddle.Tensor):
    if False:
        i = 10
        return i + 15
    y = x + 1
    return y.mean()

def tensor_method_call_2(a: paddle.Tensor, b: paddle.Tensor):
    if False:
        for i in range(10):
            print('nop')
    c = a.add(b)
    d = c.multiply(a)
    e = d.subtract(b)
    f = e.divide(a)
    g = f.pow(2) + f.abs().sqrt()
    h = (g.abs() + 1).log() - (g / g.max()).exp()
    i = h.sin() + h.cos()
    return i

def tensor_method_passed_by_user(a: paddle.Tensor, func: paddle.Tensor):
    if False:
        i = 10
        return i + 15
    return func(a)

def tensor_method_property(a: paddle.Tensor, b: paddle.Tensor):
    if False:
        return 10
    return (a.name, str(a.place), a.persistable, a.dtype, a.type, a.is_tensor(), a.clear_gradient(), a @ b.T + len(a.shape) + b.size + a.ndim + a.dim() + a.rank())

def middle_tensor_name(a: paddle.Tensor, b: paddle.Tensor):
    if False:
        return 10
    c = a + b
    return c.name

class TestTensorMethod(TestCaseBase):

    def test_tensor_method_1(self):
        if False:
            return 10
        x = paddle.rand([10])
        y = paddle.rand([2, 4, 6])
        self.assert_results(tensor_method_call_1, x)
        self.assert_results(tensor_method_call_1, y)

    def test_tensor_method_2(self):
        if False:
            print('Hello World!')
        x = paddle.rand([42])
        y = paddle.rand([42])
        self.assert_results(tensor_method_call_2, x, y)

    def test_tensor_method_passed_by_user(self):
        if False:
            print('Hello World!')
        x = paddle.rand([42])
        y = paddle.rand([42])
        self.assert_results(tensor_method_passed_by_user, x, y.add)

    def test_tensor_method_property(self):
        if False:
            i = 10
            return i + 15
        x = paddle.rand([42, 24], dtype='float64')
        y = paddle.rand([42, 24], dtype='float32')
        self.assert_results(tensor_method_property, x, y)

    @unittest.skip('TODO: dynamic tensor name is different')
    def test_middle_tensor_name(self):
        if False:
            while True:
                i = 10
        x = paddle.rand([42, 24])
        y = paddle.rand([42, 24])
        self.assert_results(middle_tensor_name, x, y)
if __name__ == '__main__':
    unittest.main()