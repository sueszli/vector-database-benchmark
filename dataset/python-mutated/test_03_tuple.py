from __future__ import annotations
import unittest
from test_case_base import TestCaseBase
import paddle
from paddle.jit.sot.psdb import check_no_breakgraph

@check_no_breakgraph
def build_tuple(x: int, y: paddle.Tensor):
    if False:
        i = 10
        return i + 15
    x = (x, y)
    return x[1] + 1

@check_no_breakgraph
def build_tuple_with_slice_subscript(x: int, y: paddle.Tensor):
    if False:
        while True:
            i = 10
    z = (x, y, 3, 4)
    return z[0:5:1]

@check_no_breakgraph
def build_tuple_with_int_subscript(x: int, y: paddle.Tensor):
    if False:
        return 10
    z = (x, y)
    return z[0]

@check_no_breakgraph
def tuple_count_int(x: int, y: paddle.Tensor):
    if False:
        return 10
    z = (x, x, 2, 1)
    return z.count(x)

def tuple_count_tensor(x: paddle.Tensor, y: tuple[paddle.Tensor]):
    if False:
        i = 10
        return i + 15
    return y.count(x)

@check_no_breakgraph
def tuple_index_int(x: int, y: paddle.Tensor):
    if False:
        print('Hello World!')
    z = (x, y, x, y, y)
    return z.index(x)

def tuple_index_tensor(x: paddle.Tensor, y: tuple[paddle.Tensor]):
    if False:
        print('Hello World!')
    return y.index(x)

class TestBuildTuple(TestCaseBase):

    def test_build_tuple(self):
        if False:
            for i in range(10):
                print('nop')
        self.assert_results(build_tuple, 1, paddle.to_tensor(2))
        self.assert_results(build_tuple_with_slice_subscript, 1, paddle.to_tensor(2))
        self.assert_results(build_tuple_with_int_subscript, 1, paddle.to_tensor(2))

class TestTupleMethods(TestCaseBase):

    def test_tuple_methods_int(self):
        if False:
            for i in range(10):
                print('nop')
        self.assert_results(tuple_count_int, 1, paddle.to_tensor(2))
        self.assert_results(tuple_index_int, 1, paddle.to_tensor(2))

    def test_tuple_methods_tensor(self):
        if False:
            for i in range(10):
                print('nop')
        a = paddle.to_tensor(1)
        b = paddle.to_tensor(2)
        self.assert_results(tuple_count_tensor, a, (a, b, a, b))
        self.assert_results(tuple_index_tensor, b, (b, b, b, a))
if __name__ == '__main__':
    unittest.main()