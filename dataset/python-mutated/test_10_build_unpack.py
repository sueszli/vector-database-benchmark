from __future__ import annotations
import unittest
from test_case_base import TestCaseBase
import paddle

def build_tuple_unpack(x: tuple[paddle.Tensor], y: tuple[paddle.Tensor]):
    if False:
        return 10
    z = (*x, *y)
    return z[0] + 1

def build_list_unpack(x: list[paddle.Tensor], y: list[paddle.Tensor]):
    if False:
        for i in range(10):
            print('nop')
    z = [*x, *y]
    return z[0] + 1

def build_tuple_unpack_with_call(x: tuple[paddle.Tensor], y: tuple[paddle.Tensor]):
    if False:
        for i in range(10):
            print('nop')
    z = build_tuple_unpack_with_call_inner(*x, *y)
    return z[0] + 1

def build_tuple_unpack_with_call_inner(a: paddle.Tensor, b: paddle.Tensor, c: paddle.Tensor, d: paddle.Tensor):
    if False:
        print('Hello World!')
    z = (a, b, c, d)
    return z

def build_map_unpack(x: dict[str, paddle.Tensor], y: dict[str, paddle.Tensor]):
    if False:
        print('Hello World!')
    z = {**x, **y}
    return z['a'] + 1

def build_map_unpack_with_call_inner(a: paddle.Tensor, b: paddle.Tensor, c: paddle.Tensor, d: paddle.Tensor):
    if False:
        i = 10
        return i + 15
    z = {'a': a, 'b': b, 'c': c, 'd': d}
    return z

def build_map_unpack_with_call(x: dict[str, paddle.Tensor], y: dict[str, paddle.Tensor]):
    if False:
        print('Hello World!')
    z = build_map_unpack_with_call_inner(**x, **y)
    return z['a'] + 1

class TestExecutor(TestCaseBase):

    def test_simple(self):
        if False:
            while True:
                i = 10
        a = paddle.to_tensor(1)
        b = paddle.to_tensor(2)
        c = paddle.to_tensor(3)
        d = paddle.to_tensor(4)
        self.assert_results(build_tuple_unpack, (a, b), (c, d))
        self.assert_results(build_list_unpack, [a, b], [c, d])
        self.assert_results(build_tuple_unpack_with_call, (a, b), (c, d))
        self.assert_results(build_map_unpack, {'a': a, 'b': b}, {'c': c, 'd': d})
        self.assert_results(build_map_unpack_with_call, {'a': a, 'b': b}, {'c': c, 'd': d})
if __name__ == '__main__':
    unittest.main()