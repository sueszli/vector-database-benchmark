from __future__ import annotations
import unittest
from test_case_base import TestCaseBase
import paddle

def unpack_tuple(x: tuple[int, paddle.Tensor]):
    if False:
        for i in range(10):
            print('nop')
    (y, z) = x
    return z + 1

def unpack_tensor(x: paddle.Tensor):
    if False:
        i = 10
        return i + 15
    (a, b) = x
    return (a, b)

def unpack_ex_tuple(x: tuple[int, int, paddle.Tensor]):
    if False:
        for i in range(10):
            print('nop')
    (*y, z) = x
    return z + 1

def unpack_ex_tensor(x: paddle.Tensor):
    if False:
        i = 10
        return i + 15
    (a, b, *c) = x
    return (a, b)

def unpack_ex_tensor_2(x: paddle.Tensor):
    if False:
        print('Hello World!')
    (a, *b, c, d) = x
    return (a, c)

class TestUnpack(TestCaseBase):

    def test_unpack_tuple(self):
        if False:
            while True:
                i = 10
        self.assert_results(unpack_tuple, (1, paddle.to_tensor(2)))

    def test_unpack_tensor(self):
        if False:
            while True:
                i = 10
        self.assert_results(unpack_tensor, paddle.to_tensor([2, 3]))

    def test_unpack_ex_tuple(self):
        if False:
            return 10
        self.assert_results(unpack_ex_tuple, (1, 1, paddle.to_tensor(2)))

    def test_unpack_ex_tensor(self):
        if False:
            i = 10
            return i + 15
        self.assert_results(unpack_ex_tensor, paddle.to_tensor([2, 3, 3, 3]))

    def test_unpack_ex_tensor_2(self):
        if False:
            for i in range(10):
                print('nop')
        self.assert_results(unpack_ex_tensor_2, paddle.to_tensor([2, 3, 3, 3]))
if __name__ == '__main__':
    unittest.main()