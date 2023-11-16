from __future__ import annotations
import unittest
from test_case_base import TestCaseBase
import paddle

def func_dup_top_1():
    if False:
        i = 10
        return i + 15
    return True == True != False

def func_dup_top_2(x):
    if False:
        for i in range(10):
            print('nop')
    y = x + 1
    return True == True != False

def func_dup_top_two(x: list[paddle.Tensor]):
    if False:
        while True:
            i = 10
    x[0] += x[1]
    return x

class TestDupTop(TestCaseBase):

    def test_dup_top(self):
        if False:
            for i in range(10):
                print('nop')
        self.assert_results(func_dup_top_1)
        self.assert_results(func_dup_top_2, paddle.to_tensor(1.0))
if __name__ == '__main__':
    unittest.main()