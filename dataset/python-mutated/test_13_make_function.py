from __future__ import annotations
import unittest
from test_case_base import TestCaseBase
import paddle

def make_fn(x: paddle.Tensor):
    if False:
        for i in range(10):
            print('nop')

    def fn(a, b=2, c=3, d=4):
        if False:
            while True:
                i = 10
        return a + b + c + d
    return fn(1) + fn(2, c=5) + x

class TestExecutor(TestCaseBase):

    def test_simple(self):
        if False:
            return 10
        self.assert_results(make_fn, paddle.to_tensor(1))
if __name__ == '__main__':
    unittest.main()