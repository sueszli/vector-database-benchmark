from __future__ import annotations
import unittest
from test_case_base import TestCaseBase
import paddle

def test_delete_fast(a):
    if False:
        for i in range(10):
            print('nop')
    a = a + 2
    t = a * 3
    del t
    return a

class TestExecutor(TestCaseBase):

    def test_simple(self):
        if False:
            while True:
                i = 10
        a = paddle.to_tensor(1)
        self.assert_results(test_delete_fast, a)
if __name__ == '__main__':
    unittest.main()