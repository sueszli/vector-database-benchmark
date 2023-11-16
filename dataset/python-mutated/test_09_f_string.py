from __future__ import annotations
import unittest
from test_case_base import TestCaseBase
import paddle
from paddle.jit.sot.psdb import assert_true

def foo(x: paddle.Tensor):
    if False:
        for i in range(10):
            print('nop')
    whilespace = 123
    hello_world = f'Hello {whilespace} World'
    z = assert_true(hello_world == 'Hello 123 World')
    x = x + 1
    return x

class TestFString(TestCaseBase):

    def test_fstring(self):
        if False:
            return 10
        self.assert_results(foo, paddle.to_tensor(1))
if __name__ == '__main__':
    unittest.main()