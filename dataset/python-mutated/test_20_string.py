from __future__ import annotations
import unittest
from test_case_base import TestCaseBase
import paddle
from paddle.jit.sot.psdb import assert_true, check_no_breakgraph

def string_format(x: paddle.Tensor):
    if False:
        print('Hello World!')
    whilespace = 123
    hello_world = f'Hello {whilespace} World'
    z = assert_true(hello_world == 'Hello 123 World')
    hello_world2 = f'Hello {whilespace}{whilespace} World'
    z = assert_true(hello_world2 == 'Hello 123123 World')
    hello_world_lower = 'Hello World'.lower()
    z = assert_true(hello_world_lower == 'hello world')
    return x + 1

def string_lower(x: paddle.Tensor):
    if False:
        while True:
            i = 10
    hello_world_lower = 'Hello World'.lower()
    z = assert_true(hello_world_lower == 'hello world')
    return x + 1

@check_no_breakgraph
def str_startswith():
    if False:
        i = 10
        return i + 15
    s = 'Hello World'
    a1 = s.startswith('Hello')
    a2 = s.startswith('World')
    a3 = s.startswith('Hello World')
    a4 = s.startswith('Hello World!')
    a5 = s.startswith('Hello', 5)
    a6 = s.startswith('Hello', 1, 4)
    a7 = s.startswith('Hello', 0, 11)
    return (a1, a2, a3, a4, a5, a6, a7)

@check_no_breakgraph
def str_endswith():
    if False:
        return 10
    s = 'Hello World'
    a1 = s.endswith('Hello')
    a2 = s.endswith('World')
    a3 = s.endswith('Hello World')
    a4 = s.endswith('Hello World!')
    a5 = s.endswith('Hello', 5)
    a6 = s.endswith('Hello', 0, 4)
    a7 = s.endswith('Hello', 1, 11)
    return (a1, a2, a3, a4, a5, a6, a7)

class TestExecutor(TestCaseBase):

    def test_string_format(self):
        if False:
            for i in range(10):
                print('nop')
        self.assert_results(string_format, paddle.to_tensor(1))

    def test_string_lower(self):
        if False:
            print('Hello World!')
        self.assert_results(string_lower, paddle.to_tensor(1))

    def test_str_startswith(self):
        if False:
            while True:
                i = 10
        self.assert_results(str_startswith)

    def test_str_endswith(self):
        if False:
            i = 10
            return i + 15
        self.assert_results(str_endswith)
if __name__ == '__main__':
    unittest.main()