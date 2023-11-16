from __future__ import annotations
import re
import unittest
import paddle
from paddle.jit.sot import symbolic_translate

def case1(x):
    if False:
        while True:
            i = 10
    return n

def case2(x):
    if False:
        return 10
    x = x + 1
    return x @ x

def case3(x):
    if False:
        print('Hello World!')
    y = x.undefined_attr
    return y

def case4_inner(x):
    if False:
        while True:
            i = 10
    y = x * 2
    print()
    y = y + 1
    return y.undefined_attr

def case4(x):
    if False:
        print('Hello World!')
    return case4_inner(x)

def case5_inner3(x):
    if False:
        return 10
    x += 1
    print(x)
    z = x + 1
    return z

def case5_inner2(x):
    if False:
        while True:
            i = 10
    x += 1
    z = case5_inner3(1 / 0)
    return z + 1

def case5_inner1(x):
    if False:
        print('Hello World!')
    return case5_inner2(x)

def case5(x):
    if False:
        print('Hello World!')
    y = case5_inner3(x)
    return case5_inner1(y) + 1

class TestException(unittest.TestCase):

    def catch_error(self, func, inputs, error_lines: int | list[int]):
        if False:
            while True:
                i = 10
        if isinstance(error_lines, int):
            error_lines = [error_lines]
        try:
            symbolic_translate(func)(inputs)
        except Exception as e:
            match_results = re.compile('File ".*", line (\\d+)').findall(str(e))
            match_results = list(map(int, match_results))
            assert match_results == error_lines, f'{match_results} is not equal {error_lines}'

    def test_all_case(self):
        if False:
            print('Hello World!')
        self.catch_error(case1, paddle.rand([2, 1]), 25)
        self.catch_error(case3, paddle.rand([2, 1]), 34)
        self.catch_error(case4, paddle.rand([2, 1]), 42)
        self.catch_error(case5, paddle.rand([3, 1]), [68, 63, 58])
if __name__ == '__main__':
    unittest.main()