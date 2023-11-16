import unittest
from test_case_base import TestCaseBase
import paddle
patched = lambda self, x: x * self.a
patched2 = lambda self, x: x * self.a + 3

class A:

    def __init__(self, a):
        if False:
            while True:
                i = 10
        self.a = a

    def __call__(self, x):
        if False:
            print('Hello World!')
        return self.add(x)

    def add(self, x):
        if False:
            return 10
        return x + self.a
    multi = patched

class B:

    def __init__(self, a):
        if False:
            i = 10
            return i + 15
        self.a = A(a)

    def __call__(self, x, func):
        if False:
            while True:
                i = 10
        return getattr(self.a, func)(x)

    def self_call(self, x, func):
        if False:
            while True:
                i = 10
        return getattr(self.a, func)(self.a, x)

def foo_1(a, x):
    if False:
        print('Hello World!')
    return a(x)

def foo_2(a, x):
    if False:
        i = 10
        return i + 15
    return a.multi(x)

def foo_3(b, x):
    if False:
        i = 10
        return i + 15
    return b(x, 'multi')

def foo_4(b, x):
    if False:
        while True:
            i = 10
    return b(x, 'add')

def foo_5(b, x):
    if False:
        i = 10
        return i + 15
    return b.self_call(x, 'multi')

class TestExecutor(TestCaseBase):

    def test_simple(self):
        if False:
            return 10
        c = B(13)
        c.a.multi = patched2
        self.assert_results(foo_1, A(13), paddle.to_tensor(2))
        self.assert_results(foo_2, A(13), paddle.to_tensor(2))
        self.assert_results(foo_3, B(13), paddle.to_tensor(2))
        self.assert_results(foo_4, B(13), paddle.to_tensor(2))
        self.assert_results(foo_5, c, paddle.to_tensor(2))
        self.assert_results(foo_4, c, paddle.to_tensor(2))
if __name__ == '__main__':
    unittest.main()