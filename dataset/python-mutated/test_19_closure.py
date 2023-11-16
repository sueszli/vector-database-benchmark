import inspect
import unittest
from test_case_base import TestCaseBase
import paddle
from paddle.jit.sot.utils import strict_mode_guard

def foo(x: int, y: paddle.Tensor):
    if False:
        i = 10
        return i + 15
    z = 3

    def local(a, b=5):
        if False:
            print('Hello World!')
        return a + x + z + b + y
    return local(4) + z

def foo2(y: paddle.Tensor, x=1):
    if False:
        return 10
    '\n    Test strip default value\n    '
    z = 3

    def local(a, b=5):
        if False:
            i = 10
            return i + 15
        return a + x + z + b + y
    return local(4)

def foo3(y: paddle.Tensor, x=1):
    if False:
        while True:
            i = 10
    '\n    Test Closure Band Default\n    '
    z = 3

    def local(a, b=5):
        if False:
            while True:
                i = 10
        nonlocal z
        z = 4
        return a + x + z + b + y
    return local(4)
global_z = 3

def test_global(y: paddle.Tensor):
    if False:
        while True:
            i = 10
    '\n    Test Global variable\n    '

    def local(a, b=5):
        if False:
            for i in range(10):
                print('nop')
        global global_z
        global_z += 1
        return a + global_z + b + y
    return local(1)

def multi(c):
    if False:
        return 10
    return c + 2

def wrapper_function(func):
    if False:
        for i in range(10):
            print('nop')
    a = 2

    def inner():
        if False:
            while True:
                i = 10
        return func(a)
    return inner
wrapped_multi = wrapper_function(multi)

def foo5(y: paddle.Tensor):
    if False:
        print('Hello World!')
    '\n    Test incoming closures\n    '
    a = wrapped_multi()
    return a

def outwrapper(func):
    if False:
        while True:
            i = 10

    def wrapper(*args, **kwargs):
        if False:
            i = 10
            return i + 15
        return func(*args, **kwargs)
    return wrapper

def foo6(y: paddle.Tensor):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test Decorator\n    '

    @outwrapper
    def load_1(a, b=5):
        if False:
            for i in range(10):
                print('nop')
        return a + b
    return load_1(1)
import numpy as np

def numpy_sum(m):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test loop call\n\n    Example: a->b->c->a\n    '
    a = np.array([1, 2, 3])
    tmp = np.sum(a)
    return m + 1

def lambda_closure(x, m):
    if False:
        return 10
    '\n    lambda closure.\n    '

    def break_graph_closure():
        if False:
            print('Hello World!')
        print('yes')
        return x + m
    return break_graph_closure()

def kwargs_wrapper(func):
    if False:
        for i in range(10):
            print('nop')
    sig = inspect.signature(func)

    def inner(*args, **kwargs):
        if False:
            while True:
                i = 10
        return func(*args, **kwargs)
    inner.__signature__ = sig
    return inner

@kwargs_wrapper
def func7(a, b):
    if False:
        return 10
    return a + b

def foo7():
    if False:
        for i in range(10):
            print('nop')
    return func7(3, 5)

def create_closure():
    if False:
        print('Hello World!')
    x = 1

    def closure():
        if False:
            for i in range(10):
                print('nop')
        return x + 1
    return closure

class TestExecutor(TestCaseBase):

    def test_closure(self):
        if False:
            print('Hello World!')
        self.assert_results(foo, 1, paddle.to_tensor(2))
        self.assert_results(foo2, paddle.to_tensor(2))
        self.assert_results(foo3, paddle.to_tensor(2))
        self.assert_results_with_global_check(test_global, ['global_z'], paddle.to_tensor(2))
        self.assert_results(foo5, paddle.to_tensor(2))
        self.assert_results(foo6, paddle.to_tensor(2))
        self.assert_results(numpy_sum, paddle.to_tensor(1))
        with strict_mode_guard(False):
            self.assert_results(lambda_closure, paddle.to_tensor(2), paddle.to_tensor(1))

class TestExecutor2(TestCaseBase):

    def test_closure(self):
        if False:
            i = 10
            return i + 15
        self.assert_results(foo7)

def test_slice_in_for_loop(x, iter_num=3):
    if False:
        for i in range(10):
            print('nop')
    x = paddle.to_tensor(x)
    a = []
    iter_num = paddle.full(shape=[1], fill_value=iter_num, dtype='int32')
    for i in range(iter_num):
        a.append(x)
    for i in range(iter_num):
        a[i] = x
    out = a[2]
    return out

class TestExecutor3(TestCaseBase):

    def test_closure(self):
        if False:
            return 10
        tx = paddle.to_tensor([1.0, 2.0, 3.0])

def non_local_test(t: paddle.Tensor):
    if False:
        i = 10
        return i + 15
    a = 1

    def func1():
        if False:
            print('Hello World!')
        nonlocal a
        t = a
        a = 2
        return t

    def func2():
        if False:
            while True:
                i = 10
        nonlocal a
        a = 1
        return a
    t += func1()
    t += func2()
    t += a
    return t

class TestExecutor4(TestCaseBase):

    def test_closure(self):
        if False:
            for i in range(10):
                print('nop')
        tx = paddle.to_tensor([1.0])
        self.assert_results(non_local_test, tx)

class TestCreateClosure(TestCaseBase):

    def test_create_closure(self):
        if False:
            i = 10
            return i + 15
        closure = create_closure()
        self.assert_results(closure)
if __name__ == '__main__':
    unittest.main()