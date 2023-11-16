"""
Test problems in nested calls.
Usually due to invalid type conversion between function boundaries.
"""
from numba import int32, int64
from numba import jit, generated_jit
from numba.core import types
from numba.tests.support import TestCase, tag
import unittest

@jit(nopython=True)
def f_inner(a, b, c):
    if False:
        while True:
            i = 10
    return (a, b, c)

def f(x, y, z):
    if False:
        for i in range(10):
            print('nop')
    return f_inner(x, c=y, b=z)

@jit(nopython=True)
def g_inner(a, b=2, c=3):
    if False:
        for i in range(10):
            print('nop')
    return (a, b, c)

def g(x, y, z):
    if False:
        print('Hello World!')
    return (g_inner(x, b=y), g_inner(a=z, c=x))

@jit(nopython=True)
def star_inner(a=5, *b):
    if False:
        for i in range(10):
            print('nop')
    return (a, b)

def star(x, y, z):
    if False:
        print('Hello World!')
    return (star_inner(a=x), star_inner(x, y, z))

def star_call(x, y, z):
    if False:
        for i in range(10):
            print('nop')
    return (star_inner(x, *y), star_inner(*z))

@jit(nopython=True)
def argcast_inner(a, b):
    if False:
        print('Hello World!')
    if b:
        a = int64(0)
    return a

def argcast(a, b):
    if False:
        for i in range(10):
            print('nop')
    return argcast_inner(int32(a), b)

@generated_jit(nopython=True)
def generated_inner(x, y=5, z=6):
    if False:
        print('Hello World!')
    if isinstance(x, types.Complex):

        def impl(x, y, z):
            if False:
                return 10
            return (x + y, z)
    else:

        def impl(x, y, z):
            if False:
                for i in range(10):
                    print('nop')
            return (x - y, z)
    return impl

def call_generated(a, b):
    if False:
        i = 10
        return i + 15
    return generated_inner(a, z=b)

class TestNestedCall(TestCase):

    def compile_func(self, pyfunc, objmode=False):
        if False:
            while True:
                i = 10

        def check(*args, **kwargs):
            if False:
                return 10
            expected = pyfunc(*args, **kwargs)
            result = f(*args, **kwargs)
            self.assertPreciseEqual(result, expected)
        flags = dict(forceobj=True) if objmode else dict(nopython=True)
        f = jit(**flags)(pyfunc)
        return (f, check)

    def test_boolean_return(self):
        if False:
            while True:
                i = 10

        @jit(nopython=True)
        def inner(x):
            if False:
                i = 10
                return i + 15
            return not x

        @jit(nopython=True)
        def outer(x):
            if False:
                return 10
            if inner(x):
                return True
            else:
                return False
        self.assertFalse(outer(True))
        self.assertTrue(outer(False))

    def test_named_args(self, objmode=False):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test a nested function call with named (keyword) arguments.\n        '
        (cfunc, check) = self.compile_func(f, objmode)
        check(1, 2, 3)
        check(1, y=2, z=3)

    def test_named_args_objmode(self):
        if False:
            return 10
        self.test_named_args(objmode=True)

    def test_default_args(self, objmode=False):
        if False:
            i = 10
            return i + 15
        '\n        Test a nested function call using default argument values.\n        '
        (cfunc, check) = self.compile_func(g, objmode)
        check(1, 2, 3)
        check(1, y=2, z=3)

    def test_default_args_objmode(self):
        if False:
            while True:
                i = 10
        self.test_default_args(objmode=True)

    def test_star_args(self):
        if False:
            return 10
        '\n        Test a nested function call to a function with *args in its signature.\n        '
        (cfunc, check) = self.compile_func(star)
        check(1, 2, 3)

    def test_star_call(self, objmode=False):
        if False:
            print('Hello World!')
        '\n        Test a function call with a *args.\n        '
        (cfunc, check) = self.compile_func(star_call, objmode)
        check(1, (2,), (3,))

    def test_star_call_objmode(self):
        if False:
            while True:
                i = 10
        self.test_star_call(objmode=True)

    def test_argcast(self):
        if False:
            print('Hello World!')
        '\n        Issue #1488: implicitly casting an argument variable should not\n        break nested calls.\n        '
        (cfunc, check) = self.compile_func(argcast)
        check(1, 0)
        check(1, 1)

    def test_call_generated(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test a nested function call to a generated jit function.\n        '
        cfunc = jit(nopython=True)(call_generated)
        self.assertPreciseEqual(cfunc(1, 2), (-4, 2))
        self.assertPreciseEqual(cfunc(1j, 2), (1j + 5, 2))
if __name__ == '__main__':
    unittest.main()