"""
Testing object mode specifics.

"""
import numpy as np
import unittest
from numba.core.compiler import compile_isolated, Flags
from numba import jit
from numba.core import utils
from numba.tests.support import TestCase

def complex_constant(n):
    if False:
        i = 10
        return i + 15
    tmp = n + 4
    return tmp + 3j

def long_constant(n):
    if False:
        return 10
    return n + 100000000000000000000000000000000000000000000000

def delitem_usecase(x):
    if False:
        return 10
    del x[:]
forceobj = Flags()
forceobj.force_pyobject = True

def loop_nest_3(x, y):
    if False:
        print('Hello World!')
    n = 0
    for i in range(x):
        for j in range(y):
            for k in range(x + y):
                n += i * j
    return n

def array_of_object(x):
    if False:
        for i in range(10):
            print('nop')
    return x

class TestObjectMode(TestCase):

    def test_complex_constant(self):
        if False:
            for i in range(10):
                print('nop')
        pyfunc = complex_constant
        cres = compile_isolated(pyfunc, (), flags=forceobj)
        cfunc = cres.entry_point
        self.assertPreciseEqual(pyfunc(12), cfunc(12))

    def test_long_constant(self):
        if False:
            while True:
                i = 10
        pyfunc = long_constant
        cres = compile_isolated(pyfunc, (), flags=forceobj)
        cfunc = cres.entry_point
        self.assertPreciseEqual(pyfunc(12), cfunc(12))

    def test_loop_nest(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test bug that decref the iterator early.\n        If the bug occurs, a segfault should occur\n        '
        pyfunc = loop_nest_3
        cres = compile_isolated(pyfunc, (), flags=forceobj)
        cfunc = cres.entry_point
        self.assertEqual(pyfunc(5, 5), cfunc(5, 5))

        def bm_pyfunc():
            if False:
                while True:
                    i = 10
            pyfunc(5, 5)

        def bm_cfunc():
            if False:
                i = 10
                return i + 15
            cfunc(5, 5)
        print(utils.benchmark(bm_pyfunc))
        print(utils.benchmark(bm_cfunc))

    def test_array_of_object(self):
        if False:
            while True:
                i = 10
        cfunc = jit(array_of_object)
        objarr = np.array([object()] * 10)
        self.assertIs(cfunc(objarr), objarr)

    def test_sequence_contains(self):
        if False:
            i = 10
            return i + 15
        '\n        Test handling of the `in` comparison\n        '

        @jit(forceobj=True)
        def foo(x, y):
            if False:
                for i in range(10):
                    print('nop')
            return x in y
        self.assertTrue(foo(1, [0, 1]))
        self.assertTrue(foo(0, [0, 1]))
        self.assertFalse(foo(2, [0, 1]))
        with self.assertRaises(TypeError) as raises:
            foo(None, None)
        self.assertIn('is not iterable', str(raises.exception))

    def test_delitem(self):
        if False:
            print('Hello World!')
        pyfunc = delitem_usecase
        cres = compile_isolated(pyfunc, (), flags=forceobj)
        cfunc = cres.entry_point
        l = [3, 4, 5]
        cfunc(l)
        self.assertPreciseEqual(l, [])
        with self.assertRaises(TypeError):
            cfunc(42)

    def test_starargs_non_tuple(self):
        if False:
            while True:
                i = 10

        def consumer(*x):
            if False:
                for i in range(10):
                    print('nop')
            return x

        @jit(forceobj=True)
        def foo(x):
            if False:
                while True:
                    i = 10
            return consumer(*x)
        arg = 'ijo'
        got = foo(arg)
        expect = foo.py_func(arg)
        self.assertEqual(got, tuple(arg))
        self.assertEqual(got, expect)

class TestObjectModeInvalidRewrite(TestCase):
    """
    Tests to ensure that rewrite passes didn't affect objmode lowering.
    """

    def _ensure_objmode(self, disp):
        if False:
            return 10
        self.assertTrue(disp.signatures)
        self.assertFalse(disp.nopython_signatures)
        return disp

    def test_static_raise_in_objmode_fallback(self):
        if False:
            return 10
        '\n        Test code based on user submitted issue at\n        https://github.com/numba/numba/issues/2159\n        '

        def test0(n):
            if False:
                return 10
            return n

        def test1(n):
            if False:
                i = 10
                return i + 15
            if n == 0:
                raise ValueError()
            return test0(n)
        compiled = jit(test1)
        self.assertEqual(test1(10), compiled(10))
        self._ensure_objmode(compiled)

    def test_static_setitem_in_objmode_fallback(self):
        if False:
            return 10
        '\n        Test code based on user submitted issue at\n        https://github.com/numba/numba/issues/2169\n        '

        def test0(n):
            if False:
                for i in range(10):
                    print('nop')
            return n

        def test(a1, a2):
            if False:
                while True:
                    i = 10
            a1 = np.asarray(a1)
            a2[0] = 1
            return test0(a1.sum() + a2.sum())
        compiled = jit(test)
        args = (np.array([3]), np.array([4]))
        self.assertEqual(test(*args), compiled(*args))
        self._ensure_objmode(compiled)

    def test_dynamic_func_objmode(self):
        if False:
            print('Hello World!')
        '\n        Test issue https://github.com/numba/numba/issues/3355\n        '
        func_text = 'def func():\n'
        func_text += '    np.array([1,2,3])\n'
        loc_vars = {}
        custom_globals = {'np': np}
        exec(func_text, custom_globals, loc_vars)
        func = loc_vars['func']
        jitted = jit(forceobj=True)(func)
        jitted()
if __name__ == '__main__':
    unittest.main()