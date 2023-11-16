import math
import warnings
from numba import jit
from numba.core.errors import TypingError, NumbaWarning
from numba.tests.support import TestCase
import unittest

class TestSelfRecursion(TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        from numba.tests import recursion_usecases
        self.mod = recursion_usecases

    def check_fib(self, cfunc):
        if False:
            while True:
                i = 10
        self.assertPreciseEqual(cfunc(10), 55)

    def test_global_explicit_sig(self):
        if False:
            return 10
        self.check_fib(self.mod.fib1)

    def test_inner_explicit_sig(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_fib(self.mod.fib2)

    def test_global_implicit_sig(self):
        if False:
            i = 10
            return i + 15
        self.check_fib(self.mod.fib3)

    def test_runaway(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(TypingError) as raises:
            self.mod.runaway_self(123)
        self.assertIn('cannot type infer runaway recursion', str(raises.exception))

    def test_type_change(self):
        if False:
            print('Hello World!')
        pfunc = self.mod.make_type_change_self()
        cfunc = self.mod.make_type_change_self(jit(nopython=True))
        args = (13, 0.125)
        self.assertPreciseEqual(pfunc(*args), cfunc(*args))

    def test_raise(self):
        if False:
            print('Hello World!')
        with self.assertRaises(ValueError) as raises:
            self.mod.raise_self(3)
        self.assertEqual(str(raises.exception), 'raise_self')

    def test_optional_return(self):
        if False:
            i = 10
            return i + 15
        pfunc = self.mod.make_optional_return_case()
        cfunc = self.mod.make_optional_return_case(jit(nopython=True))
        for arg in (0, 5, 10, 15):
            self.assertEqual(pfunc(arg), cfunc(arg))

    def test_growing_return_tuple(self):
        if False:
            i = 10
            return i + 15
        cfunc = self.mod.make_growing_tuple_case(jit(nopython=True))
        with self.assertRaises(TypingError) as raises:
            cfunc(100)
        self.assertIn('Return type of recursive function does not converge', str(raises.exception))

class TestMutualRecursion(TestCase):

    def setUp(self):
        if False:
            return 10
        from numba.tests import recursion_usecases
        self.mod = recursion_usecases

    def test_mutual_1(self):
        if False:
            i = 10
            return i + 15
        expect = math.factorial(10)
        self.assertPreciseEqual(self.mod.outer_fac(10), expect)

    def test_mutual_2(self):
        if False:
            i = 10
            return i + 15
        (pfoo, pbar) = self.mod.make_mutual2()
        (cfoo, cbar) = self.mod.make_mutual2(jit(nopython=True))
        for x in [-1, 0, 1, 3]:
            self.assertPreciseEqual(pfoo(x=x), cfoo(x=x))
            self.assertPreciseEqual(pbar(y=x, z=1), cbar(y=x, z=1))

    def test_runaway(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(TypingError) as raises:
            self.mod.runaway_mutual(123)
        self.assertIn('cannot type infer runaway recursion', str(raises.exception))

    def test_type_change(self):
        if False:
            while True:
                i = 10
        pfunc = self.mod.make_type_change_mutual()
        cfunc = self.mod.make_type_change_mutual(jit(nopython=True))
        args = (13, 0.125)
        self.assertPreciseEqual(pfunc(*args), cfunc(*args))

    def test_four_level(self):
        if False:
            for i in range(10):
                print('nop')
        pfunc = self.mod.make_four_level()
        cfunc = self.mod.make_four_level(jit(nopython=True))
        arg = 7
        self.assertPreciseEqual(pfunc(arg), cfunc(arg))

    def test_inner_error(self):
        if False:
            while True:
                i = 10
        cfunc = self.mod.make_inner_error(jit(nopython=True))
        with self.assertRaises(TypingError) as raises:
            cfunc(2)
        errmsg = "Unknown attribute 'ndim'"
        self.assertIn(errmsg, str(raises.exception))
        cfunc = self.mod.make_inner_error(jit)
        pfunc = self.mod.make_inner_error()
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=NumbaWarning)
            got = cfunc(6)
        self.assertEqual(got, pfunc(6))

    def test_raise(self):
        if False:
            i = 10
            return i + 15
        cfunc = self.mod.make_raise_mutual()
        with self.assertRaises(ValueError) as raises:
            cfunc(2)
        self.assertEqual(str(raises.exception), 'raise_mutual')
if __name__ == '__main__':
    unittest.main()