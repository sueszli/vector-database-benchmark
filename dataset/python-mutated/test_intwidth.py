import unittest
import math
import sys
from numba import jit
from numba.core import utils
from numba.tests.support import TestCase, tag
max_uint64 = 18446744073709551615

def usecase_uint64_global():
    if False:
        for i in range(10):
            print('nop')
    return max_uint64

def usecase_uint64_constant():
    if False:
        print('Hello World!')
    return 18446744073709551615

def usecase_uint64_func():
    if False:
        print('Hello World!')
    return max(18446744073709551614, 18446744073709551615)

def usecase_int64_pos():
    if False:
        while True:
            i = 10
    return 9223372036854775807

def usecase_int64_neg():
    if False:
        i = 10
        return i + 15
    return -9223372036854775808

def usecase_int64_func():
    if False:
        for i in range(10):
            print('nop')
    return max(9223372036854775807, -9223372036854775808) + min(9223372036854775807, -9223372036854775808)

class IntWidthTest(TestCase):

    def check_nullary_func(self, pyfunc, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        cfunc = jit(**kwargs)(pyfunc)
        self.assertPreciseEqual(cfunc(), pyfunc())

    def test_global_uint64(self, nopython=False):
        if False:
            return 10
        pyfunc = usecase_uint64_global
        self.check_nullary_func(pyfunc, nopython=nopython)

    def test_global_uint64_npm(self):
        if False:
            for i in range(10):
                print('nop')
        self.test_global_uint64(nopython=True)

    def test_constant_uint64(self, nopython=False):
        if False:
            i = 10
            return i + 15
        pyfunc = usecase_uint64_constant
        self.check_nullary_func(pyfunc, nopython=nopython)

    def test_constant_uint64_npm(self):
        if False:
            return 10
        self.test_constant_uint64(nopython=True)

    def test_constant_uint64_function_call(self, nopython=False):
        if False:
            while True:
                i = 10
        pyfunc = usecase_uint64_func
        self.check_nullary_func(pyfunc, nopython=nopython)

    def test_constant_uint64_function_call_npm(self):
        if False:
            i = 10
            return i + 15
        self.test_constant_uint64_function_call(nopython=True)

    def test_bit_length(self):
        if False:
            return 10
        f = utils.bit_length
        self.assertEqual(f(127), 7)
        self.assertEqual(f(-127), 7)
        self.assertEqual(f(128), 8)
        self.assertEqual(f(-128), 7)
        self.assertEqual(f(255), 8)
        self.assertEqual(f(-255), 8)
        self.assertEqual(f(256), 9)
        self.assertEqual(f(-256), 8)
        self.assertEqual(f(-257), 9)
        self.assertEqual(f(2147483647), 31)
        self.assertEqual(f(-2147483647), 31)
        self.assertEqual(f(-2147483648), 31)
        self.assertEqual(f(2147483648), 32)
        self.assertEqual(f(4294967295), 32)
        self.assertEqual(f(18446744073709551615), 64)
        self.assertEqual(f(18446744073709551616), 65)

    def test_constant_int64(self, nopython=False):
        if False:
            while True:
                i = 10
        self.check_nullary_func(usecase_int64_pos, nopython=nopython)
        self.check_nullary_func(usecase_int64_neg, nopython=nopython)
        self.check_nullary_func(usecase_int64_func, nopython=nopython)

    def test_constant_int64_npm(self):
        if False:
            for i in range(10):
                print('nop')
        self.test_constant_int64(nopython=True)
if __name__ == '__main__':
    unittest.main()