import numpy as np
import unittest
from numba.core.compiler import compile_isolated, Flags
from numba.core import errors, types
from numba import typeof
from numba.tests.support import TestCase, MemoryLeakMixin, tag
enable_pyobj_flags = Flags()
enable_pyobj_flags.enable_pyobject = True
force_pyobj_flags = Flags()
force_pyobj_flags.force_pyobject = True
no_pyobj_flags = Flags()
no_pyobj_flags.nrt = True

def unpack_list(l):
    if False:
        return 10
    (a, b, c) = l
    return (a, b, c)

def unpack_shape(a):
    if False:
        print('Hello World!')
    (x, y, z) = a.shape
    return x + y + z

def unpack_range():
    if False:
        i = 10
        return i + 15
    (a, b, c) = range(3)
    return a + b + c

def unpack_range_too_small():
    if False:
        while True:
            i = 10
    (a, b, c) = range(2)
    return a + b + c

def unpack_range_too_large():
    if False:
        for i in range(10):
            print('nop')
    (a, b, c) = range(4)
    return a + b + c

def unpack_tuple():
    if False:
        print('Hello World!')
    (a, b, c) = (1, 2, 3)
    return a + b + c

def unpack_tuple_too_small():
    if False:
        return 10
    (a, b, c) = (1, 2)
    return a + b + c

def unpack_tuple_too_large():
    if False:
        return 10
    (a, b, c) = (1, 2, 3, 4)
    return a + b + c

def unpack_heterogeneous_tuple_too_small():
    if False:
        return 10
    (a, b, c) = (1, 2.5j)
    return a + b + c

def unpack_heterogeneous_tuple_too_large():
    if False:
        i = 10
        return i + 15
    (a, b, c) = (1, 2.5, 3j, 4)
    return a + b + c

def unpack_heterogeneous_tuple():
    if False:
        while True:
            i = 10
    (a, b, c) = (1, 2.5, 3j)
    return a + b + c

def unpack_nested_heterogeneous_tuple():
    if False:
        for i in range(10):
            print('nop')
    (a, (b, c)) = (1, (2.5, 3j))
    return a + b + c

def unpack_arbitrary(seq):
    if False:
        while True:
            i = 10
    (a, b) = seq
    return (b, a)

def unpack_nrt():
    if False:
        for i in range(10):
            print('nop')
    a = np.zeros(1)
    b = np.zeros(2)
    tup = (b, a)
    (alpha, beta) = tup
    return (alpha, beta)

def chained_unpack_assign1(x, y):
    if False:
        while True:
            i = 10
    a = (b, c) = (x, y)
    (d, e) = a
    return d + e + b + c

def conditional_swap(x, y):
    if False:
        print('Hello World!')
    if x > 0:
        (x, y) = (y, x)
    return (x, y)

class TestUnpack(MemoryLeakMixin, TestCase):

    def test_unpack_list(self):
        if False:
            while True:
                i = 10
        pyfunc = unpack_list
        cr = compile_isolated(pyfunc, (), flags=force_pyobj_flags)
        cfunc = cr.entry_point
        l = [1, 2, 3]
        self.assertEqual(cfunc(l), pyfunc(l))

    def test_unpack_shape(self, flags=force_pyobj_flags):
        if False:
            return 10
        pyfunc = unpack_shape
        cr = compile_isolated(pyfunc, [types.Array(dtype=types.int32, ndim=3, layout='C')], flags=flags)
        cfunc = cr.entry_point
        a = np.zeros(shape=(1, 2, 3)).astype(np.int32)
        self.assertPreciseEqual(cfunc(a), pyfunc(a))

    def test_unpack_shape_npm(self):
        if False:
            while True:
                i = 10
        self.test_unpack_shape(flags=no_pyobj_flags)

    def test_unpack_range(self, flags=force_pyobj_flags):
        if False:
            print('Hello World!')
        self.run_nullary_func(unpack_range, flags)

    def test_unpack_range_npm(self):
        if False:
            print('Hello World!')
        self.test_unpack_range(flags=no_pyobj_flags)

    def test_unpack_tuple(self, flags=force_pyobj_flags):
        if False:
            print('Hello World!')
        self.run_nullary_func(unpack_tuple, flags)

    def test_unpack_tuple_npm(self):
        if False:
            print('Hello World!')
        self.test_unpack_tuple(flags=no_pyobj_flags)

    def test_unpack_heterogeneous_tuple(self, flags=force_pyobj_flags):
        if False:
            i = 10
            return i + 15
        self.run_nullary_func(unpack_heterogeneous_tuple, flags)

    def test_unpack_heterogeneous_tuple_npm(self):
        if False:
            print('Hello World!')
        self.test_unpack_heterogeneous_tuple(flags=no_pyobj_flags)

    def test_unpack_nested_heterogeneous_tuple(self, flags=force_pyobj_flags):
        if False:
            return 10
        self.run_nullary_func(unpack_nested_heterogeneous_tuple, flags)

    def test_unpack_nested_heterogeneous_tuple_npm(self):
        if False:
            while True:
                i = 10
        self.test_unpack_nested_heterogeneous_tuple(flags=no_pyobj_flags)

    def test_chained_unpack_assign(self, flags=force_pyobj_flags):
        if False:
            while True:
                i = 10
        pyfunc = chained_unpack_assign1
        cr = compile_isolated(pyfunc, [types.int32, types.int32], flags=flags)
        cfunc = cr.entry_point
        args = (4, 5)
        self.assertPreciseEqual(cfunc(*args), pyfunc(*args))

    def test_chained_unpack_assign_npm(self):
        if False:
            return 10
        self.test_chained_unpack_assign(flags=no_pyobj_flags)

    def check_unpack_error(self, pyfunc, flags=force_pyobj_flags, exc=ValueError):
        if False:
            print('Hello World!')
        with self.assertRaises(exc):
            cr = compile_isolated(pyfunc, (), flags=flags)
            cfunc = cr.entry_point
            cfunc()

    def test_unpack_tuple_too_small(self):
        if False:
            print('Hello World!')
        self.check_unpack_error(unpack_tuple_too_small)
        self.check_unpack_error(unpack_heterogeneous_tuple_too_small)

    def test_unpack_tuple_too_small_npm(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_unpack_error(unpack_tuple_too_small, no_pyobj_flags, errors.TypingError)
        self.check_unpack_error(unpack_heterogeneous_tuple_too_small, no_pyobj_flags, errors.TypingError)

    def test_unpack_tuple_too_large(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_unpack_error(unpack_tuple_too_large)
        self.check_unpack_error(unpack_heterogeneous_tuple_too_large)

    def test_unpack_tuple_too_large_npm(self):
        if False:
            while True:
                i = 10
        self.check_unpack_error(unpack_tuple_too_large, no_pyobj_flags, errors.TypingError)
        self.check_unpack_error(unpack_heterogeneous_tuple_too_large, no_pyobj_flags, errors.TypingError)

    def test_unpack_range_too_small(self):
        if False:
            print('Hello World!')
        self.check_unpack_error(unpack_range_too_small)

    def test_unpack_range_too_small_npm(self):
        if False:
            print('Hello World!')
        self.check_unpack_error(unpack_range_too_small, no_pyobj_flags)

    def test_unpack_range_too_large(self):
        if False:
            return 10
        self.check_unpack_error(unpack_range_too_large)

    def test_unpack_range_too_large_npm(self):
        if False:
            return 10
        self.check_unpack_error(unpack_range_too_large, no_pyobj_flags)

    def check_conditional_swap(self, flags=force_pyobj_flags):
        if False:
            i = 10
            return i + 15
        cr = compile_isolated(conditional_swap, (types.int32, types.int32), flags=flags)
        cfunc = cr.entry_point
        self.assertPreciseEqual(cfunc(4, 5), (5, 4))
        self.assertPreciseEqual(cfunc(0, 5), (0, 5))

    def test_conditional_swap(self):
        if False:
            return 10
        self.check_conditional_swap()

    def test_conditional_swap_npm(self):
        if False:
            print('Hello World!')
        self.check_conditional_swap(no_pyobj_flags)

    def test_unpack_tuple_of_arrays(self):
        if False:
            print('Hello World!')
        tup = tuple((np.zeros(i + 1) for i in range(2)))
        tupty = typeof(tup)
        pyfunc = unpack_arbitrary
        cr = compile_isolated(pyfunc, (tupty,), flags=no_pyobj_flags)
        cfunc = cr.entry_point
        self.assertPreciseEqual(cfunc(tup), pyfunc(tup))

    def test_unpack_nrt(self):
        if False:
            while True:
                i = 10
        pyfunc = unpack_nrt
        cr = compile_isolated(pyfunc, (), flags=no_pyobj_flags)
        cfunc = cr.entry_point
        self.assertPreciseEqual(cfunc(), pyfunc())

    def test_invalid_unpack(self):
        if False:
            while True:
                i = 10
        pyfunc = unpack_arbitrary
        with self.assertRaises(errors.TypingError) as raises:
            compile_isolated(pyfunc, (types.int32,), flags=no_pyobj_flags)
        self.assertIn('failed to unpack int32', str(raises.exception))
if __name__ == '__main__':
    unittest.main()