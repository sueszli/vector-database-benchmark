import unittest
from itertools import product
from numba import types, njit, typed, errors
from numba.tests.support import TestCase

class TestGetitemOnTypes(TestCase):

    def test_static_getitem_on_type(self):
        if False:
            while True:
                i = 10

        def gen(numba_type, index):
            if False:
                while True:
                    i = 10

            def foo():
                if False:
                    while True:
                        i = 10
                ty = numba_type[index]
                return typed.List.empty_list(ty)
            return foo
        tys = (types.bool_, types.float64, types.uint8, types.complex128)
        contig = slice(None, None, 1)
        noncontig = slice(None, None, None)
        indexes = (contig, noncontig, (noncontig, contig), (contig, noncontig), (noncontig, noncontig), (noncontig, noncontig, contig), (contig, noncontig, noncontig), (noncontig, noncontig, noncontig))
        for (ty, idx) in product(tys, indexes):
            compilable = njit(gen(ty, idx))
            expected = ty[idx]
            self.assertEqual(compilable()._dtype, expected)
            got = compilable.nopython_signatures[0].return_type.dtype
            self.assertEqual(got, expected)

    def test_shorthand_syntax(self):
        if False:
            print('Hello World!')

        @njit
        def foo1():
            if False:
                print('Hello World!')
            ty = types.float32[::1, :]
            return typed.List.empty_list(ty)
        self.assertEqual(foo1()._dtype, types.float32[::1, :])

        @njit
        def foo2():
            if False:
                return 10
            ty = types.complex64[:, :, :]
            return typed.List.empty_list(ty)
        self.assertEqual(foo2()._dtype, types.complex64[:, :, :])

    def test_static_getitem_on_invalid_type(self):
        if False:
            print('Hello World!')
        types.void[:]
        with self.assertRaises(errors.TypingError) as raises:

            @njit
            def foo():
                if False:
                    print('Hello World!')
                types.void[:]
            foo()
        msg = ('No implementation', 'getitem(typeref[none], slice<a:b>)')
        excstr = str(raises.exception)
        for m in msg:
            self.assertIn(m, excstr)

    def test_standard_getitem_on_type(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(errors.TypingError) as raises:

            @njit
            def foo(not_static):
                if False:
                    while True:
                        i = 10
                types.float64[not_static]
            foo(slice(None, None, 1))
        msg = ('No implementation', 'getitem(class(float64), slice<a:b>)')
        excstr = str(raises.exception)
        for m in msg:
            self.assertIn(m, excstr)
if __name__ == '__main__':
    unittest.main()