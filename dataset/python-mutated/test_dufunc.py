import itertools
import pickle
import textwrap
import numpy as np
from numba import njit, vectorize
from numba.tests.support import MemoryLeakMixin, TestCase
from numba.core.errors import TypingError
import unittest
from numba.np.ufunc import dufunc

def pyuadd(a0, a1):
    if False:
        for i in range(10):
            print('nop')
    return a0 + a1

def pysub(a0, a1):
    if False:
        return 10
    return a0 - a1

def pymult(a0, a1):
    if False:
        for i in range(10):
            print('nop')
    return a0 * a1

def pydiv(a0, a1):
    if False:
        return 10
    return a0 // a1

def pymin(a0, a1):
    if False:
        while True:
            i = 10
    return a0 if a0 < a1 else a1

class TestDUFunc(MemoryLeakMixin, unittest.TestCase):

    def nopython_dufunc(self, pyfunc):
        if False:
            while True:
                i = 10
        return dufunc.DUFunc(pyfunc, targetoptions=dict(nopython=True))

    def test_frozen(self):
        if False:
            i = 10
            return i + 15
        duadd = self.nopython_dufunc(pyuadd)
        self.assertFalse(duadd._frozen)
        duadd._frozen = True
        self.assertTrue(duadd._frozen)
        with self.assertRaises(ValueError):
            duadd._frozen = False
        with self.assertRaises(TypeError):
            duadd(np.linspace(0, 1, 10), np.linspace(1, 2, 10))

    def test_scalar(self):
        if False:
            while True:
                i = 10
        duadd = self.nopython_dufunc(pyuadd)
        self.assertEqual(pyuadd(1, 2), duadd(1, 2))

    def test_npm_call(self):
        if False:
            print('Hello World!')
        duadd = self.nopython_dufunc(pyuadd)

        @njit
        def npmadd(a0, a1, o0):
            if False:
                while True:
                    i = 10
            duadd(a0, a1, o0)
        X = np.linspace(0, 1.9, 20)
        X0 = X[:10]
        X1 = X[10:]
        out0 = np.zeros(10)
        npmadd(X0, X1, out0)
        np.testing.assert_array_equal(X0 + X1, out0)
        Y0 = X0.reshape((2, 5))
        Y1 = X1.reshape((2, 5))
        out1 = np.zeros((2, 5))
        npmadd(Y0, Y1, out1)
        np.testing.assert_array_equal(Y0 + Y1, out1)
        Y2 = X1[:5]
        out2 = np.zeros((2, 5))
        npmadd(Y0, Y2, out2)
        np.testing.assert_array_equal(Y0 + Y2, out2)

    def test_npm_call_implicit_output(self):
        if False:
            while True:
                i = 10
        duadd = self.nopython_dufunc(pyuadd)

        @njit
        def npmadd(a0, a1):
            if False:
                print('Hello World!')
            return duadd(a0, a1)
        X = np.linspace(0, 1.9, 20)
        X0 = X[:10]
        X1 = X[10:]
        out0 = npmadd(X0, X1)
        np.testing.assert_array_equal(X0 + X1, out0)
        Y0 = X0.reshape((2, 5))
        Y1 = X1.reshape((2, 5))
        out1 = npmadd(Y0, Y1)
        np.testing.assert_array_equal(Y0 + Y1, out1)
        Y2 = X1[:5]
        out2 = npmadd(Y0, Y2)
        np.testing.assert_array_equal(Y0 + Y2, out2)
        out3 = npmadd(1.0, 2.0)
        self.assertEqual(out3, 3.0)

    def test_ufunc_props(self):
        if False:
            i = 10
            return i + 15
        duadd = self.nopython_dufunc(pyuadd)
        self.assertEqual(duadd.nin, 2)
        self.assertEqual(duadd.nout, 1)
        self.assertEqual(duadd.nargs, duadd.nin + duadd.nout)
        self.assertEqual(duadd.ntypes, 0)
        self.assertEqual(duadd.types, [])
        self.assertEqual(duadd.identity, None)
        duadd(1, 2)
        self.assertEqual(duadd.ntypes, 1)
        self.assertEqual(duadd.ntypes, len(duadd.types))
        self.assertIsNone(duadd.signature)

    def test_ufunc_props_jit(self):
        if False:
            for i in range(10):
                print('nop')
        duadd = self.nopython_dufunc(pyuadd)
        duadd(1, 2)
        attributes = {'nin': duadd.nin, 'nout': duadd.nout, 'nargs': duadd.nargs, 'identity': duadd.identity, 'signature': duadd.signature}

        def get_attr_fn(attr):
            if False:
                i = 10
                return i + 15
            fn = f'\n                def impl():\n                    return duadd.{attr}\n            '
            l = {}
            exec(textwrap.dedent(fn), {'duadd': duadd}, l)
            return l['impl']
        for (attr, val) in attributes.items():
            cfunc = njit(get_attr_fn(attr))
            self.assertEqual(val, cfunc(), f'Attribute differs from original: {attr}')

class TestDUFuncMethods(TestCase):

    def _check_reduce(self, ufunc, dtype=None, initial=None):
        if False:
            return 10

        @njit
        def foo(a, axis, dtype, initial):
            if False:
                i = 10
                return i + 15
            return ufunc.reduce(a, axis=axis, dtype=dtype, initial=initial)
        inputs = [np.arange(5), np.arange(4).reshape(2, 2), np.arange(40).reshape(5, 4, 2)]
        for array in inputs:
            for axis in range(array.ndim):
                expected = foo.py_func(array, axis, dtype, initial)
                got = foo(array, axis, dtype, initial)
                self.assertPreciseEqual(expected, got)

    def _check_reduce_axis(self, ufunc, dtype, initial=None):
        if False:
            while True:
                i = 10

        @njit
        def foo(a, axis):
            if False:
                i = 10
                return i + 15
            return ufunc.reduce(a, axis=axis, initial=initial)

        def _check(*args):
            if False:
                while True:
                    i = 10
            try:
                expected = foo.py_func(array, axis)
            except ValueError as e:
                self.assertEqual(e.args[0], exc_msg)
                with self.assertRaisesRegex(TypingError, exc_msg):
                    got = foo(array, axis)
            else:
                got = foo(array, axis)
                self.assertPreciseEqual(expected, got)
        exc_msg = f"reduction operation '{ufunc.__name__}' is not reorderable, so at most one axis may be specified"
        inputs = [np.arange(40, dtype=dtype).reshape(5, 4, 2), np.arange(10, dtype=dtype)]
        for array in inputs:
            for i in range(1, array.ndim + 1):
                for axis in itertools.combinations(range(array.ndim), r=i):
                    _check(array, axis)
            for axis in ((), None):
                _check(array, axis)

    def test_add_reduce(self):
        if False:
            for i in range(10):
                print('nop')
        duadd = vectorize('int64(int64, int64)', identity=0)(pyuadd)
        self._check_reduce(duadd)
        self._check_reduce_axis(duadd, dtype=np.int64)

    def test_mul_reduce(self):
        if False:
            print('Hello World!')
        dumul = vectorize('int64(int64, int64)', identity=1)(pymult)
        self._check_reduce(dumul)

    def test_non_associative_reduce(self):
        if False:
            for i in range(10):
                print('nop')
        dusub = vectorize('int64(int64, int64)')(pysub)
        dudiv = vectorize('int64(int64, int64)')(pydiv)
        self._check_reduce(dusub)
        self._check_reduce_axis(dusub, dtype=np.int64)
        self._check_reduce(dudiv)
        self._check_reduce_axis(dudiv, dtype=np.int64)

    def test_reduce_dtype(self):
        if False:
            for i in range(10):
                print('nop')
        duadd = vectorize('float64(float64, int64)', identity=0)(pyuadd)
        self._check_reduce(duadd, dtype=np.float64)

    def test_min_reduce(self):
        if False:
            while True:
                i = 10
        dumin = vectorize('int64(int64, int64)')(pymin)
        self._check_reduce(dumin, initial=10)
        self._check_reduce_axis(dumin, dtype=np.int64)

    def test_add_reduce_initial(self):
        if False:
            print('Hello World!')
        duadd = vectorize('int64(int64, int64)', identity=0)(pyuadd)
        self._check_reduce(duadd, dtype=np.int64, initial=100)

    def test_add_reduce_no_initial_or_identity(self):
        if False:
            i = 10
            return i + 15
        duadd = vectorize('int64(int64, int64)')(pyuadd)
        self._check_reduce(duadd, dtype=np.int64)

    def test_invalid_input(self):
        if False:
            i = 10
            return i + 15
        duadd = vectorize('float64(float64, int64)', identity=0)(pyuadd)

        @njit
        def foo(a):
            if False:
                for i in range(10):
                    print('nop')
            return duadd.reduce(a)
        exc_msg = 'The first argument "array" must be array-like'
        with self.assertRaisesRegex(TypingError, exc_msg):
            foo('a')

class TestDUFuncPickling(MemoryLeakMixin, unittest.TestCase):

    def check(self, ident, result_type):
        if False:
            for i in range(10):
                print('nop')
        buf = pickle.dumps(ident)
        rebuilt = pickle.loads(buf)
        r = rebuilt(123)
        self.assertEqual(123, r)
        self.assertIsInstance(r, result_type)

        @njit
        def foo(x):
            if False:
                while True:
                    i = 10
            return rebuilt(x)
        r = foo(321)
        self.assertEqual(321, r)
        self.assertIsInstance(r, result_type)

    def test_unrestricted(self):
        if False:
            i = 10
            return i + 15

        @vectorize
        def ident(x1):
            if False:
                print('Hello World!')
            return x1
        self.check(ident, result_type=(int, np.integer))

    def test_restricted(self):
        if False:
            print('Hello World!')

        @vectorize(['float64(float64)'])
        def ident(x1):
            if False:
                i = 10
                return i + 15
            return x1
        self.check(ident, result_type=float)
if __name__ == '__main__':
    unittest.main()