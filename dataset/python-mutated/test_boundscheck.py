import numpy as np
from numba.core.compiler import compile_isolated, DEFAULT_FLAGS
from numba.cuda.testing import SerialMixin
from numba import typeof, cuda, njit
from numba.core.types import float64
from numba.tests.support import MemoryLeakMixin, override_env_config
from numba.core import config
import unittest
BOUNDSCHECK_FLAGS = DEFAULT_FLAGS.copy()
BOUNDSCHECK_FLAGS.boundscheck = True

def basic_array_access(a):
    if False:
        i = 10
        return i + 15
    return a[10]

def slice_array_access(a):
    if False:
        return 10
    return a[10:, 10]

def fancy_array_access(x):
    if False:
        i = 10
        return i + 15
    a = np.array([1, 2, 3])
    return x[a]

def fancy_array_modify(x):
    if False:
        return 10
    a = np.array([1, 2, 3])
    x[a] = 0
    return x

class TestBoundsCheckNoError(MemoryLeakMixin, unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.old_boundscheck = config.BOUNDSCHECK
        config.BOUNDSCHECK = None

    def test_basic_array_boundscheck(self):
        if False:
            while True:
                i = 10
        a = np.arange(5)
        with self.assertRaises(IndexError):
            basic_array_access(a)
        at = typeof(a)
        c_noboundscheck = compile_isolated(basic_array_access, [at], flags=DEFAULT_FLAGS)
        noboundscheck = c_noboundscheck.entry_point
        noboundscheck(a)

    def test_slice_array_boundscheck(self):
        if False:
            while True:
                i = 10
        a = np.ones((5, 5))
        b = np.ones((5, 20))
        with self.assertRaises(IndexError):
            slice_array_access(a)
        slice_array_access(b)
        at = typeof(a)
        rt = float64[:]
        c_noboundscheck = compile_isolated(slice_array_access, [at], return_type=rt, flags=DEFAULT_FLAGS)
        noboundscheck = c_noboundscheck.entry_point
        c_boundscheck = compile_isolated(slice_array_access, [at], return_type=rt, flags=BOUNDSCHECK_FLAGS)
        boundscheck = c_boundscheck.entry_point
        noboundscheck(a)
        noboundscheck(b)
        boundscheck(b)

    def test_fancy_indexing_boundscheck(self):
        if False:
            print('Hello World!')
        a = np.arange(3)
        b = np.arange(4)
        with self.assertRaises(IndexError):
            fancy_array_access(a)
        fancy_array_access(b)
        at = typeof(a)
        rt = at.dtype[:]
        c_noboundscheck = compile_isolated(fancy_array_access, [at], return_type=rt, flags=DEFAULT_FLAGS)
        noboundscheck = c_noboundscheck.entry_point
        c_boundscheck = compile_isolated(fancy_array_access, [at], return_type=rt, flags=BOUNDSCHECK_FLAGS)
        boundscheck = c_boundscheck.entry_point
        noboundscheck(a)
        noboundscheck(b)
        boundscheck(b)

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        config.BOUNDSCHECK = self.old_boundscheck

class TestNoCudaBoundsCheck(SerialMixin, unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.old_boundscheck = config.BOUNDSCHECK
        config.BOUNDSCHECK = None

    @unittest.skipIf(not cuda.is_available(), 'NO CUDA')
    def test_no_cuda_boundscheck(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(NotImplementedError):

            @cuda.jit(boundscheck=True)
            def func():
                if False:
                    while True:
                        i = 10
                pass

        @cuda.jit(boundscheck=False)
        def func3():
            if False:
                for i in range(10):
                    print('nop')
            pass
        with override_env_config('NUMBA_BOUNDSCHECK', '1'):

            @cuda.jit
            def func2(x, a):
                if False:
                    for i in range(10):
                        print('nop')
                a[1] = x[1]
            a = np.ones((1,))
            x = np.zeros((1,))
            if not config.ENABLE_CUDASIM:
                func2[1, 1](x, a)

    def tearDown(self):
        if False:
            return 10
        config.BOUNDSCHECK = self.old_boundscheck

class TestBoundsCheckError(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.old_boundscheck = config.BOUNDSCHECK
        config.BOUNDSCHECK = None

    def test_basic_array_boundscheck(self):
        if False:
            while True:
                i = 10
        a = np.arange(5)
        with self.assertRaises(IndexError):
            basic_array_access(a)
        at = typeof(a)
        c_boundscheck = compile_isolated(basic_array_access, [at], flags=BOUNDSCHECK_FLAGS)
        boundscheck = c_boundscheck.entry_point
        with self.assertRaises(IndexError):
            boundscheck(a)

    def test_slice_array_boundscheck(self):
        if False:
            print('Hello World!')
        a = np.ones((5, 5))
        b = np.ones((5, 20))
        with self.assertRaises(IndexError):
            slice_array_access(a)
        slice_array_access(b)
        at = typeof(a)
        rt = float64[:]
        c_boundscheck = compile_isolated(slice_array_access, [at], return_type=rt, flags=BOUNDSCHECK_FLAGS)
        boundscheck = c_boundscheck.entry_point
        with self.assertRaises(IndexError):
            boundscheck(a)

    def test_fancy_indexing_boundscheck(self):
        if False:
            return 10
        a = np.arange(3)
        b = np.arange(4)
        with self.assertRaises(IndexError):
            fancy_array_access(a)
        fancy_array_access(b)
        at = typeof(a)
        rt = at.dtype[:]
        c_boundscheck = compile_isolated(fancy_array_access, [at], return_type=rt, flags=BOUNDSCHECK_FLAGS)
        boundscheck = c_boundscheck.entry_point
        with self.assertRaises(IndexError):
            boundscheck(a)

    def test_fancy_indexing_with_modification_boundscheck(self):
        if False:
            for i in range(10):
                print('nop')
        a = np.arange(3)
        b = np.arange(4)
        with self.assertRaises(IndexError):
            fancy_array_modify(a)
        fancy_array_modify(b)
        at = typeof(a)
        rt = at.dtype[:]
        c_boundscheck = compile_isolated(fancy_array_modify, [at], return_type=rt, flags=BOUNDSCHECK_FLAGS)
        boundscheck = c_boundscheck.entry_point
        with self.assertRaises(IndexError):
            boundscheck(a)

    def tearDown(self):
        if False:
            return 10
        config.BOUNDSCHECK = self.old_boundscheck

class TestBoundsEnvironmentVariable(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.old_boundscheck = config.BOUNDSCHECK
        config.BOUNDSCHECK = None

        @njit
        def default(x):
            if False:
                while True:
                    i = 10
            return x[1]

        @njit(boundscheck=False)
        def off(x):
            if False:
                print('Hello World!')
            return x[1]

        @njit(boundscheck=True)
        def on(x):
            if False:
                print('Hello World!')
            return x[1]
        self.default = default
        self.off = off
        self.on = on

    def test_boundscheck_unset(self):
        if False:
            for i in range(10):
                print('nop')
        with override_env_config('NUMBA_BOUNDSCHECK', ''):
            a = np.array([1])
            self.default(a)
            self.off(a)
            with self.assertRaises(IndexError):
                self.on(a)

    def test_boundscheck_enabled(self):
        if False:
            return 10
        with override_env_config('NUMBA_BOUNDSCHECK', '1'):
            a = np.array([1])
            with self.assertRaises(IndexError):
                self.default(a)
                self.off(a)
                self.on(a)

    def test_boundscheck_disabled(self):
        if False:
            return 10
        with override_env_config('NUMBA_BOUNDSCHECK', '0'):
            a = np.array([1])
            self.default(a)
            self.off(a)
            self.on(a)

    def tearDown(self):
        if False:
            return 10
        config.BOUNDSCHECK = self.old_boundscheck
if __name__ == '__main__':
    unittest.main()