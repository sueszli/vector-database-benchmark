import ctypes
import ctypes.util
import os
import sys
import threading
import warnings
import numpy as np
import unittest
from numba.core.compiler import compile_isolated, Flags
from numba import jit
from numba.core import errors
from numba.tests.support import TestCase, tag
PyThread_get_thread_ident = ctypes.pythonapi.PyThread_get_thread_ident
PyThread_get_thread_ident.restype = ctypes.c_long
PyThread_get_thread_ident.argtypes = []
if os.name == 'nt':
    sleep = ctypes.windll.kernel32.Sleep
    sleep.argtypes = [ctypes.c_uint]
    sleep.restype = None
    sleep_factor = 1
else:
    sleep = ctypes.CDLL(ctypes.util.find_library('c')).usleep
    sleep.argtypes = [ctypes.c_uint]
    sleep.restype = ctypes.c_int
    sleep_factor = 1000

def f(a, indices):
    if False:
        print('Hello World!')
    for idx in indices:
        sleep(10 * sleep_factor)
        a[idx] = PyThread_get_thread_ident()
f_sig = 'void(int64[:], intp[:])'

def lifted_f(a, indices):
    if False:
        print('Hello World!')
    '\n    Same as f(), but inside a lifted loop\n    '
    object()
    for idx in indices:
        sleep(10 * sleep_factor)
        a[idx] = PyThread_get_thread_ident()

def object_f(a, indices):
    if False:
        print('Hello World!')
    '\n    Same as f(), but in object mode\n    '
    for idx in indices:
        sleep(10 * sleep_factor)
        object()
        a[idx] = PyThread_get_thread_ident()

class TestGILRelease(TestCase):

    def make_test_array(self, n_members):
        if False:
            return 10
        return np.arange(n_members, dtype=np.int64)

    def run_in_threads(self, func, n_threads):
        if False:
            for i in range(10):
                print('nop')
        threads = []
        func(self.make_test_array(1), np.arange(1, dtype=np.intp))
        arr = self.make_test_array(50)
        for i in range(n_threads):
            indices = np.arange(arr.size, dtype=np.intp)
            np.random.shuffle(indices)
            t = threading.Thread(target=func, args=(arr, indices))
            threads.append(t)
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return arr

    def check_gil_held(self, func):
        if False:
            for i in range(10):
                print('nop')
        arr = self.run_in_threads(func, n_threads=4)
        distinct = set(arr)
        self.assertEqual(len(distinct), 1, distinct)

    def check_gil_released(self, func):
        if False:
            while True:
                i = 10
        for n_threads in (4, 12, 32):
            arr = self.run_in_threads(func, n_threads)
            distinct = set(arr)
            try:
                self.assertGreater(len(distinct), 1, distinct)
            except AssertionError as e:
                failure = e
            else:
                return
        raise failure

    def test_gil_held(self):
        if False:
            while True:
                i = 10
        '\n        Test the GIL is held by default, by checking serialized runs\n        produce deterministic results.\n        '
        cfunc = jit(f_sig, nopython=True)(f)
        self.check_gil_held(cfunc)

    def test_gil_released(self):
        if False:
            while True:
                i = 10
        '\n        Test releasing the GIL, by checking parallel runs produce\n        unpredictable results.\n        '
        cfunc = jit(f_sig, nopython=True, nogil=True)(f)
        self.check_gil_released(cfunc)

    def test_gil_released_inside_lifted_loop(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test the GIL can by released by a lifted loop even though the\n        surrounding code uses object mode.\n        '
        cfunc = jit(f_sig, nogil=True)(lifted_f)
        self.check_gil_released(cfunc)

    def test_gil_released_by_caller(self):
        if False:
            print('Hello World!')
        '\n        Releasing the GIL in the caller is sufficient to have it\n        released in a callee.\n        '
        compiled_f = jit(f_sig, nopython=True)(f)

        @jit(f_sig, nopython=True, nogil=True)
        def caller(a, i):
            if False:
                i = 10
                return i + 15
            compiled_f(a, i)
        self.check_gil_released(caller)

    def test_gil_released_by_caller_and_callee(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Same, but with both caller and callee asking to release the GIL.\n        '
        compiled_f = jit(f_sig, nopython=True, nogil=True)(f)

        @jit(f_sig, nopython=True, nogil=True)
        def caller(a, i):
            if False:
                for i in range(10):
                    print('nop')
            compiled_f(a, i)
        self.check_gil_released(caller)

    def test_gil_ignored_by_callee(self):
        if False:
            while True:
                i = 10
        '\n        When only the callee asks to release the GIL, it gets ignored.\n        '
        compiled_f = jit(f_sig, nopython=True, nogil=True)(f)

        @jit(f_sig, nopython=True)
        def caller(a, i):
            if False:
                return 10
            compiled_f(a, i)
        self.check_gil_held(caller)

    def test_object_mode(self):
        if False:
            while True:
                i = 10
        '\n        When the function is compiled in object mode, a warning is\n        printed out.\n        '
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter('always', errors.NumbaWarning)
            cfunc = jit(f_sig, nogil=True)(object_f)
        self.assertTrue(any((w.category is errors.NumbaWarning and "Code running in object mode won't allow parallel execution" in str(w.message) for w in wlist)), wlist)
        self.run_in_threads(cfunc, 2)
if __name__ == '__main__':
    unittest.main()