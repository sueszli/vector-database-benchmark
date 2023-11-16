"""
Test threadsafety for compiler.
These tests will cause segfault if fail.
"""
import threading
import random
import numpy as np
from numba import jit, vectorize, guvectorize
from numba.tests.support import temp_directory, override_config
from numba.core import config
import unittest

def foo(n, v):
    if False:
        i = 10
        return i + 15
    return np.ones(n)

def ufunc_foo(a, b):
    if False:
        while True:
            i = 10
    return a + b

def gufunc_foo(a, b, out):
    if False:
        return 10
    out[0] = a + b

class TestThreadSafety(unittest.TestCase):

    def run_jit(self, **options):
        if False:
            while True:
                i = 10

        def runner():
            if False:
                while True:
                    i = 10
            cfunc = jit(**options)(foo)
            return cfunc(4, 10)
        return runner

    def run_compile(self, fnlist):
        if False:
            return 10
        self._cache_dir = temp_directory(self.__class__.__name__)
        with override_config('CACHE_DIR', self._cache_dir):

            def chooser():
                if False:
                    for i in range(10):
                        print('nop')
                for _ in range(10):
                    fn = random.choice(fnlist)
                    fn()
            ths = [threading.Thread(target=chooser) for i in range(4)]
            for th in ths:
                th.start()
            for th in ths:
                th.join()

    def test_concurrent_jit(self):
        if False:
            return 10
        self.run_compile([self.run_jit(nopython=True)])

    def test_concurrent_jit_cache(self):
        if False:
            return 10
        self.run_compile([self.run_jit(nopython=True, cache=True)])

    def run_vectorize(self, **options):
        if False:
            for i in range(10):
                print('nop')

        def runner():
            if False:
                return 10
            cfunc = vectorize(['(f4, f4)'], **options)(ufunc_foo)
            a = b = np.random.random(10).astype(np.float32)
            return cfunc(a, b)
        return runner

    def test_concurrent_vectorize(self):
        if False:
            for i in range(10):
                print('nop')
        self.run_compile([self.run_vectorize(nopython=True)])

    def test_concurrent_vectorize_cache(self):
        if False:
            print('Hello World!')
        self.run_compile([self.run_vectorize(nopython=True, cache=True)])

    def run_guvectorize(self, **options):
        if False:
            while True:
                i = 10

        def runner():
            if False:
                i = 10
                return i + 15
            sig = ['(f4, f4, f4[:])']
            cfunc = guvectorize(sig, '(),()->()', **options)(gufunc_foo)
            a = b = np.random.random(10).astype(np.float32)
            return cfunc(a, b)
        return runner

    def test_concurrent_guvectorize(self):
        if False:
            while True:
                i = 10
        self.run_compile([self.run_guvectorize(nopython=True)])

    def test_concurrent_guvectorize_cache(self):
        if False:
            while True:
                i = 10
        self.run_compile([self.run_guvectorize(nopython=True, cache=True)])

    def test_concurrent_mix_use(self):
        if False:
            while True:
                i = 10
        self.run_compile([self.run_jit(nopython=True, cache=True), self.run_jit(nopython=True), self.run_vectorize(nopython=True, cache=True), self.run_vectorize(nopython=True), self.run_guvectorize(nopython=True, cache=True), self.run_guvectorize(nopython=True)])
if __name__ == '__main__':
    unittest.main()