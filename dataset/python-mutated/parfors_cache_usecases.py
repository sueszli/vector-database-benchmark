import sys
import numpy as np
from numba import njit
from numba.tests.support import TestCase

@njit(parallel=True, cache=True)
def arrayexprs_case(arr):
    if False:
        return 10
    return arr / arr.sum()

@njit(parallel=True, cache=True)
def prange_case(arr):
    if False:
        while True:
            i = 10
    out = np.zeros_like(arr)
    c = 1 / arr.sum()
    for i in range(arr.size):
        out[i] = arr[i] * c
    return out

@njit(cache=True)
def caller_case(arr):
    if False:
        while True:
            i = 10
    return prange_case(arrayexprs_case(arr))

class _TestModule(TestCase):
    """
    Tests for functionality of this module's functions.
    Note this does not define any "test_*" method, instead check_module()
    should be called by hand.
    """

    def check_module(self, mod):
        if False:
            print('Hello World!')
        total_cache_hits = 0
        for fn in [mod.arrayexprs_case, mod.prange_case, mod.caller_case]:
            arr = np.ones(20)
            np.testing.assert_allclose(fn(arr), fn.py_func(arr))
            total_cache_hits += len(fn.stats.cache_hits)
        self.assertGreater(total_cache_hits, 0, msg='At least one dispatcher has used the cache')

    def run_module(self, mod):
        if False:
            for i in range(10):
                print('nop')
        for fn in [mod.arrayexprs_case, mod.prange_case, mod.caller_case]:
            arr = np.ones(20)
            np.testing.assert_allclose(fn(arr), fn.py_func(arr))

def self_test():
    if False:
        print('Hello World!')
    mod = sys.modules[__name__]
    _TestModule().check_module(mod)

def self_run():
    if False:
        i = 10
        return i + 15
    mod = sys.modules[__name__]
    _TestModule().run_module(mod)