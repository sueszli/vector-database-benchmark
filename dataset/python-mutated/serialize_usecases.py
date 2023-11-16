"""
Separate module with function samples for serialization tests,
to avoid issues with __main__.
"""
import math
from numba import jit, generated_jit
from numba.core import types

@jit((types.int32, types.int32))
def add_with_sig(a, b):
    if False:
        return 10
    return a + b

@jit
def add_without_sig(a, b):
    if False:
        while True:
            i = 10
    return a + b

@jit(nopython=True)
def add_nopython(a, b):
    if False:
        for i in range(10):
            print('nop')
    return a + b

@jit(nopython=True)
def add_nopython_fail(a, b):
    if False:
        return 10
    object()
    return a + b

def closure(a):
    if False:
        return 10

    @jit(nopython=True)
    def inner(b, c):
        if False:
            return 10
        return a + b + c
    return inner
K = 3.0
from math import sqrt

def closure_with_globals(x, **jit_args):
    if False:
        while True:
            i = 10

    @jit(**jit_args)
    def inner(y):
        if False:
            return 10
        k = max(K, K + 1)
        return math.hypot(x, y) + sqrt(k)
    return inner

@jit(nopython=True)
def other_function(x, y):
    if False:
        while True:
            i = 10
    return math.hypot(x, y)

@jit(forceobj=True)
def get_global_objmode(x):
    if False:
        for i in range(10):
            print('nop')
    return K * x
import numpy as np
import numpy.random as nprand

@jit(nopython=True)
def get_renamed_module(x):
    if False:
        return 10
    nprand.seed(42)
    return (np.cos(x), nprand.random())

def closure_calling_other_function(x):
    if False:
        print('Hello World!')

    @jit(nopython=True)
    def inner(y, z):
        if False:
            return 10
        return other_function(x, y) + z
    return inner

def closure_calling_other_closure(x):
    if False:
        i = 10
        return i + 15

    @jit(nopython=True)
    def other_inner(y):
        if False:
            i = 10
            return i + 15
        return math.hypot(x, y)

    @jit(nopython=True)
    def inner(y):
        if False:
            i = 10
            return i + 15
        return other_inner(y) + x
    return inner
k1 = 5
k2 = 42

@generated_jit(nopython=True)
def generated_add(x, y):
    if False:
        return 10
    k3 = 1
    if isinstance(x, types.Complex):

        def impl(x, y):
            if False:
                print('Hello World!')
            return x + y + k1
    else:

        def impl(x, y):
            if False:
                return 10
            return x + y + k2 + k3
    return impl

def _get_dyn_func(**jit_args):
    if False:
        return 10
    code = '\n        def dyn_func(x):\n            res = 0\n            for i in range(x):\n                res += x\n            return res\n        '
    ns = {}
    exec(code.strip(), ns)
    return jit(**jit_args)(ns['dyn_func'])
dyn_func = _get_dyn_func(nopython=True)
dyn_func_objmode = _get_dyn_func(forceobj=True)