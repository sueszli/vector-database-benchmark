from ctypes import *
import sys
import numpy as np
is_windows = sys.platform.startswith('win32')
from numba import _helperlib
libnumba = CDLL(_helperlib.__file__)
del _helperlib
c_sin = libnumba._numba_test_sin
c_sin.argtypes = [c_double]
c_sin.restype = c_double

def use_c_sin(x):
    if False:
        return 10
    return c_sin(x)
c_cos = libnumba._numba_test_cos
c_cos.argtypes = [c_double]
c_cos.restype = c_double

def use_two_funcs(x):
    if False:
        while True:
            i = 10
    return c_sin(x) - c_cos(x)
c_vsquare = libnumba._numba_test_vsquare
c_vsquare.argtypes = [c_int, c_void_p, c_void_p]
c_vcube = libnumba._numba_test_vsquare
c_vcube.argtypes = [c_int, POINTER(c_double), POINTER(c_double)]

def use_c_vsquare(x):
    if False:
        return 10
    out = np.empty_like(x)
    c_vsquare(x.size, x.ctypes, out.ctypes)
    return out

def use_c_vcube(x):
    if False:
        print('Hello World!')
    out = np.empty_like(x)
    c_vcube(x.size, x.ctypes, out.ctypes)
    return out
c_untyped = libnumba._numba_test_exp

def use_c_untyped(x):
    if False:
        print('Hello World!')
    return c_untyped(x)
ctype_wrapping = CFUNCTYPE(c_double, c_double)(use_c_sin)

def use_ctype_wrapping(x):
    if False:
        for i in range(10):
            print('nop')
    return ctype_wrapping(x)
savethread = pythonapi.PyEval_SaveThread
savethread.argtypes = []
savethread.restype = c_void_p
restorethread = pythonapi.PyEval_RestoreThread
restorethread.argtypes = [c_void_p]
restorethread.restype = None
if is_windows:
    c_sleep = windll.kernel32.Sleep
    c_sleep.argtypes = [c_uint]
    c_sleep.restype = None

    def use_c_sleep(x):
        if False:
            i = 10
            return i + 15
        c_sleep(x)

def use_c_pointer(x):
    if False:
        i = 10
        return i + 15
    '\n    Running in Python will cause a segfault.\n    '
    threadstate = savethread()
    x += 1
    restorethread(threadstate)
    return x

def use_func_pointer(fa, fb, x):
    if False:
        print('Hello World!')
    if x > 0:
        return fa(x)
    else:
        return fb(x)
mydct = {'what': 1232121}

def call_me_maybe(arr):
    if False:
        return 10
    return mydct[arr[0].decode('ascii')]
py_call_back = CFUNCTYPE(c_int, py_object)(call_me_maybe)

def take_array_ptr(ptr):
    if False:
        for i in range(10):
            print('nop')
    return ptr
c_take_array_ptr = CFUNCTYPE(c_void_p, c_void_p)(take_array_ptr)