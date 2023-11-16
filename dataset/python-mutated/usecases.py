import math
import numpy as np
from numba import jit
_GLOBAL_STR = 'abc'

def sum1d(s, e):
    if False:
        print('Hello World!')
    c = 0
    for i in range(s, e):
        c += i
    return c

def sum2d(s, e):
    if False:
        return 10
    c = 0
    for i in range(s, e):
        for j in range(s, e):
            c += i * j
    return c

def while_count(s, e):
    if False:
        print('Hello World!')
    i = s
    c = 0
    while i < e:
        c += i
        i += 1
    return c

def copy_arrays(a, b):
    if False:
        for i in range(10):
            print('nop')
    for i in range(a.shape[0]):
        b[i] = a[i]

def copy_arrays2d(a, b):
    if False:
        while True:
            i = 10
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            b[i, j] = a[i, j]

def redefine1():
    if False:
        return 10
    x = 0
    for i in range(5):
        x += 1
    x = 0.0 + x
    for i in range(5):
        x += 1
    return x

def andor(x, y):
    if False:
        for i in range(10):
            print('nop')
    return x > 0 and x < 10 or (y > 0 and y < 10)
andornopython = jit(nopython=True)(andor)

def string_concat(x, y):
    if False:
        while True:
            i = 10
    a = 'whatzup'
    return a + str(x + y)

def string_len(s):
    if False:
        i = 10
        return i + 15
    return len(s)

def string_slicing(s, start, stop):
    if False:
        return 10
    return s[start:stop]

def string_conversion(x):
    if False:
        for i in range(10):
            print('nop')
    object()
    return str(x)

def string_comparison(s1, s2, op):
    if False:
        for i in range(10):
            print('nop')
    return op(s1, s2)

def blackscholes_cnd(d):
    if False:
        while True:
            i = 10
    A1 = 0.31938153
    A2 = -0.356563782
    A3 = 1.781477937
    A4 = -1.821255978
    A5 = 1.330274429
    RSQRT2PI = 0.3989422804014327
    K = 1.0 / (1.0 + 0.2316419 * math.fabs(d))
    ret_val = RSQRT2PI * math.exp(-0.5 * d * d) * (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))))
    if d > 0:
        ret_val = 1.0 - ret_val
    return ret_val