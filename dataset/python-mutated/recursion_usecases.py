"""
Usecases of recursive functions.

Some functions are compiled at import time, hence a separate module.
"""
from numba import jit

@jit('i8(i8)', nopython=True)
def fib1(n):
    if False:
        while True:
            i = 10
    if n < 2:
        return n
    return fib1(n - 1) + fib1(n=n - 2)

def make_fib2():
    if False:
        for i in range(10):
            print('nop')

    @jit('i8(i8)', nopython=True)
    def fib2(n):
        if False:
            print('Hello World!')
        if n < 2:
            return n
        return fib2(n - 1) + fib2(n=n - 2)
    return fib2
fib2 = make_fib2()

def make_type_change_self(jit=lambda x: x):
    if False:
        for i in range(10):
            print('nop')

    @jit
    def type_change_self(x, y):
        if False:
            print('Hello World!')
        if x > 1 and y > 0:
            return x + type_change_self(x - y, y)
        else:
            return y
    return type_change_self

@jit(nopython=True)
def fib3(n):
    if False:
        for i in range(10):
            print('nop')
    if n < 2:
        return n
    return fib3(n - 1) + fib3(n - 2)

@jit(nopython=True)
def runaway_self(x):
    if False:
        for i in range(10):
            print('nop')
    return runaway_self(x)

@jit(nopython=True)
def raise_self(x):
    if False:
        i = 10
        return i + 15
    if x == 1:
        raise ValueError('raise_self')
    elif x > 0:
        return raise_self(x - 1)
    else:
        return 1

@jit(nopython=True)
def outer_fac(n):
    if False:
        while True:
            i = 10
    if n < 1:
        return 1
    return n * inner_fac(n - 1)

@jit(nopython=True)
def inner_fac(n):
    if False:
        return 10
    if n < 1:
        return 1
    return n * outer_fac(n - 1)

def make_mutual2(jit=lambda x: x):
    if False:
        while True:
            i = 10

    @jit
    def foo(x):
        if False:
            i = 10
            return i + 15
        if x > 0:
            return 2 * bar(z=1, y=x)
        return 1 + x

    @jit
    def bar(y, z):
        if False:
            for i in range(10):
                print('nop')
        return foo(x=y - z)
    return (foo, bar)

@jit(nopython=True)
def runaway_mutual(x):
    if False:
        print('Hello World!')
    return runaway_mutual_inner(x)

@jit(nopython=True)
def runaway_mutual_inner(x):
    if False:
        i = 10
        return i + 15
    return runaway_mutual(x)

def make_type_change_mutual(jit=lambda x: x):
    if False:
        return 10

    @jit
    def foo(x, y):
        if False:
            print('Hello World!')
        if x > 1 and y > 0:
            return x + bar(x - y, y)
        else:
            return y

    @jit
    def bar(x, y):
        if False:
            i = 10
            return i + 15
        if x > 1 and y > 0:
            return x + foo(x - y, y)
        else:
            return y
    return foo

def make_four_level(jit=lambda x: x):
    if False:
        print('Hello World!')

    @jit
    def first(x):
        if False:
            print('Hello World!')
        if x > 0:
            return second(x) * 2
        else:
            return 1

    @jit
    def second(x):
        if False:
            i = 10
            return i + 15
        return third(x) * 3

    @jit
    def third(x):
        if False:
            return 10
        return fourth(x) * 4

    @jit
    def fourth(x):
        if False:
            return 10
        return first(x / 2 - 1)
    return first

def make_inner_error(jit=lambda x: x):
    if False:
        for i in range(10):
            print('nop')

    @jit
    def outer(x):
        if False:
            for i in range(10):
                print('nop')
        if x > 0:
            return inner(x)
        else:
            return 1

    @jit
    def inner(x):
        if False:
            return 10
        if x > 0:
            return outer(x - 1)
        else:
            return error_fun(x)

    @jit
    def error_fun(x):
        if False:
            print('Hello World!')
        return x.ndim
    return outer

def make_raise_mutual(jit=lambda x: x):
    if False:
        print('Hello World!')

    @jit
    def outer(x):
        if False:
            while True:
                i = 10
        if x > 0:
            return inner(x)
        else:
            return 1

    @jit
    def inner(x):
        if False:
            return 10
        if x == 1:
            raise ValueError('raise_mutual')
        elif x > 0:
            return outer(x - 1)
        else:
            return 1
    return outer

def make_optional_return_case(jit=lambda x: x):
    if False:
        i = 10
        return i + 15

    @jit
    def foo(x):
        if False:
            while True:
                i = 10
        if x > 5:
            return x - 1
        else:
            return

    @jit
    def bar(x):
        if False:
            for i in range(10):
                print('nop')
        out = foo(x)
        if out is None:
            return out
        elif out < 8:
            return out
        else:
            return x * bar(out)
    return bar

def make_growing_tuple_case(jit=lambda x: x):
    if False:
        while True:
            i = 10

    @jit
    def make_list(n):
        if False:
            for i in range(10):
                print('nop')
        if n <= 0:
            return None
        return (n, make_list(n - 1))
    return make_list