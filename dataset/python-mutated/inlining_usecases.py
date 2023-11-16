""" Test cases for inlining IR from another module """
from numba import njit
from numba.core.extending import overload
_GLOBAL1 = 100

@njit(inline='always')
def bar():
    if False:
        print('Hello World!')
    return _GLOBAL1 + 10

def baz_factory(a):
    if False:
        return 10
    b = 17 + a

    @njit(inline='always')
    def baz():
        if False:
            print('Hello World!')
        return _GLOBAL1 + a - b
    return baz

def baz():
    if False:
        while True:
            i = 10
    return _GLOBAL1 + 10

@overload(baz, inline='always')
def baz_ol():
    if False:
        return 10

    def impl():
        if False:
            return 10
        return _GLOBAL1 + 10
    return impl

def bop_factory(a):
    if False:
        return 10
    b = 17 + a

    def bop():
        if False:
            i = 10
            return i + 15
        return _GLOBAL1 + a - b

    @overload(bop, inline='always')
    def baz():
        if False:
            while True:
                i = 10

        def impl():
            if False:
                while True:
                    i = 10
            return _GLOBAL1 + a - b
        return impl
    return bop