"""This module has components that use Python 3 specific syntax."""
import asyncio
import functools
from typing import Tuple

def identity(arg1, arg2: int, arg3=10, arg4: int=20, *arg5, arg6, arg7: int, arg8=30, arg9: int=40, **arg10):
    if False:
        while True:
            i = 10
    return (arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10)

class HelpTextComponent:

    def identity(self, *, alpha, beta='0'):
        if False:
            while True:
                i = 10
        return (alpha, beta)

class KeywordOnly(object):

    def double(self, *, count):
        if False:
            i = 10
            return i + 15
        return count * 2

    def triple(self, *, count):
        if False:
            return 10
        return count * 3

    def with_default(self, *, x='x'):
        if False:
            return 10
        print('x: ' + x)

class LruCacheDecoratedMethod(object):

    @functools.lru_cache()
    def lru_cache_in_class(self, arg1):
        if False:
            while True:
                i = 10
        return arg1

@functools.lru_cache()
def lru_cache_decorated(arg1):
    if False:
        print('Hello World!')
    return arg1

class WithAsyncio(object):

    @asyncio.coroutine
    def double(self, count=0):
        if False:
            i = 10
            return i + 15
        return 2 * count

class WithTypes(object):
    """Class with functions that have default arguments and types."""

    def double(self, count: float) -> float:
        if False:
            return 10
        'Returns the input multiplied by 2.\n\n    Args:\n      count: Input number that you want to double.\n\n    Returns:\n      A number that is the double of count.\n    '
        return 2 * count

    def long_type(self, long_obj: Tuple[Tuple[Tuple[Tuple[Tuple[Tuple[Tuple[Tuple[Tuple[Tuple[Tuple[Tuple[int]]]]]]]]]]]]):
        if False:
            while True:
                i = 10
        return long_obj

class WithDefaultsAndTypes(object):
    """Class with functions that have default arguments and types."""

    def double(self, count: float=0) -> float:
        if False:
            for i in range(10):
                print('nop')
        'Returns the input multiplied by 2.\n\n    Args:\n      count: Input number that you want to double.\n\n    Returns:\n      A number that is the double of count.\n    '
        return 2 * count

    def get_int(self, value: int=None):
        if False:
            while True:
                i = 10
        return 0 if value is None else value