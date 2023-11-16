"""
Should emit:
B019 - on lines 73, 77, 81, 85, 89, 93, 97, 101
"""
import functools
from functools import cache, cached_property, lru_cache

def some_other_cache():
    if False:
        i = 10
        return i + 15
    ...

@functools.cache
def compute_func(self, y):
    if False:
        for i in range(10):
            print('nop')
    ...

class Foo:

    def __init__(self, x):
        if False:
            for i in range(10):
                print('nop')
        self.x = x

    def compute_method(self, y):
        if False:
            for i in range(10):
                print('nop')
        ...

    @some_other_cache
    def user_cached_instance_method(self, y):
        if False:
            i = 10
            return i + 15
        ...

    @classmethod
    @functools.cache
    def cached_classmethod(cls, y):
        if False:
            i = 10
            return i + 15
        ...

    @classmethod
    @cache
    def other_cached_classmethod(cls, y):
        if False:
            return 10
        ...

    @classmethod
    @functools.lru_cache
    def lru_cached_classmethod(cls, y):
        if False:
            i = 10
            return i + 15
        ...

    @classmethod
    @lru_cache
    def other_lru_cached_classmethod(cls, y):
        if False:
            print('Hello World!')
        ...

    @staticmethod
    @functools.cache
    def cached_staticmethod(y):
        if False:
            return 10
        ...

    @staticmethod
    @cache
    def other_cached_staticmethod(y):
        if False:
            print('Hello World!')
        ...

    @staticmethod
    @functools.lru_cache
    def lru_cached_staticmethod(y):
        if False:
            i = 10
            return i + 15
        ...

    @staticmethod
    @lru_cache
    def other_lru_cached_staticmethod(y):
        if False:
            while True:
                i = 10
        ...

    @functools.cached_property
    def some_cached_property(self):
        if False:
            print('Hello World!')
        ...

    @cached_property
    def some_other_cached_property(self):
        if False:
            print('Hello World!')
        ...

    @functools.cache
    def cached_instance_method(self, y):
        if False:
            return 10
        ...

    @cache
    def another_cached_instance_method(self, y):
        if False:
            i = 10
            return i + 15
        ...

    @functools.cache()
    def called_cached_instance_method(self, y):
        if False:
            print('Hello World!')
        ...

    @cache()
    def another_called_cached_instance_method(self, y):
        if False:
            while True:
                i = 10
        ...

    @functools.lru_cache
    def lru_cached_instance_method(self, y):
        if False:
            return 10
        ...

    @lru_cache
    def another_lru_cached_instance_method(self, y):
        if False:
            i = 10
            return i + 15
        ...

    @functools.lru_cache()
    def called_lru_cached_instance_method(self, y):
        if False:
            while True:
                i = 10
        ...

    @lru_cache()
    def another_called_lru_cached_instance_method(self, y):
        if False:
            while True:
                i = 10
        ...