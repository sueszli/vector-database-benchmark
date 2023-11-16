"""
Copyright 2013 Steven Diamond

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import functools
from typing import Callable, TypeVar
from cvxpy.utilities import scopes
R = TypeVar('R')
T = TypeVar('T')

def lazyprop(func):
    if False:
        while True:
            i = 10
    'Wraps a property so it is lazily evaluated.'

    @property
    @functools.wraps(func)
    def _lazyprop(self):
        if False:
            i = 10
            return i + 15
        if scopes.dpp_scope_active():
            attr_name = '_lazy_dpp_' + func.__name__
        else:
            attr_name = '_lazy_' + func.__name__
        try:
            return getattr(self, attr_name)
        except AttributeError:
            setattr(self, attr_name, func(self))
        return getattr(self, attr_name)
    return _lazyprop

def _cache_key(args, kwargs):
    if False:
        return 10
    key = args + tuple(list(kwargs.items()))
    if scopes.dpp_scope_active():
        key = ('__dpp_scope_active__',) + key
    return key

def compute_once(func: Callable[[T], R]) -> Callable[[T], R]:
    if False:
        for i in range(10):
            print('nop')
    'Computes an instance method caches the result.\n\n    A result is stored for each unique combination of arguments and\n    keyword arguments. Similar to functools.lru_cache, except this works\n    decorator works for instance methods (functools.lru_cache decorates\n    functions, not methods; using it on a method leaks memory.)\n\n    This decorator should not be used when there are an unbounded or very\n    large number of argument and keyword argument combinations.\n     '

    @functools.wraps(func)
    def _compute_once(self, *args, **kwargs) -> R:
        if False:
            print('Hello World!')
        cache_name = func.__name__ + '__cache__'
        if not hasattr(self, cache_name):
            setattr(self, cache_name, {})
        cache = getattr(self, cache_name)
        key = _cache_key(args, kwargs)
        if key in cache:
            return cache[key]
        result = func(self, *args, **kwargs)
        cache[key] = result
        return result
    return _compute_once