"""Function caching"""
import os
from typing import Any, Callable, TypeVar
from joblib import Memory
from decorator import FunctionMaker

def _decorator_apply(dec, func):
    if False:
        i = 10
        return i + 15
    return FunctionMaker.create(func, 'return decfunc(%(shortsignature)s)', dict(decfunc=dec(func)), __wrapped__=func)
_F = TypeVar('_F', bound=Callable[..., Any])

class CacheManager(object):
    """The librosa cache manager class wraps joblib.Memory
    with a __call__ attribute, so that it may act as a function.

    Additionally, it provides a caching level filter, so that
    different functions can be cached or not depending on the user's
    preference for speed vs. storage usage.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        if False:
            return 10
        level = kwargs.pop('level', 10)
        self.memory: Memory = Memory(*args, **kwargs)
        self.level: int = level

    def __call__(self, level: int) -> Callable[[_F], _F]:
        if False:
            while True:
                i = 10
        '\n        Cache with an explicitly defined level.\n\n        Example usage:\n\n        @cache(level=2)\n        def semi_important_function(some_arguments):\n            ...\n        '

        def wrapper(function):
            if False:
                for i in range(10):
                    print('nop')
            'Add an input/output cache to the specified function.'
            if self.memory.location is not None and self.level >= level:
                return _decorator_apply(self.memory.cache, function)
            else:
                return function
        return wrapper

    def clear(self, *args: Any, **kwargs: Any) -> None:
        if False:
            while True:
                i = 10
        'Clear the cache'
        self.memory.clear(*args, **kwargs)

    def eval(self, *args: Any, **kwargs: Any) -> Any:
        if False:
            i = 10
            return i + 15
        'Evaluate a function'
        return self.memory.eval(*args, **kwargs)

    def format(self, *args: Any, **kwargs: Any) -> Any:
        if False:
            return 10
        'Return the formatted representation of an object'
        return self.memory.format(*args, **kwargs)

    def reduce_size(self, *args: Any, **kwargs: Any) -> None:
        if False:
            while True:
                i = 10
        'Reduce the size of the cache'
        self.memory.reduce_size(*args, **kwargs)

    def warn(self, *args: Any, **kwargs: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Raise a warning'
        self.memory.warn(*args, **kwargs)
cache: CacheManager = CacheManager(os.environ.get('LIBROSA_CACHE_DIR', None), mmap_mode=os.environ.get('LIBROSA_CACHE_MMAP', None), compress=os.environ.get('LIBROSA_CACHE_COMPRESS', False), verbose=int(os.environ.get('LIBROSA_CACHE_VERBOSE', 0)), level=int(os.environ.get('LIBROSA_CACHE_LEVEL', 10)))