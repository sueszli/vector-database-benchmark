from functools import wraps
from typing import AbstractSet, Callable, Dict, Hashable, Mapping, Tuple, Type, TypeVar
from typing_extensions import Concatenate, ParamSpec
from dagster import _check as check
S = TypeVar('S')
T = TypeVar('T')
T_Callable = TypeVar('T_Callable', bound=Callable)
P = ParamSpec('P')
NO_ARGS_HASH_VALUE = 0
CACHED_METHOD_FIELD_SUFFIX = '_cached__internal__'

def cached_method(method: Callable[Concatenate[S, P], T]) -> Callable[Concatenate[S, P], T]:
    if False:
        for i in range(10):
            print('nop')
    'Caches the results of a method call.\n\n    Usage:\n\n        .. code-block:: python\n\n            class MyClass:\n                @cached_method\n                def fetch_value_from_database(self, key):\n                    ...\n\n    The main difference between this decorator and `functools.lru_cache(max_size=None)` is that each\n    instance of the class whose method is decorated gets its own cache. This means that the cache\n    can be garbage-collected when the object is garbage-collected.\n\n    A more subtle difference between this decorator and `functools.lru_cache` is that this one\n    prioritizes preventing mistakes over ease-of-use and performance. With `functools.lru_cache`,\n    these three invocations would all result in separate cache entries:\n\n        .. code-block:: python\n\n            class MyClass:\n                @cached_method\n                def a_method(self, arg1, arg2):\n                    ...\n\n            obj = MyClass()\n            obj.a_method(arg1="a", arg2=5)\n            obj.a_method(arg2=5, arg1="a")\n            obj.a_method("a", 5)\n\n    With this decorator, the first two would point to the same cache entry, and non-kwarg arguments\n    are not allowed.\n    '
    cache_attr_name = method.__name__ + CACHED_METHOD_FIELD_SUFFIX

    @wraps(method)
    def _cached_method_wrapper(self: S, *args: P.args, **kwargs: P.kwargs) -> T:
        if False:
            print('Hello World!')
        if not hasattr(self, cache_attr_name):
            cache: Dict[Hashable, T] = {}
            setattr(self, cache_attr_name, cache)
        else:
            cache = getattr(self, cache_attr_name)
        key = _make_key(args, kwargs)
        if key not in cache:
            result = method(self, *args, **kwargs)
            cache[key] = result
        return cache[key]
    return _cached_method_wrapper

class _HashedSeq(list):
    """Adapted from https://github.com/python/cpython/blob/f9433fff476aa13af9cb314fcc6962055faa4085/Lib/functools.py#L432.

    This class guarantees that hash() will be called no more than once
    per element.  This is important because the lru_cache() will hash
    the key multiple times on a cache miss.
    """
    __slots__ = 'hashvalue'

    def __init__(self, tup: Tuple[object, ...]):
        if False:
            while True:
                i = 10
        self[:] = tup
        self.hashvalue = hash(tup)

    def __hash__(self) -> int:
        if False:
            i = 10
            return i + 15
        return self.hashvalue

def _make_key(args: Tuple[object, ...], kwds: Mapping[str, object], fasttypes: AbstractSet[Type[object]]={int, str}) -> Hashable:
    if False:
        for i in range(10):
            print('nop')
    'Adapted from https://github.com/python/cpython/blob/f9433fff476aa13af9cb314fcc6962055faa4085/Lib/functools.py#L448.\n\n    Make a cache key from optionally typed positional and keyword arguments\n    The key is constructed in a way that is flat as possible rather than\n    as a nested structure that would take more memory.\n    If there is only a single argument and its data type is known to cache\n    its hash value, then that argument is returned without a wrapper.  This\n    saves space and improves lookup speed.\n    '
    if args:
        check.failed('@cached_method does not support non-keyword arguments, because doing so would enable functionally identical sets of arguments to correspond to different cache keys.')
    if not kwds:
        return NO_ARGS_HASH_VALUE
    if len(kwds) == 1:
        (k, v) = next(iter(kwds.items()))
        if type(v) in fasttypes:
            return f'{k}.{v}'
    return _HashedSeq(tuple(sorted(kwds.items())))