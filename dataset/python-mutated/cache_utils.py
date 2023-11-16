"""Common cache logic shared by st.cache_data and st.cache_resource."""
from __future__ import annotations
import functools
import hashlib
import inspect
import math
import threading
import time
import types
from abc import abstractmethod
from collections import defaultdict
from datetime import timedelta
from typing import Any, Callable, overload
from typing_extensions import Literal
from streamlit import type_util
from streamlit.elements.spinner import spinner
from streamlit.logger import get_logger
from streamlit.runtime.caching.cache_errors import BadTTLStringError, CacheError, CacheKeyNotFoundError, UnevaluatedDataFrameError, UnhashableParamError, UnhashableTypeError, UnserializableReturnValueError, get_cached_func_name_md
from streamlit.runtime.caching.cache_type import CacheType
from streamlit.runtime.caching.cached_message_replay import CachedMessageReplayContext, CachedResult, MsgData, replay_cached_messages
from streamlit.runtime.caching.hashing import HashFuncsDict, update_hash
from streamlit.util import HASHLIB_KWARGS
_LOGGER = get_logger(__name__)
TTLCACHE_TIMER = time.monotonic

@overload
def ttl_to_seconds(ttl: float | timedelta | str | None, *, coerce_none_to_inf: Literal[False]) -> float | None:
    if False:
        i = 10
        return i + 15
    ...

@overload
def ttl_to_seconds(ttl: float | timedelta | str | None) -> float:
    if False:
        while True:
            i = 10
    ...

def ttl_to_seconds(ttl: float | timedelta | str | None, *, coerce_none_to_inf: bool=True) -> float | None:
    if False:
        while True:
            i = 10
    '\n    Convert a ttl value to a float representing "number of seconds".\n    '
    if coerce_none_to_inf and ttl is None:
        return math.inf
    if isinstance(ttl, timedelta):
        return ttl.total_seconds()
    if isinstance(ttl, str):
        import numpy as np
        import pandas as pd
        try:
            out: float = pd.Timedelta(ttl).total_seconds()
        except ValueError as ex:
            raise BadTTLStringError(ttl) from ex
        if np.isnan(out):
            raise BadTTLStringError(ttl)
        return out
    return ttl
UNEVALUATED_DATAFRAME_TYPES = ('snowflake.snowpark.table.Table', 'snowflake.snowpark.dataframe.DataFrame', 'pyspark.sql.dataframe.DataFrame')

class Cache:
    """Function cache interface. Caches persist across script runs."""

    def __init__(self):
        if False:
            return 10
        self._value_locks: dict[str, threading.Lock] = defaultdict(threading.Lock)
        self._value_locks_lock = threading.Lock()

    @abstractmethod
    def read_result(self, value_key: str) -> CachedResult:
        if False:
            for i in range(10):
                print('nop')
        'Read a value and associated messages from the cache.\n\n        Raises\n        ------\n        CacheKeyNotFoundError\n            Raised if value_key is not in the cache.\n\n        '
        raise NotImplementedError

    @abstractmethod
    def write_result(self, value_key: str, value: Any, messages: list[MsgData]) -> None:
        if False:
            i = 10
            return i + 15
        'Write a value and associated messages to the cache, overwriting any existing\n        result that uses the value_key.\n        '
        raise NotImplementedError

    def compute_value_lock(self, value_key: str) -> threading.Lock:
        if False:
            for i in range(10):
                print('nop')
        "Return the lock that should be held while computing a new cached value.\n        In a popular app with a cache that hasn't been pre-warmed, many sessions may try\n        to access a not-yet-cached value simultaneously. We use a lock to ensure that\n        only one of those sessions computes the value, and the others block until\n        the value is computed.\n        "
        with self._value_locks_lock:
            return self._value_locks[value_key]

    def clear(self):
        if False:
            while True:
                i = 10
        'Clear all values from this cache.'
        with self._value_locks_lock:
            self._value_locks.clear()
        self._clear()

    @abstractmethod
    def _clear(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Subclasses must implement this to perform cache-clearing logic.'
        raise NotImplementedError

class CachedFuncInfo:
    """Encapsulates data for a cached function instance.

    CachedFuncInfo instances are scoped to a single script run - they're not
    persistent.
    """

    def __init__(self, func: types.FunctionType, show_spinner: bool | str, allow_widgets: bool, hash_funcs: HashFuncsDict | None):
        if False:
            print('Hello World!')
        self.func = func
        self.show_spinner = show_spinner
        self.allow_widgets = allow_widgets
        self.hash_funcs = hash_funcs

    @property
    def cache_type(self) -> CacheType:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

    @property
    def cached_message_replay_ctx(self) -> CachedMessageReplayContext:
        if False:
            while True:
                i = 10
        raise NotImplementedError

    def get_function_cache(self, function_key: str) -> Cache:
        if False:
            while True:
                i = 10
        'Get or create the function cache for the given key.'
        raise NotImplementedError

def make_cached_func_wrapper(info: CachedFuncInfo) -> Callable[..., Any]:
    if False:
        for i in range(10):
            print('nop')
    "Create a callable wrapper around a CachedFunctionInfo.\n\n    Calling the wrapper will return the cached value if it's already been\n    computed, and will call the underlying function to compute and cache the\n    value otherwise.\n\n    The wrapper also has a `clear` function that can be called to clear\n    all of the wrapper's cached values.\n    "
    cached_func = CachedFunc(info)

    @functools.wraps(info.func)
    def wrapper(*args, **kwargs):
        if False:
            i = 10
            return i + 15
        return cached_func(*args, **kwargs)
    wrapper.clear = cached_func.clear
    return wrapper

class CachedFunc:

    def __init__(self, info: CachedFuncInfo):
        if False:
            for i in range(10):
                print('nop')
        self._info = info
        self._function_key = _make_function_key(info.cache_type, info.func)

    def __call__(self, *args, **kwargs) -> Any:
        if False:
            print('Hello World!')
        "The wrapper. We'll only call our underlying function on a cache miss."
        name = self._info.func.__qualname__
        if isinstance(self._info.show_spinner, bool):
            if len(args) == 0 and len(kwargs) == 0:
                message = f'Running `{name}()`.'
            else:
                message = f'Running `{name}(...)`.'
        else:
            message = self._info.show_spinner
        if self._info.show_spinner or isinstance(self._info.show_spinner, str):
            with spinner(message, cache=True):
                return self._get_or_create_cached_value(args, kwargs)
        else:
            return self._get_or_create_cached_value(args, kwargs)

    def _get_or_create_cached_value(self, func_args: tuple[Any, ...], func_kwargs: dict[str, Any]) -> Any:
        if False:
            return 10
        cache = self._info.get_function_cache(self._function_key)
        value_key = _make_value_key(cache_type=self._info.cache_type, func=self._info.func, func_args=func_args, func_kwargs=func_kwargs, hash_funcs=self._info.hash_funcs)
        try:
            cached_result = cache.read_result(value_key)
            return self._handle_cache_hit(cached_result)
        except CacheKeyNotFoundError:
            return self._handle_cache_miss(cache, value_key, func_args, func_kwargs)

    def _handle_cache_hit(self, result: CachedResult) -> Any:
        if False:
            i = 10
            return i + 15
        "Handle a cache hit: replay the result's cached messages, and return its value."
        replay_cached_messages(result, self._info.cache_type, self._info.func)
        return result.value

    def _handle_cache_miss(self, cache: Cache, value_key: str, func_args: tuple[Any, ...], func_kwargs: dict[str, Any]) -> Any:
        if False:
            i = 10
            return i + 15
        'Handle a cache miss: compute a new cached value, write it back to the cache,\n        and return that newly-computed value.\n        '
        with cache.compute_value_lock(value_key):
            try:
                cached_result = cache.read_result(value_key)
                return self._handle_cache_hit(cached_result)
            except CacheKeyNotFoundError:
                with self._info.cached_message_replay_ctx.calling_cached_function(self._info.func, self._info.allow_widgets):
                    computed_value = self._info.func(*func_args, **func_kwargs)
                messages = self._info.cached_message_replay_ctx._most_recent_messages
                try:
                    cache.write_result(value_key, computed_value, messages)
                    return computed_value
                except (CacheError, RuntimeError):
                    if True in [type_util.is_type(computed_value, type_name) for type_name in UNEVALUATED_DATAFRAME_TYPES]:
                        raise UnevaluatedDataFrameError(f'\n                            The function {get_cached_func_name_md(self._info.func)} is decorated with `st.cache_data` but it returns an unevaluated dataframe\n                            of type `{type_util.get_fqn_type(computed_value)}`. Please call `collect()` or `to_pandas()` on the dataframe before returning it,\n                            so `st.cache_data` can serialize and cache it.')
                    raise UnserializableReturnValueError(return_value=computed_value, func=self._info.func)

    def clear(self):
        if False:
            i = 10
            return i + 15
        "Clear the wrapped function's associated cache."
        cache = self._info.get_function_cache(self._function_key)
        cache.clear()

def _make_value_key(cache_type: CacheType, func: types.FunctionType, func_args: tuple[Any, ...], func_kwargs: dict[str, Any], hash_funcs: HashFuncsDict | None) -> str:
    if False:
        while True:
            i = 10
    'Create the key for a value within a cache.\n\n    This key is generated from the function\'s arguments. All arguments\n    will be hashed, except for those named with a leading "_".\n\n    Raises\n    ------\n    StreamlitAPIException\n        Raised (with a nicely-formatted explanation message) if we encounter\n        an un-hashable arg.\n    '
    arg_pairs: list[tuple[str | None, Any]] = []
    for arg_idx in range(len(func_args)):
        arg_name = _get_positional_arg_name(func, arg_idx)
        arg_pairs.append((arg_name, func_args[arg_idx]))
    for (kw_name, kw_val) in func_kwargs.items():
        arg_pairs.append((kw_name, kw_val))
    args_hasher = hashlib.new('md5', **HASHLIB_KWARGS)
    for (arg_name, arg_value) in arg_pairs:
        if arg_name is not None and arg_name.startswith('_'):
            _LOGGER.debug('Not hashing %s because it starts with _', arg_name)
            continue
        try:
            update_hash(arg_name, hasher=args_hasher, cache_type=cache_type, hash_source=func)
            update_hash(arg_value, hasher=args_hasher, cache_type=cache_type, hash_funcs=hash_funcs, hash_source=func)
        except UnhashableTypeError as exc:
            raise UnhashableParamError(cache_type, func, arg_name, arg_value, exc)
    value_key = args_hasher.hexdigest()
    _LOGGER.debug('Cache key: %s', value_key)
    return value_key

def _make_function_key(cache_type: CacheType, func: types.FunctionType) -> str:
    if False:
        return 10
    "Create the unique key for a function's cache.\n\n    A function's key is stable across reruns of the app, and changes when\n    the function's source code changes.\n    "
    func_hasher = hashlib.new('md5', **HASHLIB_KWARGS)
    update_hash((func.__module__, func.__qualname__), hasher=func_hasher, cache_type=cache_type, hash_source=func)
    source_code: str | bytes
    try:
        source_code = inspect.getsource(func)
    except OSError as e:
        _LOGGER.debug("Failed to retrieve function's source code when building its key; falling back to bytecode. err={0}", e)
        source_code = func.__code__.co_code
    update_hash(source_code, hasher=func_hasher, cache_type=cache_type, hash_source=func)
    cache_key = func_hasher.hexdigest()
    return cache_key

def _get_positional_arg_name(func: types.FunctionType, arg_index: int) -> str | None:
    if False:
        print('Hello World!')
    "Return the name of a function's positional argument.\n\n    If arg_index is out of range, or refers to a parameter that is not a\n    named positional argument (e.g. an *args, **kwargs, or keyword-only param),\n    return None instead.\n    "
    if arg_index < 0:
        return None
    params: list[inspect.Parameter] = list(inspect.signature(func).parameters.values())
    if arg_index >= len(params):
        return None
    if params[arg_index].kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.POSITIONAL_ONLY):
        return params[arg_index].name
    return None