"""@st.cache_resource implementation"""
from __future__ import annotations
import math
import threading
import types
from datetime import timedelta
from typing import Any, Callable, TypeVar, cast, overload
from cachetools import TTLCache
from typing_extensions import TypeAlias
import streamlit as st
from streamlit.deprecation_util import show_deprecation_warning
from streamlit.logger import get_logger
from streamlit.runtime.caching import cache_utils
from streamlit.runtime.caching.cache_errors import CacheKeyNotFoundError
from streamlit.runtime.caching.cache_type import CacheType
from streamlit.runtime.caching.cache_utils import Cache, CachedFuncInfo, make_cached_func_wrapper, ttl_to_seconds
from streamlit.runtime.caching.cached_message_replay import CachedMessageReplayContext, CachedResult, ElementMsgData, MsgData, MultiCacheResults
from streamlit.runtime.caching.hashing import HashFuncsDict
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx
from streamlit.runtime.stats import CacheStat, CacheStatsProvider
from streamlit.vendor.pympler.asizeof import asizeof
_LOGGER = get_logger(__name__)
CACHE_RESOURCE_MESSAGE_REPLAY_CTX = CachedMessageReplayContext(CacheType.RESOURCE)
ValidateFunc: TypeAlias = Callable[[Any], bool]

def _equal_validate_funcs(a: ValidateFunc | None, b: ValidateFunc | None) -> bool:
    if False:
        while True:
            i = 10
    'True if the two validate functions are equal for the purposes of\n    determining whether a given function cache needs to be recreated.\n    '
    return a is None and b is None or (a is not None and b is not None)

class ResourceCaches(CacheStatsProvider):
    """Manages all ResourceCache instances"""

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self._caches_lock = threading.Lock()
        self._function_caches: dict[str, ResourceCache] = {}

    def get_cache(self, key: str, display_name: str, max_entries: int | float | None, ttl: float | timedelta | str | None, validate: ValidateFunc | None, allow_widgets: bool) -> ResourceCache:
        if False:
            print('Hello World!')
        "Return the mem cache for the given key.\n\n        If it doesn't exist, create a new one with the given params.\n        "
        if max_entries is None:
            max_entries = math.inf
        ttl_seconds = ttl_to_seconds(ttl)
        with self._caches_lock:
            cache = self._function_caches.get(key)
            if cache is not None and cache.ttl_seconds == ttl_seconds and (cache.max_entries == max_entries) and _equal_validate_funcs(cache.validate, validate):
                return cache
            _LOGGER.debug('Creating new ResourceCache (key=%s)', key)
            cache = ResourceCache(key=key, display_name=display_name, max_entries=max_entries, ttl_seconds=ttl_seconds, validate=validate, allow_widgets=allow_widgets)
            self._function_caches[key] = cache
            return cache

    def clear_all(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Clear all resource caches.'
        with self._caches_lock:
            self._function_caches = {}

    def get_stats(self) -> list[CacheStat]:
        if False:
            i = 10
            return i + 15
        with self._caches_lock:
            function_caches = self._function_caches.copy()
        stats: list[CacheStat] = []
        for cache in function_caches.values():
            stats.extend(cache.get_stats())
        return stats
_resource_caches = ResourceCaches()

def get_resource_cache_stats_provider() -> CacheStatsProvider:
    if False:
        return 10
    'Return the StatsProvider for all @st.cache_resource functions.'
    return _resource_caches

class CachedResourceFuncInfo(CachedFuncInfo):
    """Implements the CachedFuncInfo interface for @st.cache_resource"""

    def __init__(self, func: types.FunctionType, show_spinner: bool | str, max_entries: int | None, ttl: float | timedelta | str | None, validate: ValidateFunc | None, allow_widgets: bool, hash_funcs: HashFuncsDict | None=None):
        if False:
            return 10
        super().__init__(func, show_spinner=show_spinner, allow_widgets=allow_widgets, hash_funcs=hash_funcs)
        self.max_entries = max_entries
        self.ttl = ttl
        self.validate = validate

    @property
    def cache_type(self) -> CacheType:
        if False:
            for i in range(10):
                print('nop')
        return CacheType.RESOURCE

    @property
    def cached_message_replay_ctx(self) -> CachedMessageReplayContext:
        if False:
            i = 10
            return i + 15
        return CACHE_RESOURCE_MESSAGE_REPLAY_CTX

    @property
    def display_name(self) -> str:
        if False:
            print('Hello World!')
        'A human-readable name for the cached function'
        return f'{self.func.__module__}.{self.func.__qualname__}'

    def get_function_cache(self, function_key: str) -> Cache:
        if False:
            for i in range(10):
                print('nop')
        return _resource_caches.get_cache(key=function_key, display_name=self.display_name, max_entries=self.max_entries, ttl=self.ttl, validate=self.validate, allow_widgets=self.allow_widgets)

class CacheResourceAPI:
    """Implements the public st.cache_resource API: the @st.cache_resource decorator,
    and st.cache_resource.clear().
    """

    def __init__(self, decorator_metric_name: str, deprecation_warning: str | None=None):
        if False:
            i = 10
            return i + 15
        "Create a CacheResourceAPI instance.\n\n        Parameters\n        ----------\n        decorator_metric_name\n            The metric name to record for decorator usage. `@st.experimental_singleton` is\n            deprecated, but we're still supporting it and tracking its usage separately\n            from `@st.cache_resource`.\n\n        deprecation_warning\n            An optional deprecation warning to show when the API is accessed.\n        "
        self._decorator = gather_metrics(decorator_metric_name, self._decorator)
        self._deprecation_warning = deprecation_warning
    F = TypeVar('F', bound=Callable[..., Any])

    @overload
    def __call__(self, func: F) -> F:
        if False:
            for i in range(10):
                print('nop')
        ...

    @overload
    def __call__(self, *, ttl: float | timedelta | str | None=None, max_entries: int | None=None, show_spinner: bool | str=True, validate: ValidateFunc | None=None, experimental_allow_widgets: bool=False, hash_funcs: HashFuncsDict | None=None) -> Callable[[F], F]:
        if False:
            print('Hello World!')
        ...

    def __call__(self, func: F | None=None, *, ttl: float | timedelta | str | None=None, max_entries: int | None=None, show_spinner: bool | str=True, validate: ValidateFunc | None=None, experimental_allow_widgets: bool=False, hash_funcs: HashFuncsDict | None=None):
        if False:
            i = 10
            return i + 15
        return self._decorator(func, ttl=ttl, max_entries=max_entries, show_spinner=show_spinner, validate=validate, experimental_allow_widgets=experimental_allow_widgets, hash_funcs=hash_funcs)

    def _decorator(self, func: F | None, *, ttl: float | timedelta | str | None, max_entries: int | None, show_spinner: bool | str, validate: ValidateFunc | None, experimental_allow_widgets: bool, hash_funcs: HashFuncsDict | None=None):
        if False:
            print('Hello World!')
        'Decorator to cache functions that return global resources (e.g. database connections, ML models).\n\n        Cached objects are shared across all users, sessions, and reruns. They\n        must be thread-safe because they can be accessed from multiple threads\n        concurrently. If thread safety is an issue, consider using ``st.session_state``\n        to store resources per session instead.\n\n        You can clear a function\'s cache with ``func.clear()`` or clear the entire\n        cache with ``st.cache_resource.clear()``.\n\n        To cache data, use ``st.cache_data`` instead. Learn more about caching at\n        https://docs.streamlit.io/library/advanced-features/caching.\n\n        Parameters\n        ----------\n        func : callable\n            The function that creates the cached resource. Streamlit hashes the\n            function\'s source code.\n\n        ttl : float, timedelta, str, or None\n            The maximum time to keep an entry in the cache. Can be one of:\n\n            * ``None`` if cache entries should never expire (default).\n            * A number specifying the time in seconds.\n            * A string specifying the time in a format supported by `Pandas\'s\n              Timedelta constructor <https://pandas.pydata.org/docs/reference/api/pandas.Timedelta.html>`_,\n              e.g. ``"1d"``, ``"1.5 days"``, or ``"1h23s"``.\n            * A ``timedelta`` object from `Python\'s built-in datetime library\n              <https://docs.python.org/3/library/datetime.html#timedelta-objects>`_,\n              e.g. ``timedelta(days=1)``.\n\n            Note that ``ttl`` will be ignored if ``persist="disk"`` or ``persist=True``.\n\n        max_entries : int or None\n            The maximum number of entries to keep in the cache, or None\n            for an unbounded cache. When a new entry is added to a full cache,\n            the oldest cached entry will be removed. Defaults to None.\n\n        show_spinner : bool or str\n            Enable the spinner. Default is True to show a spinner when there is\n            a "cache miss" and the cached resource is being created. If string,\n            value of show_spinner param will be used for spinner text.\n\n        validate : callable or None\n            An optional validation function for cached data. ``validate`` is called\n            each time the cached value is accessed. It receives the cached value as\n            its only parameter and it must return a boolean. If ``validate`` returns\n            False, the current cached value is discarded, and the decorated function\n            is called to compute a new value. This is useful e.g. to check the\n            health of database connections.\n\n        experimental_allow_widgets : bool\n            Allow widgets to be used in the cached function. Defaults to False.\n            Support for widgets in cached functions is currently experimental.\n            Setting this parameter to True may lead to excessive memory use since the\n            widget value is treated as an additional input parameter to the cache.\n            We may remove support for this option at any time without notice.\n\n        hash_funcs : dict or None\n            Mapping of types or fully qualified names to hash functions.\n            This is used to override the behavior of the hasher inside Streamlit\'s\n            caching mechanism: when the hasher encounters an object, it will first\n            check to see if its type matches a key in this dict and, if so, will use\n            the provided function to generate a hash for it. See below for an example\n            of how this can be used.\n\n        Example\n        -------\n        >>> import streamlit as st\n        >>>\n        >>> @st.cache_resource\n        ... def get_database_session(url):\n        ...     # Create a database session object that points to the URL.\n        ...     return session\n        ...\n        >>> s1 = get_database_session(SESSION_URL_1)\n        >>> # Actually executes the function, since this is the first time it was\n        >>> # encountered.\n        >>>\n        >>> s2 = get_database_session(SESSION_URL_1)\n        >>> # Does not execute the function. Instead, returns its previously computed\n        >>> # value. This means that now the connection object in s1 is the same as in s2.\n        >>>\n        >>> s3 = get_database_session(SESSION_URL_2)\n        >>> # This is a different URL, so the function executes.\n\n        By default, all parameters to a cache_resource function must be hashable.\n        Any parameter whose name begins with ``_`` will not be hashed. You can use\n        this as an "escape hatch" for parameters that are not hashable:\n\n        >>> import streamlit as st\n        >>>\n        >>> @st.cache_resource\n        ... def get_database_session(_sessionmaker, url):\n        ...     # Create a database connection object that points to the URL.\n        ...     return connection\n        ...\n        >>> s1 = get_database_session(create_sessionmaker(), DATA_URL_1)\n        >>> # Actually executes the function, since this is the first time it was\n        >>> # encountered.\n        >>>\n        >>> s2 = get_database_session(create_sessionmaker(), DATA_URL_1)\n        >>> # Does not execute the function. Instead, returns its previously computed\n        >>> # value - even though the _sessionmaker parameter was different\n        >>> # in both calls.\n\n        A cache_resource function\'s cache can be procedurally cleared:\n\n        >>> import streamlit as st\n        >>>\n        >>> @st.cache_resource\n        ... def get_database_session(_sessionmaker, url):\n        ...     # Create a database connection object that points to the URL.\n        ...     return connection\n        ...\n        >>> get_database_session.clear()\n        >>> # Clear all cached entries for this function.\n\n        To override the default hashing behavior, pass a custom hash function.\n        You can do that by mapping a type (e.g. ``Person``) to a hash\n        function (``str``) like this:\n\n        >>> import streamlit as st\n        >>> from pydantic import BaseModel\n        >>>\n        >>> class Person(BaseModel):\n        ...     name: str\n        >>>\n        >>> @st.cache_resource(hash_funcs={Person: str})\n        ... def get_person_name(person: Person):\n        ...     return person.name\n\n        Alternatively, you can map the type\'s fully-qualified name\n        (e.g. ``"__main__.Person"``) to the hash function instead:\n\n        >>> import streamlit as st\n        >>> from pydantic import BaseModel\n        >>>\n        >>> class Person(BaseModel):\n        ...     name: str\n        >>>\n        >>> @st.cache_resource(hash_funcs={"__main__.Person": str})\n        ... def get_person_name(person: Person):\n        ...     return person.name\n        '
        self._maybe_show_deprecation_warning()
        if func is None:
            return lambda f: make_cached_func_wrapper(CachedResourceFuncInfo(func=f, show_spinner=show_spinner, max_entries=max_entries, ttl=ttl, validate=validate, allow_widgets=experimental_allow_widgets, hash_funcs=hash_funcs))
        return make_cached_func_wrapper(CachedResourceFuncInfo(func=cast(types.FunctionType, func), show_spinner=show_spinner, max_entries=max_entries, ttl=ttl, validate=validate, allow_widgets=experimental_allow_widgets, hash_funcs=hash_funcs))

    @gather_metrics('clear_resource_caches')
    def clear(self) -> None:
        if False:
            while True:
                i = 10
        'Clear all cache_resource caches.'
        self._maybe_show_deprecation_warning()
        _resource_caches.clear_all()

    def _maybe_show_deprecation_warning(self):
        if False:
            i = 10
            return i + 15
        'If the API is being accessed with the deprecated `st.experimental_singleton` name,\n        show a deprecation warning.\n        '
        if self._deprecation_warning is not None:
            show_deprecation_warning(self._deprecation_warning)

class ResourceCache(Cache):
    """Manages cached values for a single st.cache_resource function."""

    def __init__(self, key: str, max_entries: float, ttl_seconds: float, validate: ValidateFunc | None, display_name: str, allow_widgets: bool):
        if False:
            return 10
        super().__init__()
        self.key = key
        self.display_name = display_name
        self._mem_cache: TTLCache[str, MultiCacheResults] = TTLCache(maxsize=max_entries, ttl=ttl_seconds, timer=cache_utils.TTLCACHE_TIMER)
        self._mem_cache_lock = threading.Lock()
        self.validate = validate
        self.allow_widgets = allow_widgets

    @property
    def max_entries(self) -> float:
        if False:
            for i in range(10):
                print('nop')
        return cast(float, self._mem_cache.maxsize)

    @property
    def ttl_seconds(self) -> float:
        if False:
            i = 10
            return i + 15
        return cast(float, self._mem_cache.ttl)

    def read_result(self, key: str) -> CachedResult:
        if False:
            while True:
                i = 10
        "Read a value and associated messages from the cache.\n        Raise `CacheKeyNotFoundError` if the value doesn't exist.\n        "
        with self._mem_cache_lock:
            if key not in self._mem_cache:
                raise CacheKeyNotFoundError()
            multi_results: MultiCacheResults = self._mem_cache[key]
            ctx = get_script_run_ctx()
            if not ctx:
                raise CacheKeyNotFoundError()
            widget_key = multi_results.get_current_widget_key(ctx, CacheType.RESOURCE)
            if widget_key not in multi_results.results:
                raise CacheKeyNotFoundError()
            result = multi_results.results[widget_key]
            if self.validate is not None and (not self.validate(result.value)):
                del multi_results.results[widget_key]
                raise CacheKeyNotFoundError()
            return result

    @gather_metrics('_cache_resource_object')
    def write_result(self, key: str, value: Any, messages: list[MsgData]) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Write a value and associated messages to the cache.'
        ctx = get_script_run_ctx()
        if ctx is None:
            return
        main_id = st._main.id
        sidebar_id = st.sidebar.id
        if self.allow_widgets:
            widgets = {msg.widget_metadata.widget_id for msg in messages if isinstance(msg, ElementMsgData) and msg.widget_metadata is not None}
        else:
            widgets = set()
        with self._mem_cache_lock:
            try:
                multi_results = self._mem_cache[key]
            except KeyError:
                multi_results = MultiCacheResults(widget_ids=widgets, results={})
            multi_results.widget_ids.update(widgets)
            widget_key = multi_results.get_current_widget_key(ctx, CacheType.RESOURCE)
            result = CachedResult(value, messages, main_id, sidebar_id)
            multi_results.results[widget_key] = result
            self._mem_cache[key] = multi_results

    def _clear(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        with self._mem_cache_lock:
            self._mem_cache.clear()

    def get_stats(self) -> list[CacheStat]:
        if False:
            print('Hello World!')
        with self._mem_cache_lock:
            cache_entries = list(self._mem_cache.values())
        return [CacheStat(category_name='st_cache_resource', cache_name=self.display_name, byte_length=asizeof(entry)) for entry in cache_entries]