"""@st.cache_data: pickle-based caching"""
from __future__ import annotations
import pickle
import threading
import types
from datetime import timedelta
from typing import Any, Callable, TypeVar, Union, cast, overload
from typing_extensions import Literal, TypeAlias
import streamlit as st
from streamlit import runtime
from streamlit.deprecation_util import show_deprecation_warning
from streamlit.errors import StreamlitAPIException
from streamlit.logger import get_logger
from streamlit.runtime.caching.cache_errors import CacheError, CacheKeyNotFoundError
from streamlit.runtime.caching.cache_type import CacheType
from streamlit.runtime.caching.cache_utils import Cache, CachedFuncInfo, make_cached_func_wrapper, ttl_to_seconds
from streamlit.runtime.caching.cached_message_replay import CachedMessageReplayContext, CachedResult, ElementMsgData, MsgData, MultiCacheResults
from streamlit.runtime.caching.hashing import HashFuncsDict
from streamlit.runtime.caching.storage import CacheStorage, CacheStorageContext, CacheStorageError, CacheStorageKeyNotFoundError, CacheStorageManager
from streamlit.runtime.caching.storage.cache_storage_protocol import InvalidCacheStorageContext
from streamlit.runtime.caching.storage.dummy_cache_storage import MemoryCacheStorageManager
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx
from streamlit.runtime.stats import CacheStat, CacheStatsProvider
_LOGGER = get_logger(__name__)
CACHE_DATA_MESSAGE_REPLAY_CTX = CachedMessageReplayContext(CacheType.DATA)
CachePersistType: TypeAlias = Union[Literal['disk'], None]

class CachedDataFuncInfo(CachedFuncInfo):
    """Implements the CachedFuncInfo interface for @st.cache_data"""

    def __init__(self, func: types.FunctionType, show_spinner: bool | str, persist: CachePersistType, max_entries: int | None, ttl: float | timedelta | str | None, allow_widgets: bool, hash_funcs: HashFuncsDict | None=None):
        if False:
            i = 10
            return i + 15
        super().__init__(func, show_spinner=show_spinner, allow_widgets=allow_widgets, hash_funcs=hash_funcs)
        self.persist = persist
        self.max_entries = max_entries
        self.ttl = ttl
        self.validate_params()

    @property
    def cache_type(self) -> CacheType:
        if False:
            return 10
        return CacheType.DATA

    @property
    def cached_message_replay_ctx(self) -> CachedMessageReplayContext:
        if False:
            i = 10
            return i + 15
        return CACHE_DATA_MESSAGE_REPLAY_CTX

    @property
    def display_name(self) -> str:
        if False:
            print('Hello World!')
        'A human-readable name for the cached function'
        return f'{self.func.__module__}.{self.func.__qualname__}'

    def get_function_cache(self, function_key: str) -> Cache:
        if False:
            while True:
                i = 10
        return _data_caches.get_cache(key=function_key, persist=self.persist, max_entries=self.max_entries, ttl=self.ttl, display_name=self.display_name, allow_widgets=self.allow_widgets)

    def validate_params(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Validate the params passed to @st.cache_data are compatible with cache storage\n\n        When called, this method could log warnings if cache params are invalid\n        for current storage.\n        '
        _data_caches.validate_cache_params(function_name=self.func.__name__, persist=self.persist, max_entries=self.max_entries, ttl=self.ttl)

class DataCaches(CacheStatsProvider):
    """Manages all DataCache instances"""

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self._caches_lock = threading.Lock()
        self._function_caches: dict[str, DataCache] = {}

    def get_cache(self, key: str, persist: CachePersistType, max_entries: int | None, ttl: int | float | timedelta | str | None, display_name: str, allow_widgets: bool) -> DataCache:
        if False:
            while True:
                i = 10
        "Return the mem cache for the given key.\n\n        If it doesn't exist, create a new one with the given params.\n        "
        ttl_seconds = ttl_to_seconds(ttl, coerce_none_to_inf=False)
        with self._caches_lock:
            cache = self._function_caches.get(key)
            if cache is not None and cache.ttl_seconds == ttl_seconds and (cache.max_entries == max_entries) and (cache.persist == persist):
                return cache
            if cache is not None:
                _LOGGER.debug('Closing existing DataCache storage (key=%s, persist=%s, max_entries=%s, ttl=%s) before creating new one with different params', key, persist, max_entries, ttl)
                cache.storage.close()
            _LOGGER.debug('Creating new DataCache (key=%s, persist=%s, max_entries=%s, ttl=%s)', key, persist, max_entries, ttl)
            cache_context = self.create_cache_storage_context(function_key=key, function_name=display_name, ttl_seconds=ttl_seconds, max_entries=max_entries, persist=persist)
            cache_storage_manager = self.get_storage_manager()
            storage = cache_storage_manager.create(cache_context)
            cache = DataCache(key=key, storage=storage, persist=persist, max_entries=max_entries, ttl_seconds=ttl_seconds, display_name=display_name, allow_widgets=allow_widgets)
            self._function_caches[key] = cache
            return cache

    def clear_all(self) -> None:
        if False:
            i = 10
            return i + 15
        'Clear all in-memory and on-disk caches.'
        with self._caches_lock:
            try:
                self.get_storage_manager().clear_all()
            except NotImplementedError:
                for data_cache in self._function_caches.values():
                    data_cache.clear()
                    data_cache.storage.close()
            self._function_caches = {}

    def get_stats(self) -> list[CacheStat]:
        if False:
            for i in range(10):
                print('nop')
        with self._caches_lock:
            function_caches = self._function_caches.copy()
        stats: list[CacheStat] = []
        for cache in function_caches.values():
            stats.extend(cache.get_stats())
        return stats

    def validate_cache_params(self, function_name: str, persist: CachePersistType, max_entries: int | None, ttl: int | float | timedelta | str | None) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Validate that the cache params are valid for given storage.\n\n        Raises\n        ------\n        InvalidCacheStorageContext\n            Raised if the cache storage manager is not able to work with provided\n            CacheStorageContext.\n        '
        ttl_seconds = ttl_to_seconds(ttl, coerce_none_to_inf=False)
        cache_context = self.create_cache_storage_context(function_key='DUMMY_KEY', function_name=function_name, ttl_seconds=ttl_seconds, max_entries=max_entries, persist=persist)
        try:
            self.get_storage_manager().check_context(cache_context)
        except InvalidCacheStorageContext as e:
            _LOGGER.error('Cache params for function %s are incompatible with current cache storage manager: %s', function_name, e)
            raise

    def create_cache_storage_context(self, function_key: str, function_name: str, persist: CachePersistType, ttl_seconds: float | None, max_entries: int | None) -> CacheStorageContext:
        if False:
            i = 10
            return i + 15
        return CacheStorageContext(function_key=function_key, function_display_name=function_name, ttl_seconds=ttl_seconds, max_entries=max_entries, persist=persist)

    def get_storage_manager(self) -> CacheStorageManager:
        if False:
            while True:
                i = 10
        if runtime.exists():
            return runtime.get_instance().cache_storage_manager
        else:
            _LOGGER.warning('No runtime found, using MemoryCacheStorageManager')
            return MemoryCacheStorageManager()
_data_caches = DataCaches()

def get_data_cache_stats_provider() -> CacheStatsProvider:
    if False:
        for i in range(10):
            print('nop')
    'Return the StatsProvider for all @st.cache_data functions.'
    return _data_caches

class CacheDataAPI:
    """Implements the public st.cache_data API: the @st.cache_data decorator, and
    st.cache_data.clear().
    """

    def __init__(self, decorator_metric_name: str, deprecation_warning: str | None=None):
        if False:
            i = 10
            return i + 15
        "Create a CacheDataAPI instance.\n\n        Parameters\n        ----------\n        decorator_metric_name\n            The metric name to record for decorator usage. `@st.experimental_memo` is\n            deprecated, but we're still supporting it and tracking its usage separately\n            from `@st.cache_data`.\n\n        deprecation_warning\n            An optional deprecation warning to show when the API is accessed.\n        "
        self._decorator = gather_metrics(decorator_metric_name, self._decorator)
        self._deprecation_warning = deprecation_warning
    F = TypeVar('F', bound=Callable[..., Any])

    @overload
    def __call__(self, func: F) -> F:
        if False:
            return 10
        ...

    @overload
    def __call__(self, *, ttl: float | timedelta | str | None=None, max_entries: int | None=None, show_spinner: bool | str=True, persist: CachePersistType | bool=None, experimental_allow_widgets: bool=False, hash_funcs: HashFuncsDict | None=None) -> Callable[[F], F]:
        if False:
            for i in range(10):
                print('nop')
        ...

    def __call__(self, func: F | None=None, *, ttl: float | timedelta | str | None=None, max_entries: int | None=None, show_spinner: bool | str=True, persist: CachePersistType | bool=None, experimental_allow_widgets: bool=False, hash_funcs: HashFuncsDict | None=None):
        if False:
            return 10
        return self._decorator(func, ttl=ttl, max_entries=max_entries, persist=persist, show_spinner=show_spinner, experimental_allow_widgets=experimental_allow_widgets, hash_funcs=hash_funcs)

    def _decorator(self, func: F | None=None, *, ttl: float | timedelta | str | None, max_entries: int | None, show_spinner: bool | str, persist: CachePersistType | bool, experimental_allow_widgets: bool, hash_funcs: HashFuncsDict | None=None):
        if False:
            print('Hello World!')
        'Decorator to cache functions that return data (e.g. dataframe transforms, database queries, ML inference).\n\n        Cached objects are stored in "pickled" form, which means that the return\n        value of a cached function must be pickleable. Each caller of the cached\n        function gets its own copy of the cached data.\n\n        You can clear a function\'s cache with ``func.clear()`` or clear the entire\n        cache with ``st.cache_data.clear()``.\n\n        To cache global resources, use ``st.cache_resource`` instead. Learn more\n        about caching at https://docs.streamlit.io/library/advanced-features/caching.\n\n        Parameters\n        ----------\n        func : callable\n            The function to cache. Streamlit hashes the function\'s source code.\n\n        ttl : float, timedelta, str, or None\n            The maximum time to keep an entry in the cache. Can be one of:\n\n            * ``None`` if cache entries should never expire (default).\n            * A number specifying the time in seconds.\n            * A string specifying the time in a format supported by `Pandas\'s\n              Timedelta constructor <https://pandas.pydata.org/docs/reference/api/pandas.Timedelta.html>`_,\n              e.g. ``"1d"``, ``"1.5 days"``, or ``"1h23s"``.\n            * A ``timedelta`` object from `Python\'s built-in datetime library\n              <https://docs.python.org/3/library/datetime.html#timedelta-objects>`_,\n              e.g. ``timedelta(days=1)``.\n\n            Note that ``ttl`` will be ignored if ``persist="disk"`` or ``persist=True``.\n\n        max_entries : int or None\n            The maximum number of entries to keep in the cache, or None\n            for an unbounded cache. When a new entry is added to a full cache,\n            the oldest cached entry will be removed. Defaults to None.\n\n        show_spinner : bool or str\n            Enable the spinner. Default is True to show a spinner when there is\n            a "cache miss" and the cached data is being created. If string,\n            value of show_spinner param will be used for spinner text.\n\n        persist : "disk", bool, or None\n            Optional location to persist cached data to. Passing "disk" (or True)\n            will persist the cached data to the local disk. None (or False) will disable\n            persistence. The default is None.\n\n        experimental_allow_widgets : bool\n            Allow widgets to be used in the cached function. Defaults to False.\n            Support for widgets in cached functions is currently experimental.\n            Setting this parameter to True may lead to excessive memory use since the\n            widget value is treated as an additional input parameter to the cache.\n            We may remove support for this option at any time without notice.\n\n        hash_funcs : dict or None\n            Mapping of types or fully qualified names to hash functions.\n            This is used to override the behavior of the hasher inside Streamlit\'s\n            caching mechanism: when the hasher encounters an object, it will first\n            check to see if its type matches a key in this dict and, if so, will use\n            the provided function to generate a hash for it. See below for an example\n            of how this can be used.\n\n        Example\n        -------\n        >>> import streamlit as st\n        >>>\n        >>> @st.cache_data\n        ... def fetch_and_clean_data(url):\n        ...     # Fetch data from URL here, and then clean it up.\n        ...     return data\n        ...\n        >>> d1 = fetch_and_clean_data(DATA_URL_1)\n        >>> # Actually executes the function, since this is the first time it was\n        >>> # encountered.\n        >>>\n        >>> d2 = fetch_and_clean_data(DATA_URL_1)\n        >>> # Does not execute the function. Instead, returns its previously computed\n        >>> # value. This means that now the data in d1 is the same as in d2.\n        >>>\n        >>> d3 = fetch_and_clean_data(DATA_URL_2)\n        >>> # This is a different URL, so the function executes.\n\n        To set the ``persist`` parameter, use this command as follows:\n\n        >>> import streamlit as st\n        >>>\n        >>> @st.cache_data(persist="disk")\n        ... def fetch_and_clean_data(url):\n        ...     # Fetch data from URL here, and then clean it up.\n        ...     return data\n\n        By default, all parameters to a cached function must be hashable.\n        Any parameter whose name begins with ``_`` will not be hashed. You can use\n        this as an "escape hatch" for parameters that are not hashable:\n\n        >>> import streamlit as st\n        >>>\n        >>> @st.cache_data\n        ... def fetch_and_clean_data(_db_connection, num_rows):\n        ...     # Fetch data from _db_connection here, and then clean it up.\n        ...     return data\n        ...\n        >>> connection = make_database_connection()\n        >>> d1 = fetch_and_clean_data(connection, num_rows=10)\n        >>> # Actually executes the function, since this is the first time it was\n        >>> # encountered.\n        >>>\n        >>> another_connection = make_database_connection()\n        >>> d2 = fetch_and_clean_data(another_connection, num_rows=10)\n        >>> # Does not execute the function. Instead, returns its previously computed\n        >>> # value - even though the _database_connection parameter was different\n        >>> # in both calls.\n\n        A cached function\'s cache can be procedurally cleared:\n\n        >>> import streamlit as st\n        >>>\n        >>> @st.cache_data\n        ... def fetch_and_clean_data(_db_connection, num_rows):\n        ...     # Fetch data from _db_connection here, and then clean it up.\n        ...     return data\n        ...\n        >>> fetch_and_clean_data.clear()\n        >>> # Clear all cached entries for this function.\n\n        To override the default hashing behavior, pass a custom hash function.\n        You can do that by mapping a type (e.g. ``datetime.datetime``) to a hash\n        function (``lambda dt: dt.isoformat()``) like this:\n\n        >>> import streamlit as st\n        >>> import datetime\n        >>>\n        >>> @st.cache_data(hash_funcs={datetime.datetime: lambda dt: dt.isoformat()})\n        ... def convert_to_utc(dt: datetime.datetime):\n        ...     return dt.astimezone(datetime.timezone.utc)\n\n        Alternatively, you can map the type\'s fully-qualified name\n        (e.g. ``"datetime.datetime"``) to the hash function instead:\n\n        >>> import streamlit as st\n        >>> import datetime\n        >>>\n        >>> @st.cache_data(hash_funcs={"datetime.datetime": lambda dt: dt.isoformat()})\n        ... def convert_to_utc(dt: datetime.datetime):\n        ...     return dt.astimezone(datetime.timezone.utc)\n\n        '
        persist_string: CachePersistType
        if persist is True:
            persist_string = 'disk'
        elif persist is False:
            persist_string = None
        else:
            persist_string = persist
        if persist_string not in (None, 'disk'):
            raise StreamlitAPIException(f"Unsupported persist option '{persist}'. Valid values are 'disk' or None.")
        self._maybe_show_deprecation_warning()

        def wrapper(f):
            if False:
                return 10
            return make_cached_func_wrapper(CachedDataFuncInfo(func=f, persist=persist_string, show_spinner=show_spinner, max_entries=max_entries, ttl=ttl, allow_widgets=experimental_allow_widgets, hash_funcs=hash_funcs))
        if func is None:
            return wrapper
        return make_cached_func_wrapper(CachedDataFuncInfo(func=cast(types.FunctionType, func), persist=persist_string, show_spinner=show_spinner, max_entries=max_entries, ttl=ttl, allow_widgets=experimental_allow_widgets, hash_funcs=hash_funcs))

    @gather_metrics('clear_data_caches')
    def clear(self) -> None:
        if False:
            i = 10
            return i + 15
        'Clear all in-memory and on-disk data caches.'
        self._maybe_show_deprecation_warning()
        _data_caches.clear_all()

    def _maybe_show_deprecation_warning(self):
        if False:
            return 10
        'If the API is being accessed with the deprecated `st.experimental_memo` name,\n        show a deprecation warning.\n        '
        if self._deprecation_warning is not None:
            show_deprecation_warning(self._deprecation_warning)

class DataCache(Cache):
    """Manages cached values for a single st.cache_data function."""

    def __init__(self, key: str, storage: CacheStorage, persist: CachePersistType, max_entries: int | None, ttl_seconds: float | None, display_name: str, allow_widgets: bool=False):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.key = key
        self.display_name = display_name
        self.storage = storage
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self.persist = persist
        self.allow_widgets = allow_widgets

    def get_stats(self) -> list[CacheStat]:
        if False:
            return 10
        if isinstance(self.storage, CacheStatsProvider):
            return self.storage.get_stats()
        return []

    def read_result(self, key: str) -> CachedResult:
        if False:
            print('Hello World!')
        "Read a value and messages from the cache. Raise `CacheKeyNotFoundError`\n        if the value doesn't exist, and `CacheError` if the value exists but can't\n        be unpickled.\n        "
        try:
            pickled_entry = self.storage.get(key)
        except CacheStorageKeyNotFoundError as e:
            raise CacheKeyNotFoundError(str(e)) from e
        except CacheStorageError as e:
            raise CacheError(str(e)) from e
        try:
            entry = pickle.loads(pickled_entry)
            if not isinstance(entry, MultiCacheResults):
                self.storage.delete(key)
                raise CacheKeyNotFoundError()
            ctx = get_script_run_ctx()
            if not ctx:
                raise CacheKeyNotFoundError()
            widget_key = entry.get_current_widget_key(ctx, CacheType.DATA)
            if widget_key in entry.results:
                return entry.results[widget_key]
            else:
                raise CacheKeyNotFoundError()
        except pickle.UnpicklingError as exc:
            raise CacheError(f'Failed to unpickle {key}') from exc

    @gather_metrics('_cache_data_object')
    def write_result(self, key: str, value: Any, messages: list[MsgData]) -> None:
        if False:
            while True:
                i = 10
        'Write a value and associated messages to the cache.\n        The value must be pickleable.\n        '
        ctx = get_script_run_ctx()
        if ctx is None:
            return
        main_id = st._main.id
        sidebar_id = st.sidebar.id
        if self.allow_widgets:
            widgets = {msg.widget_metadata.widget_id for msg in messages if isinstance(msg, ElementMsgData) and msg.widget_metadata is not None}
        else:
            widgets = set()
        multi_cache_results: MultiCacheResults | None = None
        try:
            multi_cache_results = self._read_multi_results_from_storage(key)
        except (CacheKeyNotFoundError, pickle.UnpicklingError):
            pass
        if multi_cache_results is None:
            multi_cache_results = MultiCacheResults(widget_ids=widgets, results={})
        multi_cache_results.widget_ids.update(widgets)
        widget_key = multi_cache_results.get_current_widget_key(ctx, CacheType.DATA)
        result = CachedResult(value, messages, main_id, sidebar_id)
        multi_cache_results.results[widget_key] = result
        try:
            pickled_entry = pickle.dumps(multi_cache_results)
        except (pickle.PicklingError, TypeError) as exc:
            raise CacheError(f'Failed to pickle {key}') from exc
        self.storage.set(key, pickled_entry)

    def _clear(self) -> None:
        if False:
            print('Hello World!')
        self.storage.clear()

    def _read_multi_results_from_storage(self, key: str) -> MultiCacheResults:
        if False:
            for i in range(10):
                print('nop')
        'Look up the results from storage and ensure it has the right type.\n\n        Raises a `CacheKeyNotFoundError` if the key has no entry, or if the\n        entry is malformed.\n        '
        try:
            pickled = self.storage.get(key)
        except CacheStorageKeyNotFoundError as e:
            raise CacheKeyNotFoundError(str(e)) from e
        maybe_results = pickle.loads(pickled)
        if isinstance(maybe_results, MultiCacheResults):
            return maybe_results
        else:
            self.storage.delete(key)
            raise CacheKeyNotFoundError()