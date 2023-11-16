"""Declares the LocalDiskCacheStorageManager class, which is used
to create LocalDiskCacheStorage instances wrapped by InMemoryCacheStorageWrapper,
InMemoryCacheStorageWrapper wrapper allows to have first layer of in-memory cache,
before accessing to LocalDiskCacheStorage itself.

Declares the LocalDiskCacheStorage class, which is used to store cached
values on disk.

How these classes work together
-------------------------------

- LocalDiskCacheStorageManager : each instance of this is able
to create LocalDiskCacheStorage instances wrapped by InMemoryCacheStorageWrapper,
and to clear data from cache storage folder. It is also LocalDiskCacheStorageManager
responsibility to check if the context is valid for the storage, and to log warning
if the context is not valid.

- LocalDiskCacheStorage : each instance of this is able to get, set, delete, and clear
entries from disk for a single `@st.cache_data` decorated function if `persist="disk"`
is used in CacheStorageContext.


    ┌───────────────────────────────┐
    │  LocalDiskCacheStorageManager │
    │                               │
    │     - clear_all               │
    │     - check_context           │
    │                               │
    └──┬────────────────────────────┘
       │
       │                ┌──────────────────────────────┐
       │                │                              │
       │ create(context)│  InMemoryCacheStorageWrapper │
       └────────────────►                              │
                        │    ┌─────────────────────┐   │
                        │    │                     │   │
                        │    │   LocalDiskStorage  │   │
                        │    │                     │   │
                        │    └─────────────────────┘   │
                        │                              │
                        └──────────────────────────────┘

"""
from __future__ import annotations
import math
import os
import shutil
from streamlit import util
from streamlit.file_util import get_streamlit_file_path, streamlit_read, streamlit_write
from streamlit.logger import get_logger
from streamlit.runtime.caching.storage.cache_storage_protocol import CacheStorage, CacheStorageContext, CacheStorageError, CacheStorageKeyNotFoundError, CacheStorageManager
from streamlit.runtime.caching.storage.in_memory_cache_storage_wrapper import InMemoryCacheStorageWrapper
_CACHE_DIR_NAME = 'cache'
_CACHED_FILE_EXTENSION = 'memo'
_LOGGER = get_logger(__name__)

class LocalDiskCacheStorageManager(CacheStorageManager):

    def create(self, context: CacheStorageContext) -> CacheStorage:
        if False:
            for i in range(10):
                print('nop')
        'Creates a new cache storage instance wrapped with in-memory cache layer'
        persist_storage = LocalDiskCacheStorage(context)
        return InMemoryCacheStorageWrapper(persist_storage=persist_storage, context=context)

    def clear_all(self) -> None:
        if False:
            while True:
                i = 10
        cache_path = get_cache_folder_path()
        if os.path.isdir(cache_path):
            shutil.rmtree(cache_path)

    def check_context(self, context: CacheStorageContext) -> None:
        if False:
            for i in range(10):
                print('nop')
        if context.persist == 'disk' and context.ttl_seconds is not None and (not math.isinf(context.ttl_seconds)):
            _LOGGER.warning(f"The cached function '{context.function_display_name}' has a TTL that will be ignored. Persistent cached functions currently don't support TTL.")

class LocalDiskCacheStorage(CacheStorage):
    """Cache storage that persists data to disk
    This is the default cache persistence layer for `@st.cache_data`
    """

    def __init__(self, context: CacheStorageContext):
        if False:
            for i in range(10):
                print('nop')
        self.function_key = context.function_key
        self.persist = context.persist
        self._ttl_seconds = context.ttl_seconds
        self._max_entries = context.max_entries

    @property
    def ttl_seconds(self) -> float:
        if False:
            return 10
        return self._ttl_seconds if self._ttl_seconds is not None else math.inf

    @property
    def max_entries(self) -> float:
        if False:
            return 10
        return float(self._max_entries) if self._max_entries is not None else math.inf

    def get(self, key: str) -> bytes:
        if False:
            i = 10
            return i + 15
        '\n        Returns the stored value for the key if persisted,\n        raise CacheStorageKeyNotFoundError if not found, or not configured\n        with persist="disk"\n        '
        if self.persist == 'disk':
            path = self._get_cache_file_path(key)
            try:
                with streamlit_read(path, binary=True) as input:
                    value = input.read()
                    _LOGGER.debug('Disk cache HIT: %s', key)
                    return bytes(value)
            except FileNotFoundError:
                raise CacheStorageKeyNotFoundError('Key not found in disk cache')
            except Exception as ex:
                _LOGGER.error(ex)
                raise CacheStorageError('Unable to read from cache') from ex
        else:
            raise CacheStorageKeyNotFoundError(f'Local disk cache storage is disabled (persist={self.persist})')

    def set(self, key: str, value: bytes) -> None:
        if False:
            print('Hello World!')
        'Sets the value for a given key'
        if self.persist == 'disk':
            path = self._get_cache_file_path(key)
            try:
                with streamlit_write(path, binary=True) as output:
                    output.write(value)
            except util.Error as e:
                _LOGGER.debug(e)
                try:
                    os.remove(path)
                except (FileNotFoundError, IOError, OSError):
                    pass
                raise CacheStorageError('Unable to write to cache') from e

    def delete(self, key: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Delete a cache file from disk. If the file does not exist on disk,\n        return silently. If another exception occurs, log it. Does not throw.\n        '
        if self.persist == 'disk':
            path = self._get_cache_file_path(key)
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
            except Exception as ex:
                _LOGGER.exception('Unable to remove a file from the disk cache', exc_info=ex)

    def clear(self) -> None:
        if False:
            return 10
        'Delete all keys for the current storage'
        cache_dir = get_cache_folder_path()
        if os.path.isdir(cache_dir):
            for file_name in os.listdir(cache_dir):
                if self._is_cache_file(file_name):
                    os.remove(os.path.join(cache_dir, file_name))

    def close(self) -> None:
        if False:
            while True:
                i = 10
        'Dummy implementation of close, we don\'t need to actually "close" anything'

    def _get_cache_file_path(self, value_key: str) -> str:
        if False:
            i = 10
            return i + 15
        'Return the path of the disk cache file for the given value.'
        cache_dir = get_cache_folder_path()
        return os.path.join(cache_dir, f'{self.function_key}-{value_key}.{_CACHED_FILE_EXTENSION}')

    def _is_cache_file(self, fname: str) -> bool:
        if False:
            i = 10
            return i + 15
        'Return true if the given file name is a cache file for this storage.'
        return fname.startswith(f'{self.function_key}-') and fname.endswith(f'.{_CACHED_FILE_EXTENSION}')

def get_cache_folder_path() -> str:
    if False:
        return 10
    return get_streamlit_file_path(_CACHE_DIR_NAME)