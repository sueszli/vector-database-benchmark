from __future__ import annotations
from streamlit.runtime.caching.storage.cache_storage_protocol import CacheStorage, CacheStorageContext, CacheStorageKeyNotFoundError, CacheStorageManager
from streamlit.runtime.caching.storage.in_memory_cache_storage_wrapper import InMemoryCacheStorageWrapper

class MemoryCacheStorageManager(CacheStorageManager):

    def create(self, context: CacheStorageContext) -> CacheStorage:
        if False:
            while True:
                i = 10
        'Creates a new cache storage instance wrapped with in-memory cache layer'
        persist_storage = DummyCacheStorage()
        return InMemoryCacheStorageWrapper(persist_storage=persist_storage, context=context)

    def clear_all(self) -> None:
        if False:
            return 10
        raise NotImplementedError

    def check_context(self, context: CacheStorageContext) -> None:
        if False:
            i = 10
            return i + 15
        pass

class DummyCacheStorage(CacheStorage):

    def get(self, key: str) -> bytes:
        if False:
            i = 10
            return i + 15
        '\n        Dummy gets the value for a given key,\n        always raises an CacheStorageKeyNotFoundError\n        '
        raise CacheStorageKeyNotFoundError('Key not found in dummy cache')

    def set(self, key: str, value: bytes) -> None:
        if False:
            for i in range(10):
                print('nop')
        pass

    def delete(self, key: str) -> None:
        if False:
            i = 10
            return i + 15
        pass

    def clear(self) -> None:
        if False:
            while True:
                i = 10
        pass

    def close(self) -> None:
        if False:
            print('Hello World!')
        pass