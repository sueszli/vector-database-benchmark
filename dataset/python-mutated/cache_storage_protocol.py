""" Declares the CacheStorageContext dataclass, which contains parameter information for
each function decorated by `@st.cache_data` (for example: ttl, max_entries etc.)

Declares the CacheStorageManager protocol, which implementations are used
to create CacheStorage instances and to optionally clear all cache storages,
that were created by this manager, and to check if the context is valid for the storage.

Declares the CacheStorage protocol, which implementations are used to store cached
values for a single `@st.cache_data` decorated function serialized as bytes.

How these classes work together
-------------------------------
- CacheStorageContext : this is a dataclass that contains the parameters from
`@st.cache_data` that are passed to the CacheStorageManager.create() method.

- CacheStorageManager : each instance of this is able to create CacheStorage
instances, and optionally to clear data of all cache storages.

- CacheStorage : each instance of this is able to get, set, delete, and clear
entries for a single `@st.cache_data` decorated function.

  ┌───────────────────────────────┐
  │                               │
  │    CacheStorageManager        │
  │                               │
  │     - clear_all(optional)     │
  │     - check_context           │
  │                               │
  └──┬────────────────────────────┘
     │
     │                ┌──────────────────────┐
     │                │  CacheStorage        │
     │ create(context)│                      │
     └────────────────►    - get             │
                      │    - set             │
                      │    - delete          │
                      │    - close (optional)│
                      │    - clear           │
                      └──────────────────────┘
"""
from __future__ import annotations
from abc import abstractmethod
from dataclasses import dataclass
from typing_extensions import Literal, Protocol

class CacheStorageError(Exception):
    """Base exception raised by the cache storage"""

class CacheStorageKeyNotFoundError(CacheStorageError):
    """Raised when the key is not found in the cache storage"""

class InvalidCacheStorageContext(CacheStorageError):
    """Raised if the cache storage manager is not able to work with
    provided CacheStorageContext.
    """

@dataclass(frozen=True)
class CacheStorageContext:
    """Context passed to the cache storage during initialization
    This is the normalized parameters that are passed to CacheStorageManager.create()
    method.

    Parameters
    ----------
    function_key: str
        A hash computed based on function name and source code decorated
        by `@st.cache_data`

    function_display_name: str
        The display name of the function that is decorated by `@st.cache_data`

    ttl_seconds : float or None
        The time-to-live for the keys in storage, in seconds. If None, the entry
        will never expire.

    max_entries : int or None
        The maximum number of entries to store in the cache storage.
        If None, the cache storage will not limit the number of entries.

    persist : Literal["disk"] or None
        The persistence mode for the cache storage.
        Legacy parameter, that used in Streamlit current cache storage implementation.
        Could be ignored by cache storage implementation, if storage does not support
        persistence or it persistent by default.
    """
    function_key: str
    function_display_name: str
    ttl_seconds: float | None = None
    max_entries: int | None = None
    persist: Literal['disk'] | None = None

class CacheStorage(Protocol):
    """Cache storage protocol, that should be implemented by the concrete cache storages.
    Used to store cached values for a single `@st.cache_data` decorated function
    serialized as bytes.

    CacheStorage instances should be created by `CacheStorageManager.create()` method.

    Notes
    -----
    Threading: The methods of this protocol could be called from multiple threads.
    This is a responsibility of the concrete implementation to ensure thread safety
    guarantees.
    """

    @abstractmethod
    def get(self, key: str) -> bytes:
        if False:
            i = 10
            return i + 15
        'Returns the stored value for the key.\n\n        Raises\n        ------\n        CacheStorageKeyNotFoundError\n            Raised if the key is not in the storage.\n        '
        raise NotImplementedError

    @abstractmethod
    def set(self, key: str, value: bytes) -> None:
        if False:
            i = 10
            return i + 15
        'Sets the value for a given key'
        raise NotImplementedError

    @abstractmethod
    def delete(self, key: str) -> None:
        if False:
            print('Hello World!')
        'Delete a given key'
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        if False:
            while True:
                i = 10
        'Remove all keys for the storage'
        raise NotImplementedError

    def close(self) -> None:
        if False:
            print('Hello World!')
        'Closes the cache storage, it is optional to implement, and should be used\n        to close open resources, before we delete the storage instance.\n        e.g. close the database connection etc.\n        '
        pass

class CacheStorageManager(Protocol):
    """Cache storage manager protocol, that should be implemented by the concrete
    cache storage managers.

    It is responsible for:
        - Creating cache storage instances for the specific
        decorated functions,
        - Validating the context for the cache storages.
        - Optionally clearing all cache storages in optimal way.

    It should be created during Runtime initialization.
    """

    @abstractmethod
    def create(self, context: CacheStorageContext) -> CacheStorage:
        if False:
            return 10
        'Creates a new cache storage instance\n        Please note that the ttl, max_entries and other context fields are specific\n        for whole storage, not for individual key.\n\n        Notes\n        -----\n        Threading: Should be safe to call from any thread.\n        '
        raise NotImplementedError

    def clear_all(self) -> None:
        if False:
            while True:
                i = 10
        'Remove everything what possible from the cache storages in optimal way.\n        meaningful default behaviour is to raise NotImplementedError, so this is not\n        abstractmethod.\n\n        The method is optional to implement: cache data API will fall back to remove\n        all available storages one by one via storage.clear() method\n        if clear_all raises NotImplementedError.\n\n        Raises\n        ------\n        NotImplementedError\n            Raised if the storage manager does not provide an ability to clear\n            all storages at once in optimal way.\n\n        Notes\n        -----\n        Threading: This method could be called from multiple threads.\n        This is a responsibility of the concrete implementation to ensure\n        thread safety guarantees.\n        '
        raise NotImplementedError

    def check_context(self, context: CacheStorageContext) -> None:
        if False:
            print('Hello World!')
        'Checks if the context is valid for the storage manager.\n        This method should not return anything, but log message or raise an exception\n        if the context is invalid.\n\n        In case of raising an exception, we not handle it and let the exception to be\n        propagated.\n\n        check_context is called only once at the moment of creating `@st.cache_data`\n        decorator for specific function, so it is not called for every cache hit.\n\n        Parameters\n        ----------\n        context: CacheStorageContext\n            The context to check for the storage manager, dummy function_key in context\n            will be used, since it is not computed at the point of calling this method.\n\n        Raises\n        ------\n        InvalidCacheStorageContext\n            Raised if the cache storage manager is not able to work with provided\n            CacheStorageContext. When possible we should log message instead, since\n            this exception will be propagated to the user.\n\n        Notes\n        -----\n        Threading: Should be safe to call from any thread.\n        '
        pass