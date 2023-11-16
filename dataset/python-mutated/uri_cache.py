import logging
from typing import Set, Callable
default_logger = logging.getLogger(__name__)
DEFAULT_MAX_URI_CACHE_SIZE_BYTES = 1024 ** 3 * 10

class URICache:
    """
    Caches URIs up to a specified total size limit.

    URIs are represented by strings.  Each URI has an associated size on disk.

    When a URI is added to the URICache, it is marked as "in use".
    When a URI is no longer in use, the user of this class should call
    `mark_unused` to signal that the URI is safe for deletion.

    URIs in the cache can be marked as "in use" by calling `mark_used`.

    Deletion of URIs on disk does not occur until the size limit is exceeded.
    When this happens, URIs that are not in use are deleted randomly until the
    size limit is satisfied, or there are no more URIs that are not in use.

    It is possible for the total size on disk to exceed the size limit if all
    the URIs are in use.

    """

    def __init__(self, delete_fn: Callable[[str, logging.Logger], int]=lambda uri, logger: 0, max_total_size_bytes: int=DEFAULT_MAX_URI_CACHE_SIZE_BYTES, debug_mode: bool=False):
        if False:
            print('Hello World!')
        self._used_uris: Set[str] = set()
        self._unused_uris: Set[str] = set()
        self._delete_fn = delete_fn
        self._total_size_bytes = 0
        self.max_total_size_bytes = max_total_size_bytes
        self._debug_mode = debug_mode

    def mark_unused(self, uri: str, logger: logging.Logger=default_logger):
        if False:
            i = 10
            return i + 15
        'Mark a URI as unused and okay to be deleted.'
        if uri not in self._used_uris:
            logger.info(f'URI {uri} is already unused.')
        else:
            self._unused_uris.add(uri)
            self._used_uris.remove(uri)
        logger.info(f'Marked URI {uri} unused.')
        self._evict_if_needed(logger)
        self._check_valid()

    def mark_used(self, uri: str, logger: logging.Logger=default_logger):
        if False:
            return 10
        'Mark a URI as in use.  URIs in use will not be deleted.'
        if uri in self._used_uris:
            return
        elif uri in self._unused_uris:
            self._used_uris.add(uri)
            self._unused_uris.remove(uri)
        else:
            raise ValueError(f'Got request to mark URI {uri} used, but this URI is not present in the cache.')
        logger.info(f'Marked URI {uri} used.')
        self._check_valid()

    def add(self, uri: str, size_bytes: int, logger: logging.Logger=default_logger):
        if False:
            return 10
        'Add a URI to the cache and mark it as in use.'
        if uri in self._unused_uris:
            self._unused_uris.remove(uri)
        self._used_uris.add(uri)
        self._total_size_bytes += size_bytes
        self._evict_if_needed(logger)
        self._check_valid()
        logger.info(f'Added URI {uri} with size {size_bytes}')

    def get_total_size_bytes(self) -> int:
        if False:
            while True:
                i = 10
        return self._total_size_bytes

    def _evict_if_needed(self, logger: logging.Logger=default_logger):
        if False:
            while True:
                i = 10
        'Evict unused URIs (if they exist) until total size <= max size.'
        while self._unused_uris and self.get_total_size_bytes() > self.max_total_size_bytes:
            arbitrary_unused_uri = next(iter(self._unused_uris))
            self._unused_uris.remove(arbitrary_unused_uri)
            num_bytes_deleted = self._delete_fn(arbitrary_unused_uri, logger)
            self._total_size_bytes -= num_bytes_deleted
            logger.info(f'Deleted URI {arbitrary_unused_uri} with size {num_bytes_deleted}.')

    def _check_valid(self):
        if False:
            return 10
        '(Debug mode only) Check "used" and "unused" sets are disjoint.'
        if self._debug_mode:
            assert self._used_uris & self._unused_uris == set()

    def __contains__(self, uri):
        if False:
            print('Hello World!')
        return uri in self._used_uris or uri in self._unused_uris

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return str(self.__dict__)