"""Utilities for caching OCSP responses."""
from __future__ import annotations
from collections import namedtuple
from datetime import datetime as _datetime
from datetime import timezone
from typing import TYPE_CHECKING, Any
from pymongo.lock import _create_lock
if TYPE_CHECKING:
    from cryptography.x509.ocsp import OCSPRequest, OCSPResponse

class _OCSPCache:
    """A cache for OCSP responses."""
    CACHE_KEY_TYPE = namedtuple('OcspResponseCacheKey', ['hash_algorithm', 'issuer_name_hash', 'issuer_key_hash', 'serial_number'])

    def __init__(self) -> None:
        if False:
            print('Hello World!')
        self._data: dict[Any, OCSPResponse] = {}
        self._lock = _create_lock()

    def _get_cache_key(self, ocsp_request: OCSPRequest) -> CACHE_KEY_TYPE:
        if False:
            for i in range(10):
                print('nop')
        return self.CACHE_KEY_TYPE(hash_algorithm=ocsp_request.hash_algorithm.name.lower(), issuer_name_hash=ocsp_request.issuer_name_hash, issuer_key_hash=ocsp_request.issuer_key_hash, serial_number=ocsp_request.serial_number)

    def __setitem__(self, key: OCSPRequest, value: OCSPResponse) -> None:
        if False:
            for i in range(10):
                print('nop')
        "Add/update a cache entry.\n\n        'key' is of type cryptography.x509.ocsp.OCSPRequest\n        'value' is of type cryptography.x509.ocsp.OCSPResponse\n\n        Validity of the OCSP response must be checked by caller.\n        "
        with self._lock:
            cache_key = self._get_cache_key(key)
            if value.next_update is None:
                self._data.pop(cache_key, None)
                return
            if not value.this_update <= _datetime.now(tz=timezone.utc).replace(tzinfo=None) < value.next_update:
                return
            cached_value = self._data.get(cache_key, None)
            if cached_value is None or (cached_value.next_update is not None and cached_value.next_update < value.next_update):
                self._data[cache_key] = value

    def __getitem__(self, item: OCSPRequest) -> OCSPResponse:
        if False:
            print('Hello World!')
        "Get a cache entry if it exists.\n\n        'item' is of type cryptography.x509.ocsp.OCSPRequest\n\n        Raises KeyError if the item is not in the cache.\n        "
        with self._lock:
            cache_key = self._get_cache_key(item)
            value = self._data[cache_key]
            assert value.this_update is not None
            assert value.next_update is not None
            if value.this_update <= _datetime.now(tz=timezone.utc).replace(tzinfo=None) < value.next_update:
                return value
            self._data.pop(cache_key, None)
            raise KeyError(cache_key)