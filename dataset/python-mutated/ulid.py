"""Helpers to generate ulids."""
from __future__ import annotations
import time
from ulid_transform import bytes_to_ulid, ulid_at_time, ulid_hex, ulid_to_bytes
__all__ = ['ulid', 'ulid_hex', 'ulid_at_time', 'ulid_to_bytes', 'bytes_to_ulid']

def ulid(timestamp: float | None=None) -> str:
    if False:
        for i in range(10):
            print('nop')
    'Generate a ULID.\n\n    This ulid should not be used for cryptographically secure\n    operations.\n\n     01AN4Z07BY      79KA1307SR9X4MV3\n    |----------|    |----------------|\n     Timestamp          Randomness\n       48bits             80bits\n\n    This string can be loaded directly with https://github.com/ahawker/ulid\n\n    import homeassistant.util.ulid as ulid_util\n    import ulid\n    ulid.parse(ulid_util.ulid())\n    '
    return ulid_at_time(timestamp or time.time())