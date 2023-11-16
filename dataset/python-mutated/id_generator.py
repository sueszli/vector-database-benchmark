import random
import time
_VERSION_BITS = 4
_TS_BITS = 32
_RANDOM_BITS = 28
_TOTAL_BITS = _VERSION_BITS + _TS_BITS + _RANDOM_BITS
assert _TOTAL_BITS == 64
_VERSION = 2
assert _VERSION % 2 == 0
_INDEXER_EPOCH_START = 1641024000

def reverse_bits(number: int, bit_size: int) -> int:
    if False:
        print('Hello World!')
    return int(bin(number)[2:].zfill(bit_size)[::-1], 2)
_VERSION_PREFIX = reverse_bits(_VERSION, _VERSION_BITS)

def get_id() -> int:
    if False:
        return 10
    '\n    Generates IDs for use by indexer storages that do not have autoincrement sequences.\n\n    This function does not provide any guarantee of uniqueness, just a low probability of collisions.\n    It relies on the database to be strongly consistent and reject writes with duplicate IDs. These should\n    be retried with a newly generated ID.\n\n    The ID generated is in roughly incrementing order.\n\n    Metric IDs are 64 bit but this function only generates IDs that fit in 63 bits. The leading bit is always zero.\n    This is because they were stored in Postgres as BigInt (signed 64 bit) and we do not want to change that now.\n    In ClickHouse it is an unsigned 64 bit integer.\n    '
    now = int(time.time())
    time_since_epoch = now - _INDEXER_EPOCH_START
    rand = random.getrandbits(_RANDOM_BITS)
    id = _VERSION_PREFIX << _TOTAL_BITS - _VERSION_BITS
    id |= time_since_epoch << _TOTAL_BITS - _VERSION_BITS - _TS_BITS
    id |= rand
    return id