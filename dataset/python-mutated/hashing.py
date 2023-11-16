from __future__ import annotations
import binascii
import hashlib
hashers = []
try:
    import cityhash
except ImportError:
    pass
else:
    if cityhash.__version__ >= '0.2.2':

        def _hash_cityhash(buf):
            if False:
                while True:
                    i = 10
            '\n            Produce a 16-bytes hash of *buf* using CityHash.\n            '
            h = cityhash.CityHash128(buf)
            return h.to_bytes(16, 'little')
        hashers.append(_hash_cityhash)
try:
    import xxhash
except ImportError:
    pass
else:

    def _hash_xxhash(buf):
        if False:
            i = 10
            return i + 15
        '\n        Produce a 8-bytes hash of *buf* using xxHash.\n        '
        return xxhash.xxh64(buf).digest()
    hashers.append(_hash_xxhash)
try:
    import mmh3
except ImportError:
    pass
else:

    def _hash_murmurhash(buf):
        if False:
            while True:
                i = 10
        '\n        Produce a 16-bytes hash of *buf* using MurmurHash.\n        '
        return mmh3.hash_bytes(buf)
    hashers.append(_hash_murmurhash)

def _hash_sha1(buf):
    if False:
        while True:
            i = 10
    '\n    Produce a 20-bytes hash of *buf* using SHA1.\n    '
    return hashlib.sha1(buf).digest()
hashers.append(_hash_sha1)

def hash_buffer(buf, hasher=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Hash a bytes-like (buffer-compatible) object.  This function returns\n    a good quality hash but is not cryptographically secure.  The fastest\n    available algorithm is selected.  A fixed-length bytes object is returned.\n    '
    if hasher is not None:
        try:
            return hasher(buf)
        except (TypeError, OverflowError):
            pass
    for hasher in hashers:
        try:
            return hasher(buf)
        except (TypeError, OverflowError):
            pass
    raise TypeError(f'unsupported type for hashing: {type(buf)}')

def hash_buffer_hex(buf, hasher=None):
    if False:
        while True:
            i = 10
    '\n    Same as hash_buffer, but returns its result in hex-encoded form.\n    '
    h = hash_buffer(buf, hasher)
    s = binascii.b2a_hex(h)
    return s.decode()