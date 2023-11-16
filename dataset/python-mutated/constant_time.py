from __future__ import annotations
import hmac

def bytes_eq(a: bytes, b: bytes) -> bool:
    if False:
        return 10
    if not isinstance(a, bytes) or not isinstance(b, bytes):
        raise TypeError('a and b must be bytes.')
    return hmac.compare_digest(a, b)