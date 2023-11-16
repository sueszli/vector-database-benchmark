"""Utils for interoperability with other libraries.

Just CFFI pointer casting for now.
"""
from typing import Any

def cast_int_addr(n: Any) -> int:
    if False:
        i = 10
        return i + 15
    'Cast an address to a Python int\n\n    This could be a Python integer or a CFFI pointer\n    '
    if isinstance(n, int):
        return n
    try:
        import cffi
    except ImportError:
        pass
    else:
        ffi = cffi.FFI()
        if isinstance(n, ffi.CData):
            return int(ffi.cast('size_t', n))
    raise ValueError('Cannot cast %r to int' % n)