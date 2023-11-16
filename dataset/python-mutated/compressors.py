"""
Patched ``BZ2File`` and ``LZMAFile`` to handle pickle protocol 5.
"""
from __future__ import annotations
from pickle import PickleBuffer
from pandas.compat._constants import PY310
try:
    import bz2
    has_bz2 = True
except ImportError:
    has_bz2 = False
try:
    import lzma
    has_lzma = True
except ImportError:
    has_lzma = False

def flatten_buffer(b: bytes | bytearray | memoryview | PickleBuffer) -> bytes | bytearray | memoryview:
    if False:
        while True:
            i = 10
    '\n    Return some 1-D `uint8` typed buffer.\n\n    Coerces anything that does not match that description to one that does\n    without copying if possible (otherwise will copy).\n    '
    if isinstance(b, (bytes, bytearray)):
        return b
    if not isinstance(b, PickleBuffer):
        b = PickleBuffer(b)
    try:
        return b.raw()
    except BufferError:
        return memoryview(b).tobytes('A')
if has_bz2:

    class BZ2File(bz2.BZ2File):
        if not PY310:

            def write(self, b) -> int:
                if False:
                    while True:
                        i = 10
                return super().write(flatten_buffer(b))
if has_lzma:

    class LZMAFile(lzma.LZMAFile):
        if not PY310:

            def write(self, b) -> int:
                if False:
                    i = 10
                    return i + 15
                return super().write(flatten_buffer(b))