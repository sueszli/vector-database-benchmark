from __future__ import annotations
import sys
from contextlib import suppress
if sys.version_info < (3, 11):
    import tomli as tomllib
else:
    import tomllib
if sys.version_info < (3, 10):
    import importlib_metadata as metadata
else:
    from importlib import metadata
WINDOWS = sys.platform == 'win32'

def decode(string: bytes | str, encodings: list[str] | None=None) -> str:
    if False:
        return 10
    if not isinstance(string, bytes):
        return string
    encodings = encodings or ['utf-8', 'latin1', 'ascii']
    for encoding in encodings:
        with suppress(UnicodeEncodeError, UnicodeDecodeError):
            return string.decode(encoding)
    return string.decode(encodings[0], errors='ignore')

def encode(string: str, encodings: list[str] | None=None) -> bytes:
    if False:
        return 10
    if isinstance(string, bytes):
        return string
    encodings = encodings or ['utf-8', 'latin1', 'ascii']
    for encoding in encodings:
        with suppress(UnicodeEncodeError, UnicodeDecodeError):
            return string.encode(encoding)
    return string.encode(encodings[0], errors='ignore')
__all__ = ['WINDOWS', 'decode', 'encode', 'metadata', 'tomllib']