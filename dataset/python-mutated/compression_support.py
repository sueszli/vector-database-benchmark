from __future__ import annotations
import warnings
from typing import Any, Iterable, Optional, Union
try:
    import snappy
    _HAVE_SNAPPY = True
except ImportError:
    _HAVE_SNAPPY = False
try:
    import zlib
    _HAVE_ZLIB = True
except ImportError:
    _HAVE_ZLIB = False
try:
    from zstandard import ZstdCompressor, ZstdDecompressor
    _HAVE_ZSTD = True
except ImportError:
    _HAVE_ZSTD = False
from pymongo.hello import HelloCompat
from pymongo.monitoring import _SENSITIVE_COMMANDS
_SUPPORTED_COMPRESSORS = {'snappy', 'zlib', 'zstd'}
_NO_COMPRESSION = {HelloCompat.CMD, HelloCompat.LEGACY_CMD}
_NO_COMPRESSION.update(_SENSITIVE_COMMANDS)

def validate_compressors(dummy: Any, value: Union[str, Iterable[str]]) -> list[str]:
    if False:
        for i in range(10):
            print('nop')
    try:
        compressors = value.split(',')
    except AttributeError:
        compressors = list(value)
    for compressor in compressors[:]:
        if compressor not in _SUPPORTED_COMPRESSORS:
            compressors.remove(compressor)
            warnings.warn(f'Unsupported compressor: {compressor}', stacklevel=2)
        elif compressor == 'snappy' and (not _HAVE_SNAPPY):
            compressors.remove(compressor)
            warnings.warn('Wire protocol compression with snappy is not available. You must install the python-snappy module for snappy support.', stacklevel=2)
        elif compressor == 'zlib' and (not _HAVE_ZLIB):
            compressors.remove(compressor)
            warnings.warn('Wire protocol compression with zlib is not available. The zlib module is not available.', stacklevel=2)
        elif compressor == 'zstd' and (not _HAVE_ZSTD):
            compressors.remove(compressor)
            warnings.warn('Wire protocol compression with zstandard is not available. You must install the zstandard module for zstandard support.', stacklevel=2)
    return compressors

def validate_zlib_compression_level(option: str, value: Any) -> int:
    if False:
        return 10
    try:
        level = int(value)
    except Exception:
        raise TypeError(f'{option} must be an integer, not {value!r}.') from None
    if level < -1 or level > 9:
        raise ValueError('%s must be between -1 and 9, not %d.' % (option, level))
    return level

class CompressionSettings:

    def __init__(self, compressors: list[str], zlib_compression_level: int):
        if False:
            return 10
        self.compressors = compressors
        self.zlib_compression_level = zlib_compression_level

    def get_compression_context(self, compressors: Optional[list[str]]) -> Union[SnappyContext, ZlibContext, ZstdContext, None]:
        if False:
            while True:
                i = 10
        if compressors:
            chosen = compressors[0]
            if chosen == 'snappy':
                return SnappyContext()
            elif chosen == 'zlib':
                return ZlibContext(self.zlib_compression_level)
            elif chosen == 'zstd':
                return ZstdContext()
            return None
        return None

class SnappyContext:
    compressor_id = 1

    @staticmethod
    def compress(data: bytes) -> bytes:
        if False:
            print('Hello World!')
        return snappy.compress(data)

class ZlibContext:
    compressor_id = 2

    def __init__(self, level: int):
        if False:
            while True:
                i = 10
        self.level = level

    def compress(self, data: bytes) -> bytes:
        if False:
            return 10
        return zlib.compress(data, self.level)

class ZstdContext:
    compressor_id = 3

    @staticmethod
    def compress(data: bytes) -> bytes:
        if False:
            for i in range(10):
                print('nop')
        return ZstdCompressor().compress(data)

def decompress(data: bytes, compressor_id: int) -> bytes:
    if False:
        i = 10
        return i + 15
    if compressor_id == SnappyContext.compressor_id:
        return snappy.uncompress(bytes(data))
    elif compressor_id == ZlibContext.compressor_id:
        return zlib.decompress(data)
    elif compressor_id == ZstdContext.compressor_id:
        return ZstdDecompressor().decompress(data)
    else:
        raise ValueError('Unknown compressorId %d' % (compressor_id,))