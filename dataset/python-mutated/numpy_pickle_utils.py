"""Utilities for fast persistence of big data, with optional compression."""
import pickle
import io
import sys
import warnings
import contextlib
from .compressor import _ZFILE_PREFIX
from .compressor import _COMPRESSORS
try:
    import numpy as np
except ImportError:
    np = None
Unpickler = pickle._Unpickler
Pickler = pickle._Pickler
xrange = range
try:
    import bz2
except ImportError:
    bz2 = None
_IO_BUFFER_SIZE = 1024 ** 2

def _is_raw_file(fileobj):
    if False:
        print('Hello World!')
    'Check if fileobj is a raw file object, e.g created with open.'
    fileobj = getattr(fileobj, 'raw', fileobj)
    return isinstance(fileobj, io.FileIO)

def _get_prefixes_max_len():
    if False:
        while True:
            i = 10
    prefixes = [len(compressor.prefix) for compressor in _COMPRESSORS.values()]
    prefixes += [len(_ZFILE_PREFIX)]
    return max(prefixes)

def _is_numpy_array_byte_order_mismatch(array):
    if False:
        i = 10
        return i + 15
    'Check if numpy array is having byte order mismatch'
    return sys.byteorder == 'big' and (array.dtype.byteorder == '<' or (array.dtype.byteorder == '|' and array.dtype.fields and all((e[0].byteorder == '<' for e in array.dtype.fields.values())))) or (sys.byteorder == 'little' and (array.dtype.byteorder == '>' or (array.dtype.byteorder == '|' and array.dtype.fields and all((e[0].byteorder == '>' for e in array.dtype.fields.values())))))

def _ensure_native_byte_order(array):
    if False:
        print('Hello World!')
    'Use the byte order of the host while preserving values\n\n    Does nothing if array already uses the system byte order.\n    '
    if _is_numpy_array_byte_order_mismatch(array):
        array = array.byteswap().view(array.dtype.newbyteorder('='))
    return array

def _detect_compressor(fileobj):
    if False:
        i = 10
        return i + 15
    "Return the compressor matching fileobj.\n\n    Parameters\n    ----------\n    fileobj: file object\n\n    Returns\n    -------\n    str in {'zlib', 'gzip', 'bz2', 'lzma', 'xz', 'compat', 'not-compressed'}\n    "
    max_prefix_len = _get_prefixes_max_len()
    if hasattr(fileobj, 'peek'):
        first_bytes = fileobj.peek(max_prefix_len)
    else:
        first_bytes = fileobj.read(max_prefix_len)
        fileobj.seek(0)
    if first_bytes.startswith(_ZFILE_PREFIX):
        return 'compat'
    else:
        for (name, compressor) in _COMPRESSORS.items():
            if first_bytes.startswith(compressor.prefix):
                return name
    return 'not-compressed'

def _buffered_read_file(fobj):
    if False:
        for i in range(10):
            print('nop')
    'Return a buffered version of a read file object.'
    return io.BufferedReader(fobj, buffer_size=_IO_BUFFER_SIZE)

def _buffered_write_file(fobj):
    if False:
        for i in range(10):
            print('nop')
    'Return a buffered version of a write file object.'
    return io.BufferedWriter(fobj, buffer_size=_IO_BUFFER_SIZE)

@contextlib.contextmanager
def _read_fileobject(fileobj, filename, mmap_mode=None):
    if False:
        while True:
            i = 10
    "Utility function opening the right fileobject from a filename.\n\n    The magic number is used to choose between the type of file object to open:\n    * regular file object (default)\n    * zlib file object\n    * gzip file object\n    * bz2 file object\n    * lzma file object (for xz and lzma compressor)\n\n    Parameters\n    ----------\n    fileobj: file object\n    compressor: str in {'zlib', 'gzip', 'bz2', 'lzma', 'xz', 'compat',\n                        'not-compressed'}\n    filename: str\n        filename path corresponding to the fileobj parameter.\n    mmap_mode: str\n        memory map mode that should be used to open the pickle file. This\n        parameter is useful to verify that the user is not trying to one with\n        compression. Default: None.\n\n    Returns\n    -------\n        a file like object\n\n    "
    compressor = _detect_compressor(fileobj)
    if compressor == 'compat':
        warnings.warn("The file '%s' has been generated with a joblib version less than 0.10. Please regenerate this pickle file." % filename, DeprecationWarning, stacklevel=2)
        yield filename
    else:
        if compressor in _COMPRESSORS:
            compressor_wrapper = _COMPRESSORS[compressor]
            inst = compressor_wrapper.decompressor_file(fileobj)
            fileobj = _buffered_read_file(inst)
        if mmap_mode is not None:
            if isinstance(fileobj, io.BytesIO):
                warnings.warn('In memory persistence is not compatible with mmap_mode "%(mmap_mode)s" flag passed. mmap_mode option will be ignored.' % locals(), stacklevel=2)
            elif compressor != 'not-compressed':
                warnings.warn('mmap_mode "%(mmap_mode)s" is not compatible with compressed file %(filename)s. "%(mmap_mode)s" flag will be ignored.' % locals(), stacklevel=2)
            elif not _is_raw_file(fileobj):
                warnings.warn('"%(fileobj)r" is not a raw file, mmap_mode "%(mmap_mode)s" flag will be ignored.' % locals(), stacklevel=2)
        yield fileobj

def _write_fileobject(filename, compress=('zlib', 3)):
    if False:
        print('Hello World!')
    'Return the right compressor file object in write mode.'
    compressmethod = compress[0]
    compresslevel = compress[1]
    if compressmethod in _COMPRESSORS.keys():
        file_instance = _COMPRESSORS[compressmethod].compressor_file(filename, compresslevel=compresslevel)
        return _buffered_write_file(file_instance)
    else:
        file_instance = _COMPRESSORS['zlib'].compressor_file(filename, compresslevel=compresslevel)
        return _buffered_write_file(file_instance)
BUFFER_SIZE = 2 ** 18

def _read_bytes(fp, size, error_template='ran out of data'):
    if False:
        print('Hello World!')
    'Read from file-like object until size bytes are read.\n\n    TODO python2_drop: is it still needed? The docstring mentions python 2.6\n    and it looks like this can be at least simplified ...\n\n    Raises ValueError if not EOF is encountered before size bytes are read.\n    Non-blocking objects only supported if they derive from io objects.\n\n    Required as e.g. ZipExtFile in python 2.6 can return less data than\n    requested.\n\n    This function was taken from numpy/lib/format.py in version 1.10.2.\n\n    Parameters\n    ----------\n    fp: file-like object\n    size: int\n    error_template: str\n\n    Returns\n    -------\n    a bytes object\n        The data read in bytes.\n\n    '
    data = bytes()
    while True:
        try:
            r = fp.read(size - len(data))
            data += r
            if len(r) == 0 or len(data) == size:
                break
        except io.BlockingIOError:
            pass
    if len(data) != size:
        msg = 'EOF: reading %s, expected %d bytes got %d'
        raise ValueError(msg % (error_template, size, len(data)))
    else:
        return data