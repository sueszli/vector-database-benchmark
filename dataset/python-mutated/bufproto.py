"""
Typing support for the buffer protocol (PEP 3118).
"""
import array
from numba.core import types
_pep3118_int_types = set('bBhHiIlLqQnN')
_pep3118_scalar_map = {'f': types.float32, 'd': types.float64, 'Zf': types.complex64, 'Zd': types.complex128}
_type_map = {bytearray: types.ByteArray, array.array: types.PyArray}
_type_map[memoryview] = types.MemoryView
_type_map[bytes] = types.Bytes

def decode_pep3118_format(fmt, itemsize):
    if False:
        return 10
    '\n    Return the Numba type for an item with format string *fmt* and size\n    *itemsize* (in bytes).\n    '
    if fmt in _pep3118_int_types:
        name = 'int%d' % (itemsize * 8,)
        if fmt.isupper():
            name = 'u' + name
        return types.Integer(name)
    try:
        return _pep3118_scalar_map[fmt.lstrip('=')]
    except KeyError:
        raise ValueError('unsupported PEP 3118 format %r' % (fmt,))

def get_type_class(typ):
    if False:
        for i in range(10):
            print('nop')
    '\n    Get the Numba type class for buffer-compatible Python *typ*.\n    '
    try:
        return _type_map[typ]
    except KeyError:
        return types.Buffer

def infer_layout(val):
    if False:
        for i in range(10):
            print('nop')
    '\n    Infer layout of the given memoryview *val*.\n    '
    return 'C' if val.c_contiguous else 'F' if val.f_contiguous else 'A'