from __future__ import annotations
import unicodedata
import numpy as np
from xarray import coding
from xarray.core.variable import Variable
_specialchars = '_.@+- !"#$%&\\()*,:;<=>?[]^`{|}~'
_reserved_names = {'byte', 'char', 'short', 'ushort', 'int', 'uint', 'int64', 'uint64', 'float', 'real', 'double', 'bool', 'string'}
_nc3_dtype_coercions = {'int64': 'int32', 'uint64': 'int32', 'uint32': 'int32', 'uint16': 'int16', 'uint8': 'int8', 'bool': 'int8'}
STRING_ENCODING = 'utf-8'

def coerce_nc3_dtype(arr):
    if False:
        for i in range(10):
            print('nop')
    'Coerce an array to a data type that can be stored in a netCDF-3 file\n\n    This function performs the dtype conversions as specified by the\n    ``_nc3_dtype_coercions`` mapping:\n        int64  -> int32\n        uint64 -> int32\n        uint32 -> int32\n        uint16 -> int16\n        uint8  -> int8\n        bool   -> int8\n\n    Data is checked for equality, or equivalence (non-NaN values) using the\n    ``(cast_array == original_array).all()``.\n    '
    dtype = str(arr.dtype)
    if dtype in _nc3_dtype_coercions:
        new_dtype = _nc3_dtype_coercions[dtype]
        cast_arr = arr.astype(new_dtype)
        if not (cast_arr == arr).all():
            raise ValueError(f'could not safely cast array from dtype {dtype} to {new_dtype}')
        arr = cast_arr
    return arr

def encode_nc3_attr_value(value):
    if False:
        return 10
    if isinstance(value, bytes):
        pass
    elif isinstance(value, str):
        value = value.encode(STRING_ENCODING)
    else:
        value = coerce_nc3_dtype(np.atleast_1d(value))
        if value.ndim > 1:
            raise ValueError('netCDF attributes must be 1-dimensional')
    return value

def encode_nc3_attrs(attrs):
    if False:
        print('Hello World!')
    return {k: encode_nc3_attr_value(v) for (k, v) in attrs.items()}

def _maybe_prepare_times(var):
    if False:
        while True:
            i = 10
    data = var.data
    if data.dtype.kind in 'iu':
        units = var.attrs.get('units', None)
        if units is not None:
            if coding.variables._is_time_like(units):
                mask = data == np.iinfo(np.int64).min
                if mask.any():
                    data = np.where(mask, var.attrs.get('_FillValue', np.nan), data)
    return data

def encode_nc3_variable(var):
    if False:
        i = 10
        return i + 15
    for coder in [coding.strings.EncodedStringCoder(allows_unicode=False), coding.strings.CharacterArrayCoder()]:
        var = coder.encode(var)
    data = _maybe_prepare_times(var)
    data = coerce_nc3_dtype(data)
    attrs = encode_nc3_attrs(var.attrs)
    return Variable(var.dims, data, attrs, var.encoding)

def _isalnumMUTF8(c):
    if False:
        while True:
            i = 10
    'Return True if the given UTF-8 encoded character is alphanumeric\n    or multibyte.\n\n    Input is not checked!\n    '
    return c.isalnum() or len(c.encode('utf-8')) > 1

def is_valid_nc3_name(s):
    if False:
        while True:
            i = 10
    'Test whether an object can be validly converted to a netCDF-3\n    dimension, variable or attribute name\n\n    Earlier versions of the netCDF C-library reference implementation\n    enforced a more restricted set of characters in creating new names,\n    but permitted reading names containing arbitrary bytes. This\n    specification extends the permitted characters in names to include\n    multi-byte UTF-8 encoded Unicode and additional printing characters\n    from the US-ASCII alphabet. The first character of a name must be\n    alphanumeric, a multi-byte UTF-8 character, or \'_\' (reserved for\n    special names with meaning to implementations, such as the\n    "_FillValue" attribute). Subsequent characters may also include\n    printing special characters, except for \'/\' which is not allowed in\n    names. Names that have trailing space characters are also not\n    permitted.\n    '
    if not isinstance(s, str):
        return False
    num_bytes = len(s.encode('utf-8'))
    return unicodedata.normalize('NFC', s) == s and s not in _reserved_names and (num_bytes >= 0) and ('/' not in s) and (s[-1] != ' ') and (_isalnumMUTF8(s[0]) or s[0] == '_') and all((_isalnumMUTF8(c) or c in _specialchars for c in s))