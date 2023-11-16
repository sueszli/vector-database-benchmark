from .type_bool import bool as types_bool
from .type_double import is_float, fp16 as types_fp16, fp32 as types_fp32, fp64 as types_fp64
from .type_list import is_list
from .type_int import is_int, int8 as types_int8, int16 as types_int16, int32 as types_int32, int64 as types_int64, uint8 as types_uint8, uint16 as types_uint16, uint32 as types_uint32, uint64 as types_uint64
from .type_str import str as types_str
from .type_unknown import unknown
import numpy as np
import sympy as sm
import six
from .get_type_info import get_type_info
_types_TO_NPTYPES = {types_bool: np.bool_, types_int8: np.int8, types_int16: np.int16, types_int32: np.int32, types_int64: np.int64, types_uint8: np.uint8, types_uint16: np.uint16, types_uint32: np.uint32, types_uint64: np.uint64, types_fp16: np.float16, types_fp32: np.float32, types_fp64: np.float64, types_str: np.str_}
_types_TO_STRINGS = {types_bool: 'bool', types_int8: 'i8', types_int16: 'i16', types_int32: 'i32', types_int64: 'i64', types_uint8: 'u8', types_uint16: 'u16', types_uint32: 'u32', types_uint64: 'u64', types_fp16: 'fp16', types_fp32: 'fp32', types_fp64: 'fp64', types_str: 'str'}

def np_dtype_to_py_type(np_dtype):
    if False:
        for i in range(10):
            print('nop')
    if np_dtype in [np.int32, np.int64]:
        return int
    if np_dtype == np.bool:
        return bool
    if np_dtype in [np.float32, np.float64]:
        return float
    raise NotImplementedError('{} is not supported'.format(np_dtype))
_STRINGS_TO_types = {v: k for (k, v) in _types_TO_STRINGS.items()}

def string_to_builtin(s):
    if False:
        while True:
            i = 10
    '\n    Given a str, return its corresponding builtin type.\n    '
    return _STRINGS_TO_types.get(s, None)

def builtin_to_string(builtin_type):
    if False:
        print('Hello World!')
    '\n    Given a builtin type, return its corresponding string representation.\n    '
    return _types_TO_STRINGS.get(builtin_type, None)

def nptype_from_builtin(btype):
    if False:
        print('Hello World!')
    '\n    Given a builtin type, return its corresponding Numpy dtype.\n    '
    return _types_TO_NPTYPES.get(btype, None)

def promote_types(dtype1, dtype2):
    if False:
        i = 10
        return i + 15
    '\n    Get the smallest type to which the given scalar types can be cast.\n\n    Args:\n        dtype1 (builtin):\n        dtype2 (builtin):\n\n    Returns:\n        A builtin datatype or None.\n    '
    nptype1 = nptype_from_builtin(dtype1)
    nptype2 = nptype_from_builtin(dtype2)
    if np.issubdtype(nptype1, np.floating) and np.issubdtype(nptype2, np.signedinteger):
        nppromoted = nptype1
    elif np.issubdtype(nptype2, np.floating) and np.issubdtype(nptype1, np.signedinteger):
        nppromoted = nptype2
    else:
        nppromoted = np.promote_types(nptype1, nptype2)
    return numpy_type_to_builtin_type(nppromoted)

def is_primitive(btype):
    if False:
        i = 10
        return i + 15
    '\n    Is the indicated builtin type a primitive?\n    '
    return btype is types_bool or btype is types_str or is_float(btype) or is_int(btype)

def is_scalar(btype):
    if False:
        while True:
            i = 10
    '\n    Is the given builtin type a scalar integer, float, or boolean?\n    '
    return btype is types_bool or is_int(btype) or is_float(btype)

def is_tensor(tensor_type):
    if False:
        return 10
    if tensor_type is None:
        return False
    try:
        type_info = get_type_info(tensor_type).name
    except TypeError:
        return False
    return type_info == 'tensor'

def is_str(t):
    if False:
        print('Hello World!')
    if t is None:
        return False
    try:
        type_info = get_type_info(t).name
    except TypeError:
        return False
    return type_info == 'str'

def is_tuple(t):
    if False:
        while True:
            i = 10
    if t is None:
        return False
    try:
        type_info = get_type_info(t).name
    except TypeError:
        return False
    return type_info == 'tuple'

def is_builtin(t):
    if False:
        i = 10
        return i + 15
    return is_scalar(t) or is_tensor(t) or is_str(t) or is_tuple(t)

def numpy_type_to_builtin_type(nptype):
    if False:
        print('Hello World!')
    if type(nptype) == np.dtype:
        nptype = nptype.type
    if np.issubclass_(nptype, np.bool) or np.issubclass_(nptype, np.bool_):
        return types_bool
    elif np.issubclass_(nptype, np.int8):
        return types_int8
    elif np.issubclass_(nptype, np.int16):
        return types_int16
    elif np.issubclass_(nptype, np.int32):
        return types_int32
    elif np.issubclass_(nptype, np.int64):
        return types_int64
    elif np.issubclass_(nptype, np.uint8):
        return types_int8
    elif np.issubclass_(nptype, np.uint16):
        return types_int16
    elif np.issubclass_(nptype, np.uint32):
        return types_int32
    elif np.issubclass_(nptype, np.uint64):
        return types_int64
    elif np.issubclass_(nptype, np.int):
        return types_int32
    elif np.issubclass_(nptype, np.object_):
        return types_int32
    elif np.issubclass_(nptype, np.float16):
        return types_fp16
    elif np.issubclass_(nptype, np.float32) or np.issubclass_(nptype, np.single):
        return types_fp32
    elif np.issubclass_(nptype, np.float64) or np.issubclass_(nptype, np.double):
        return types_fp64
    elif np.issubclass_(nptype, six.string_types) or np.issubclass_(nptype, np.string_) or np.issubclass_(nptype, np.str_):
        return types_str
    else:
        raise TypeError('Unsupported numpy type: %s' % nptype)

def type_to_builtin_type(type):
    if False:
        return 10
    if type.__module__ == np.__name__:
        return numpy_type_to_builtin_type(type)
    if np.issubclass_(type, bool):
        return types_bool
    elif np.issubclass_(type, six.integer_types):
        return types_int32
    elif np.issubclass_(type, six.string_types):
        return types_str
    elif np.issubclass_(type, float):
        return types_fp32
    else:
        raise TypeError('Could not determine builtin type for ' + str(type))

def numpy_val_to_builtin_val(npval):
    if False:
        print('Hello World!')
    if np.isscalar(npval):
        ret_type = type_to_builtin_type(type(npval))
        ret = ret_type()
        ret.val = npval
        return (ret, ret_type)
    else:
        builtintype = numpy_type_to_builtin_type(npval.dtype)
        from . import tensor as types_tensor
        ret_type = types_tensor(builtintype, npval.shape)
        ret = ret_type()
        ret.val = npval
        return (ret, ret_type)

def is_subtype_tensor(type1, type2):
    if False:
        return 10
    if type1.get_primitive() != type2.get_primitive():
        return False
    shape1 = type1.get_shape()
    shape2 = type2.get_shape()
    if len(shape1) != len(shape2):
        return False
    for (d1, d2) in zip(shape1, shape2):
        if d1 == d2:
            continue
        d1_is_symbolic = issubclass(type(d1), sm.Basic)
        d2_is_symbolic = issubclass(type(d2), sm.Basic)
        if d1_is_symbolic and d2_is_symbolic:
            continue
        if d1_is_symbolic and (not d2_is_symbolic):
            return False
        if not d1_is_symbolic and (not d2_is_symbolic) and (d1 != d2):
            return False
    return True

def is_subtype(type1, type2):
    if False:
        return 10
    '\n    Return True if type1 is a subtype of type2. False otherwise.\n    '
    if type2 == unknown:
        return True
    if is_list(type2):
        return is_list(type1) and is_subtype(type1.T[0], type2.T[0])
    if is_tensor(type1) and is_tensor(type2):
        return is_subtype_tensor(type1, type2)
    return type1 == type2