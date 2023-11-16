"""
Support for typing ctypes function pointers.
"""
import ctypes
import sys
from numba.core import types
from numba.core.typing import templates
from .typeof import typeof_impl
_FROM_CTYPES = {ctypes.c_bool: types.boolean, ctypes.c_int8: types.int8, ctypes.c_int16: types.int16, ctypes.c_int32: types.int32, ctypes.c_int64: types.int64, ctypes.c_uint8: types.uint8, ctypes.c_uint16: types.uint16, ctypes.c_uint32: types.uint32, ctypes.c_uint64: types.uint64, ctypes.c_float: types.float32, ctypes.c_double: types.float64, ctypes.c_void_p: types.voidptr, ctypes.py_object: types.ffi_forced_object}
_TO_CTYPES = {v: k for (k, v) in _FROM_CTYPES.items()}

def from_ctypes(ctypeobj):
    if False:
        i = 10
        return i + 15
    '\n    Convert the given ctypes type to a Numba type.\n    '
    if ctypeobj is None:
        return types.none
    assert isinstance(ctypeobj, type), ctypeobj

    def _convert_internal(ctypeobj):
        if False:
            for i in range(10):
                print('nop')
        if issubclass(ctypeobj, ctypes._Pointer):
            valuety = _convert_internal(ctypeobj._type_)
            if valuety is not None:
                return types.CPointer(valuety)
        else:
            return _FROM_CTYPES.get(ctypeobj)
    ty = _convert_internal(ctypeobj)
    if ty is None:
        raise TypeError('Unsupported ctypes type: %s' % ctypeobj)
    return ty

def to_ctypes(ty):
    if False:
        print('Hello World!')
    '\n    Convert the given Numba type to a ctypes type.\n    '
    assert isinstance(ty, types.Type), ty
    if ty is types.none:
        return None

    def _convert_internal(ty):
        if False:
            return 10
        if isinstance(ty, types.CPointer):
            return ctypes.POINTER(_convert_internal(ty.dtype))
        else:
            return _TO_CTYPES.get(ty)
    ctypeobj = _convert_internal(ty)
    if ctypeobj is None:
        raise TypeError("Cannot convert Numba type '%s' to ctypes type" % (ty,))
    return ctypeobj

def is_ctypes_funcptr(obj):
    if False:
        return 10
    try:
        ctypes.cast(obj, ctypes.c_void_p)
    except ctypes.ArgumentError:
        return False
    else:
        return hasattr(obj, 'argtypes') and hasattr(obj, 'restype')

def get_pointer(ctypes_func):
    if False:
        while True:
            i = 10
    '\n    Get a pointer to the underlying function for a ctypes function as an\n    integer.\n    '
    return ctypes.cast(ctypes_func, ctypes.c_void_p).value

def make_function_type(cfnptr):
    if False:
        print('Hello World!')
    '\n    Return a Numba type for the given ctypes function pointer.\n    '
    if cfnptr.argtypes is None:
        raise TypeError("ctypes function %r doesn't define its argument types; consider setting the `argtypes` attribute" % (cfnptr.__name__,))
    cargs = [from_ctypes(a) for a in cfnptr.argtypes]
    cret = from_ctypes(cfnptr.restype)
    if cret == types.voidptr:
        cret = types.uintp
    if sys.platform == 'win32' and (not cfnptr._flags_ & ctypes._FUNCFLAG_CDECL):
        cconv = 'x86_stdcallcc'
    else:
        cconv = None
    sig = templates.signature(cret, *cargs)
    return types.ExternalFunctionPointer(sig, cconv=cconv, get_pointer=get_pointer)