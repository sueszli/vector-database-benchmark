""" Define analogs of numpy dtypes supported by pytorch.
Define the scalar types and supported dtypes and numpy <--> torch dtype mappings.
"""
import builtins
import torch
from . import _dtypes_impl

class generic:

    @property
    def name(self):
        if False:
            while True:
                i = 10
        return self.__class__.__name__

    def __new__(cls, value):
        if False:
            while True:
                i = 10
        from ._ndarray import asarray, ndarray
        if isinstance(value, str) and value in ['inf', 'nan']:
            value = {'inf': torch.inf, 'nan': torch.nan}[value]
        if isinstance(value, ndarray):
            return value.astype(cls)
        else:
            return asarray(value, dtype=cls)

class number(generic):
    pass

class integer(number):
    pass

class inexact(number):
    pass

class signedinteger(integer):
    pass

class unsignedinteger(integer):
    pass

class floating(inexact):
    pass

class complexfloating(inexact):
    pass

class int8(signedinteger):
    name = 'int8'
    typecode = 'b'
    torch_dtype = torch.int8

class int16(signedinteger):
    name = 'int16'
    typecode = 'h'
    torch_dtype = torch.int16

class int32(signedinteger):
    name = 'int32'
    typecode = 'i'
    torch_dtype = torch.int32

class int64(signedinteger):
    name = 'int64'
    typecode = 'l'
    torch_dtype = torch.int64

class uint8(unsignedinteger):
    name = 'uint8'
    typecode = 'B'
    torch_dtype = torch.uint8

class float16(floating):
    name = 'float16'
    typecode = 'e'
    torch_dtype = torch.float16

class float32(floating):
    name = 'float32'
    typecode = 'f'
    torch_dtype = torch.float32

class float64(floating):
    name = 'float64'
    typecode = 'd'
    torch_dtype = torch.float64

class complex64(complexfloating):
    name = 'complex64'
    typecode = 'F'
    torch_dtype = torch.complex64

class complex128(complexfloating):
    name = 'complex128'
    typecode = 'D'
    torch_dtype = torch.complex128

class bool_(generic):
    name = 'bool_'
    typecode = '?'
    torch_dtype = torch.bool
_name_aliases = {'intp': int64, 'int_': int64, 'intc': int32, 'byte': int8, 'short': int16, 'longlong': int64, 'ubyte': uint8, 'half': float16, 'single': float32, 'double': float64, 'float_': float64, 'csingle': complex64, 'singlecomplex': complex64, 'cdouble': complex128, 'cfloat': complex128, 'complex_': complex128}
for (name, obj) in _name_aliases.items():
    vars()[name] = obj
sctypes = {'int': [int8, int16, int32, int64], 'uint': [uint8], 'float': [float16, float32, float64], 'complex': [complex64, complex128], 'others': [bool_]}
_names = {st.name: st for cat in sctypes for st in sctypes[cat]}
_typecodes = {st.typecode: st for cat in sctypes for st in sctypes[cat]}
_torch_dtypes = {st.torch_dtype: st for cat in sctypes for st in sctypes[cat]}
_aliases = {'u1': uint8, 'i1': int8, 'i2': int16, 'i4': int32, 'i8': int64, 'b': int8, 'f2': float16, 'f4': float32, 'f8': float64, 'c8': complex64, 'c16': complex128, 'bool_': bool_}
_python_types = {int: int64, float: float64, complex: complex128, builtins.bool: bool_, int.__name__: int64, float.__name__: float64, complex.__name__: complex128, builtins.bool.__name__: bool_}

def sctype_from_string(s):
    if False:
        for i in range(10):
            print('nop')
    "Normalize a string value: a type 'name' or a typecode or a width alias."
    if s in _names:
        return _names[s]
    if s in _name_aliases.keys():
        return _name_aliases[s]
    if s in _typecodes:
        return _typecodes[s]
    if s in _aliases:
        return _aliases[s]
    if s in _python_types:
        return _python_types[s]
    raise TypeError(f'data type {s!r} not understood')

def sctype_from_torch_dtype(torch_dtype):
    if False:
        return 10
    return _torch_dtypes[torch_dtype]

def dtype(arg):
    if False:
        i = 10
        return i + 15
    if arg is None:
        arg = _dtypes_impl.default_dtypes().float_dtype
    return DType(arg)

class DType:

    def __init__(self, arg):
        if False:
            return 10
        if isinstance(arg, torch.dtype):
            sctype = _torch_dtypes[arg]
        elif isinstance(arg, torch.Tensor):
            sctype = _torch_dtypes[arg.dtype]
        elif issubclass_(arg, generic):
            sctype = arg
        elif isinstance(arg, DType):
            sctype = arg._scalar_type
        elif hasattr(arg, 'dtype'):
            sctype = arg.dtype._scalar_type
        else:
            sctype = sctype_from_string(arg)
        self._scalar_type = sctype

    @property
    def name(self):
        if False:
            return 10
        return self._scalar_type.name

    @property
    def type(self):
        if False:
            i = 10
            return i + 15
        return self._scalar_type

    @property
    def kind(self):
        if False:
            while True:
                i = 10
        return _torch_dtypes[self.torch_dtype].name[0]

    @property
    def typecode(self):
        if False:
            return 10
        return self._scalar_type.typecode

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        if isinstance(other, DType):
            return self._scalar_type == other._scalar_type
        try:
            other_instance = DType(other)
        except TypeError:
            return False
        return self._scalar_type == other_instance._scalar_type

    @property
    def torch_dtype(self):
        if False:
            print('Hello World!')
        return self._scalar_type.torch_dtype

    def __hash__(self):
        if False:
            print('Hello World!')
        return hash(self._scalar_type.name)

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return f'dtype("{self.name}")'
    __str__ = __repr__

    @property
    def itemsize(self):
        if False:
            for i in range(10):
                print('nop')
        elem = self.type(1)
        return elem.tensor.element_size()

    def __getstate__(self):
        if False:
            for i in range(10):
                print('nop')
        return self._scalar_type

    def __setstate__(self, value):
        if False:
            for i in range(10):
                print('nop')
        self._scalar_type = value
typecodes = {'All': 'efdFDBbhil?', 'AllFloat': 'efdFD', 'AllInteger': 'Bbhil', 'Integer': 'bhil', 'UnsignedInteger': 'B', 'Float': 'efd', 'Complex': 'FD'}

def set_default_dtype(fp_dtype='numpy', int_dtype='numpy'):
    if False:
        for i in range(10):
            print('nop')
    'Set the (global) defaults for fp, complex, and int dtypes.\n\n    The complex dtype is inferred from the float (fp) dtype. It has\n    a width at least twice the width of the float dtype,\n    i.e., it\'s complex128 for float64 and complex64 for float32.\n\n    Parameters\n    ----------\n    fp_dtype\n        Allowed values are "numpy", "pytorch" or dtype_like things which\n        can be converted into a DType instance.\n        Default is "numpy" (i.e. float64).\n    int_dtype\n        Allowed values are "numpy", "pytorch" or dtype_like things which\n        can be converted into a DType instance.\n        Default is "numpy" (i.e. int64).\n\n    Returns\n    -------\n    The old default dtype state: a namedtuple with attributes ``float_dtype``,\n    ``complex_dtypes`` and ``int_dtype``. These attributes store *pytorch*\n    dtypes.\n\n    Notes\n    ------------\n    This functions has a side effect: it sets the global state with the provided dtypes.\n\n    The complex dtype has bit width of at least twice the width of the float\n    dtype, i.e. it\'s complex128 for float64 and complex64 for float32.\n\n    '
    if fp_dtype not in ['numpy', 'pytorch']:
        fp_dtype = dtype(fp_dtype).torch_dtype
    if int_dtype not in ['numpy', 'pytorch']:
        int_dtype = dtype(int_dtype).torch_dtype
    if fp_dtype == 'numpy':
        float_dtype = torch.float64
    elif fp_dtype == 'pytorch':
        float_dtype = torch.float32
    else:
        float_dtype = fp_dtype
    complex_dtype = {torch.float64: torch.complex128, torch.float32: torch.complex64, torch.float16: torch.complex64}[float_dtype]
    if int_dtype in ['numpy', 'pytorch']:
        int_dtype = torch.int64
    else:
        int_dtype = int_dtype
    new_defaults = _dtypes_impl.DefaultDTypes(float_dtype=float_dtype, complex_dtype=complex_dtype, int_dtype=int_dtype)
    old_defaults = _dtypes_impl.default_dtypes
    _dtypes_impl._default_dtypes = new_defaults
    return old_defaults

def issubclass_(arg, klass):
    if False:
        while True:
            i = 10
    try:
        return issubclass(arg, klass)
    except TypeError:
        return False

def issubdtype(arg1, arg2):
    if False:
        while True:
            i = 10
    if not issubclass_(arg1, generic):
        arg1 = dtype(arg1).type
    if not issubclass_(arg2, generic):
        arg2 = dtype(arg2).type
    return issubclass(arg1, arg2)
__all__ = ['dtype', 'DType', 'typecodes', 'issubdtype', 'set_default_dtype']
__all__ += list(_names.keys())
__all__ += list(_name_aliases.keys())
__all__ += ['sctypes', 'generic', 'number', 'integer', 'signedinteger', 'unsignedinteger', 'inexact', 'floating', 'complexfloating']