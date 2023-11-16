"""
numerictypes: Define the numeric type objects

This module is designed so "from numerictypes import \\*" is safe.
Exported symbols include:

  Dictionary with all registered number types (including aliases):
    sctypeDict

  Type objects (not all will be available, depends on platform):
      see variable sctypes for which ones you have

    Bit-width names

    int8 int16 int32 int64 int128
    uint8 uint16 uint32 uint64 uint128
    float16 float32 float64 float96 float128 float256
    complex32 complex64 complex128 complex192 complex256 complex512
    datetime64 timedelta64

    c-based names

    bool_

    object_

    void, str_

    byte, ubyte,
    short, ushort
    intc, uintc,
    intp, uintp,
    int_, uint,
    longlong, ulonglong,

    single, csingle,
    double, cdouble,
    longdouble, clongdouble,

   As part of the type-hierarchy:    xx -- is bit-width

   generic
     +-> bool_                                  (kind=b)
     +-> number
     |   +-> integer
     |   |   +-> signedinteger     (intxx)      (kind=i)
     |   |   |     byte
     |   |   |     short
     |   |   |     intc
     |   |   |     intp
     |   |   |     int_
     |   |   |     longlong
     |   |   \\-> unsignedinteger  (uintxx)     (kind=u)
     |   |         ubyte
     |   |         ushort
     |   |         uintc
     |   |         uintp
     |   |         uint
     |   |         ulonglong
     |   +-> inexact
     |       +-> floating          (floatxx)    (kind=f)
     |       |     half
     |       |     single
     |       |     double
     |       |     longdouble
     |       \\-> complexfloating  (complexxx)  (kind=c)
     |             csingle
     |             cdouble
     |             clongdouble
     +-> flexible
     |   +-> character
     |   |     bytes_                           (kind=S)
     |   |     str_                             (kind=U)
     |   |
     |   \\-> void                              (kind=V)
     \\-> object_ (not used much)               (kind=O)

"""
import numbers
import warnings
from .multiarray import ndarray, array, dtype, datetime_data, datetime_as_string, busday_offset, busday_count, is_busday, busdaycalendar
from .._utils import set_module
__all__ = ['ScalarType', 'typecodes', 'issubdtype', 'datetime_data', 'datetime_as_string', 'busday_offset', 'busday_count', 'is_busday', 'busdaycalendar']
from ._string_helpers import english_lower, english_upper, english_capitalize, LOWER_TABLE, UPPER_TABLE
from ._type_aliases import sctypeDict, allTypes, sctypes
from ._dtype import _kind_name
from builtins import bool, int, float, complex, object, str, bytes
generic = allTypes['generic']
genericTypeRank = ['bool', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64', 'int128', 'uint128', 'float16', 'float32', 'float64', 'float80', 'float96', 'float128', 'float256', 'complex32', 'complex64', 'complex128', 'complex160', 'complex192', 'complex256', 'complex512', 'object']

@set_module('numpy')
def maximum_sctype(t):
    if False:
        return 10
    "\n    Return the scalar type of highest precision of the same kind as the input.\n\n    .. deprecated:: 2.0\n        Use an explicit dtype like int64 or float64 instead.\n\n    Parameters\n    ----------\n    t : dtype or dtype specifier\n        The input data type. This can be a `dtype` object or an object that\n        is convertible to a `dtype`.\n\n    Returns\n    -------\n    out : dtype\n        The highest precision data type of the same kind (`dtype.kind`) as `t`.\n\n    See Also\n    --------\n    obj2sctype, mintypecode, sctype2char\n    dtype\n\n    Examples\n    --------\n    >>> from numpy._core.numerictypes import maximum_sctype\n    >>> maximum_sctype(int)\n    <class 'numpy.int64'>\n    >>> maximum_sctype(np.uint8)\n    <class 'numpy.uint64'>\n    >>> maximum_sctype(complex)\n    <class 'numpy.complex256'> # may vary\n\n    >>> maximum_sctype(str)\n    <class 'numpy.str_'>\n\n    >>> maximum_sctype('i2')\n    <class 'numpy.int64'>\n    >>> maximum_sctype('f4')\n    <class 'numpy.float128'> # may vary\n\n    "
    warnings.warn('`maximum_sctype` is deprecated. Use an explicit dtype like int64 or float64 instead. (deprecated in NumPy 2.0)', DeprecationWarning, stacklevel=2)
    g = obj2sctype(t)
    if g is None:
        return t
    t = g
    base = _kind_name(dtype(t))
    if base in sctypes:
        return sctypes[base][-1]
    else:
        return t

@set_module('numpy')
def issctype(rep):
    if False:
        print('Hello World!')
    "\n    Determines whether the given object represents a scalar data-type.\n\n    Parameters\n    ----------\n    rep : any\n        If `rep` is an instance of a scalar dtype, True is returned. If not,\n        False is returned.\n\n    Returns\n    -------\n    out : bool\n        Boolean result of check whether `rep` is a scalar dtype.\n\n    See Also\n    --------\n    issubsctype, issubdtype, obj2sctype, sctype2char\n\n    Examples\n    --------\n    >>> from numpy._core.numerictypes import issctype\n    >>> issctype(np.int32)\n    True\n    >>> issctype(list)\n    False\n    >>> issctype(1.1)\n    False\n\n    Strings are also a scalar type:\n\n    >>> issctype(np.dtype('str'))\n    True\n\n    "
    if not isinstance(rep, (type, dtype)):
        return False
    try:
        res = obj2sctype(rep)
        if res and res != object_:
            return True
        return False
    except Exception:
        return False

@set_module('numpy')
def obj2sctype(rep, default=None):
    if False:
        i = 10
        return i + 15
    "\n    Return the scalar dtype or NumPy equivalent of Python type of an object.\n\n    Parameters\n    ----------\n    rep : any\n        The object of which the type is returned.\n    default : any, optional\n        If given, this is returned for objects whose types can not be\n        determined. If not given, None is returned for those objects.\n\n    Returns\n    -------\n    dtype : dtype or Python type\n        The data type of `rep`.\n\n    See Also\n    --------\n    sctype2char, issctype, issubsctype, issubdtype\n\n    Examples\n    --------\n    >>> from numpy._core.numerictypes import obj2sctype\n    >>> obj2sctype(np.int32)\n    <class 'numpy.int32'>\n    >>> obj2sctype(np.array([1., 2.]))\n    <class 'numpy.float64'>\n    >>> obj2sctype(np.array([1.j]))\n    <class 'numpy.complex128'>\n\n    >>> obj2sctype(dict)\n    <class 'numpy.object_'>\n    >>> obj2sctype('string')\n\n    >>> obj2sctype(1, default=list)\n    <class 'list'>\n\n    "
    if isinstance(rep, type) and issubclass(rep, generic):
        return rep
    if isinstance(rep, ndarray):
        return rep.dtype.type
    try:
        res = dtype(rep)
    except Exception:
        return default
    else:
        return res.type

@set_module('numpy')
def issubclass_(arg1, arg2):
    if False:
        print('Hello World!')
    '\n    Determine if a class is a subclass of a second class.\n\n    `issubclass_` is equivalent to the Python built-in ``issubclass``,\n    except that it returns False instead of raising a TypeError if one\n    of the arguments is not a class.\n\n    Parameters\n    ----------\n    arg1 : class\n        Input class. True is returned if `arg1` is a subclass of `arg2`.\n    arg2 : class or tuple of classes.\n        Input class. If a tuple of classes, True is returned if `arg1` is a\n        subclass of any of the tuple elements.\n\n    Returns\n    -------\n    out : bool\n        Whether `arg1` is a subclass of `arg2` or not.\n\n    See Also\n    --------\n    issubsctype, issubdtype, issctype\n\n    Examples\n    --------\n    >>> np.issubclass_(np.int32, int)\n    False\n    >>> np.issubclass_(np.int32, float)\n    False\n    >>> np.issubclass_(np.float64, float)\n    True\n\n    '
    try:
        return issubclass(arg1, arg2)
    except TypeError:
        return False

@set_module('numpy')
def issubsctype(arg1, arg2):
    if False:
        print('Hello World!')
    "\n    Determine if the first argument is a subclass of the second argument.\n\n    Parameters\n    ----------\n    arg1, arg2 : dtype or dtype specifier\n        Data-types.\n\n    Returns\n    -------\n    out : bool\n        The result.\n\n    See Also\n    --------\n    issctype, issubdtype, obj2sctype\n\n    Examples\n    --------\n    >>> from numpy._core import issubsctype\n    >>> issubsctype('S8', str)\n    False\n    >>> issubsctype(np.array([1]), int)\n    True\n    >>> issubsctype(np.array([1]), float)\n    False\n\n    "
    return issubclass(obj2sctype(arg1), obj2sctype(arg2))

@set_module('numpy')
def issubdtype(arg1, arg2):
    if False:
        while True:
            i = 10
    "\n    Returns True if first argument is a typecode lower/equal in type hierarchy.\n\n    This is like the builtin :func:`issubclass`, but for `dtype`\\ s.\n\n    Parameters\n    ----------\n    arg1, arg2 : dtype_like\n        `dtype` or object coercible to one\n\n    Returns\n    -------\n    out : bool\n\n    See Also\n    --------\n    :ref:`arrays.scalars` : Overview of the numpy type hierarchy.\n\n    Examples\n    --------\n    `issubdtype` can be used to check the type of arrays:\n\n    >>> ints = np.array([1, 2, 3], dtype=np.int32)\n    >>> np.issubdtype(ints.dtype, np.integer)\n    True\n    >>> np.issubdtype(ints.dtype, np.floating)\n    False\n\n    >>> floats = np.array([1, 2, 3], dtype=np.float32)\n    >>> np.issubdtype(floats.dtype, np.integer)\n    False\n    >>> np.issubdtype(floats.dtype, np.floating)\n    True\n\n    Similar types of different sizes are not subdtypes of each other:\n\n    >>> np.issubdtype(np.float64, np.float32)\n    False\n    >>> np.issubdtype(np.float32, np.float64)\n    False\n\n    but both are subtypes of `floating`:\n\n    >>> np.issubdtype(np.float64, np.floating)\n    True\n    >>> np.issubdtype(np.float32, np.floating)\n    True\n\n    For convenience, dtype-like objects are allowed too:\n\n    >>> np.issubdtype('S1', np.bytes_)\n    True\n    >>> np.issubdtype('i4', np.signedinteger)\n    True\n\n    "
    if not issubclass_(arg1, generic):
        arg1 = dtype(arg1).type
    if not issubclass_(arg2, generic):
        arg2 = dtype(arg2).type
    return issubclass(arg1, arg2)

@set_module('numpy')
def sctype2char(sctype):
    if False:
        i = 10
        return i + 15
    "\n    Return the string representation of a scalar dtype.\n\n    Parameters\n    ----------\n    sctype : scalar dtype or object\n        If a scalar dtype, the corresponding string character is\n        returned. If an object, `sctype2char` tries to infer its scalar type\n        and then return the corresponding string character.\n\n    Returns\n    -------\n    typechar : str\n        The string character corresponding to the scalar type.\n\n    Raises\n    ------\n    ValueError\n        If `sctype` is an object for which the type can not be inferred.\n\n    See Also\n    --------\n    obj2sctype, issctype, issubsctype, mintypecode\n\n    Examples\n    --------\n    >>> from numpy._core.numerictypes import sctype2char\n    >>> for sctype in [np.int32, np.double, np.cdouble, np.bytes_, np.ndarray]:\n    ...     print(sctype2char(sctype))\n    l # may vary\n    d\n    D\n    S\n    O\n\n    >>> x = np.array([1., 2-1.j])\n    >>> sctype2char(x)\n    'D'\n    >>> sctype2char(list)\n    'O'\n\n    "
    sctype = obj2sctype(sctype)
    if sctype is None:
        raise ValueError('unrecognized type')
    if sctype not in sctypeDict.values():
        raise KeyError(sctype)
    return dtype(sctype).char

def _scalar_type_key(typ):
    if False:
        while True:
            i = 10
    'A ``key`` function for `sorted`.'
    dt = dtype(typ)
    return (dt.kind.lower(), dt.itemsize)
ScalarType = [int, float, complex, bool, bytes, str, memoryview]
ScalarType += sorted(set(sctypeDict.values()), key=_scalar_type_key)
ScalarType = tuple(ScalarType)
for key in allTypes:
    globals()[key] = allTypes[key]
    __all__.append(key)
del key
typecodes = {'Character': 'c', 'Integer': 'bhilqp', 'UnsignedInteger': 'BHILQP', 'Float': 'efdg', 'Complex': 'FDG', 'AllInteger': 'bBhHiIlLqQpP', 'AllFloat': 'efdgFDG', 'Datetime': 'Mm', 'All': '?bhilqpBHILQPefdgFDGSUVOMm'}
typeDict = sctypeDict

def _register_types():
    if False:
        for i in range(10):
            print('nop')
    numbers.Integral.register(integer)
    numbers.Complex.register(inexact)
    numbers.Real.register(floating)
    numbers.Number.register(number)
_register_types()