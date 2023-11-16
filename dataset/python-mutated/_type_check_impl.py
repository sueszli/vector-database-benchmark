"""Automatically adapted for numpy Sep 19, 2005 by convertcode.py

"""
import functools
__all__ = ['iscomplexobj', 'isrealobj', 'imag', 'iscomplex', 'isreal', 'nan_to_num', 'real', 'real_if_close', 'typename', 'mintypecode', 'common_type']
from .._utils import set_module
import numpy._core.numeric as _nx
from numpy._core.numeric import asarray, asanyarray, isnan, zeros
from numpy._core import overrides, getlimits
from ._ufunclike_impl import isneginf, isposinf
array_function_dispatch = functools.partial(overrides.array_function_dispatch, module='numpy')
_typecodes_by_elsize = 'GDFgdfQqLlIiHhBb?'

@set_module('numpy')
def mintypecode(typechars, typeset='GDFgdf', default='d'):
    if False:
        return 10
    "\n    Return the character for the minimum-size type to which given types can\n    be safely cast.\n\n    The returned type character must represent the smallest size dtype such\n    that an array of the returned type can handle the data from an array of\n    all types in `typechars` (or if `typechars` is an array, then its\n    dtype.char).\n\n    Parameters\n    ----------\n    typechars : list of str or array_like\n        If a list of strings, each string should represent a dtype.\n        If array_like, the character representation of the array dtype is used.\n    typeset : str or list of str, optional\n        The set of characters that the returned character is chosen from.\n        The default set is 'GDFgdf'.\n    default : str, optional\n        The default character, this is returned if none of the characters in\n        `typechars` matches a character in `typeset`.\n\n    Returns\n    -------\n    typechar : str\n        The character representing the minimum-size type that was found.\n\n    See Also\n    --------\n    dtype\n\n    Examples\n    --------\n    >>> np.mintypecode(['d', 'f', 'S'])\n    'd'\n    >>> x = np.array([1.1, 2-3.j])\n    >>> np.mintypecode(x)\n    'D'\n\n    >>> np.mintypecode('abceh', default='G')\n    'G'\n\n    "
    typecodes = (isinstance(t, str) and t or asarray(t).dtype.char for t in typechars)
    intersection = set((t for t in typecodes if t in typeset))
    if not intersection:
        return default
    if 'F' in intersection and 'd' in intersection:
        return 'D'
    return min(intersection, key=_typecodes_by_elsize.index)

def _real_dispatcher(val):
    if False:
        for i in range(10):
            print('nop')
    return (val,)

@array_function_dispatch(_real_dispatcher)
def real(val):
    if False:
        return 10
    '\n    Return the real part of the complex argument.\n\n    Parameters\n    ----------\n    val : array_like\n        Input array.\n\n    Returns\n    -------\n    out : ndarray or scalar\n        The real component of the complex argument. If `val` is real, the type\n        of `val` is used for the output.  If `val` has complex elements, the\n        returned type is float.\n\n    See Also\n    --------\n    real_if_close, imag, angle\n\n    Examples\n    --------\n    >>> a = np.array([1+2j, 3+4j, 5+6j])\n    >>> a.real\n    array([1.,  3.,  5.])\n    >>> a.real = 9\n    >>> a\n    array([9.+2.j,  9.+4.j,  9.+6.j])\n    >>> a.real = np.array([9, 8, 7])\n    >>> a\n    array([9.+2.j,  8.+4.j,  7.+6.j])\n    >>> np.real(1 + 1j)\n    1.0\n\n    '
    try:
        return val.real
    except AttributeError:
        return asanyarray(val).real

def _imag_dispatcher(val):
    if False:
        return 10
    return (val,)

@array_function_dispatch(_imag_dispatcher)
def imag(val):
    if False:
        i = 10
        return i + 15
    '\n    Return the imaginary part of the complex argument.\n\n    Parameters\n    ----------\n    val : array_like\n        Input array.\n\n    Returns\n    -------\n    out : ndarray or scalar\n        The imaginary component of the complex argument. If `val` is real,\n        the type of `val` is used for the output.  If `val` has complex\n        elements, the returned type is float.\n\n    See Also\n    --------\n    real, angle, real_if_close\n\n    Examples\n    --------\n    >>> a = np.array([1+2j, 3+4j, 5+6j])\n    >>> a.imag\n    array([2.,  4.,  6.])\n    >>> a.imag = np.array([8, 10, 12])\n    >>> a\n    array([1. +8.j,  3.+10.j,  5.+12.j])\n    >>> np.imag(1 + 1j)\n    1.0\n\n    '
    try:
        return val.imag
    except AttributeError:
        return asanyarray(val).imag

def _is_type_dispatcher(x):
    if False:
        for i in range(10):
            print('nop')
    return (x,)

@array_function_dispatch(_is_type_dispatcher)
def iscomplex(x):
    if False:
        i = 10
        return i + 15
    '\n    Returns a bool array, where True if input element is complex.\n\n    What is tested is whether the input has a non-zero imaginary part, not if\n    the input type is complex.\n\n    Parameters\n    ----------\n    x : array_like\n        Input array.\n\n    Returns\n    -------\n    out : ndarray of bools\n        Output array.\n\n    See Also\n    --------\n    isreal\n    iscomplexobj : Return True if x is a complex type or an array of complex\n                   numbers.\n\n    Examples\n    --------\n    >>> np.iscomplex([1+1j, 1+0j, 4.5, 3, 2, 2j])\n    array([ True, False, False, False, False,  True])\n\n    '
    ax = asanyarray(x)
    if issubclass(ax.dtype.type, _nx.complexfloating):
        return ax.imag != 0
    res = zeros(ax.shape, bool)
    return res[()]

@array_function_dispatch(_is_type_dispatcher)
def isreal(x):
    if False:
        while True:
            i = 10
    '\n    Returns a bool array, where True if input element is real.\n\n    If element has complex type with zero imaginary part, the return value\n    for that element is True.\n\n    Parameters\n    ----------\n    x : array_like\n        Input array.\n\n    Returns\n    -------\n    out : ndarray, bool\n        Boolean array of same shape as `x`.\n\n    Notes\n    -----\n    `isreal` may behave unexpectedly for string or object arrays (see examples)\n\n    See Also\n    --------\n    iscomplex\n    isrealobj : Return True if x is not a complex type.\n\n    Examples\n    --------\n    >>> a = np.array([1+1j, 1+0j, 4.5, 3, 2, 2j], dtype=complex)\n    >>> np.isreal(a)\n    array([False,  True,  True,  True,  True, False])\n\n    The function does not work on string arrays.\n\n    >>> a = np.array([2j, "a"], dtype="U")\n    >>> np.isreal(a)  # Warns about non-elementwise comparison\n    False\n\n    Returns True for all elements in input array of ``dtype=object`` even if\n    any of the elements is complex.\n\n    >>> a = np.array([1, "2", 3+4j], dtype=object)\n    >>> np.isreal(a)\n    array([ True,  True,  True])\n\n    isreal should not be used with object arrays\n\n    >>> a = np.array([1+2j, 2+1j], dtype=object)\n    >>> np.isreal(a)\n    array([ True,  True])\n\n    '
    return imag(x) == 0

@array_function_dispatch(_is_type_dispatcher)
def iscomplexobj(x):
    if False:
        for i in range(10):
            print('nop')
    '\n    Check for a complex type or an array of complex numbers.\n\n    The type of the input is checked, not the value. Even if the input\n    has an imaginary part equal to zero, `iscomplexobj` evaluates to True.\n\n    Parameters\n    ----------\n    x : any\n        The input can be of any type and shape.\n\n    Returns\n    -------\n    iscomplexobj : bool\n        The return value, True if `x` is of a complex type or has at least\n        one complex element.\n\n    See Also\n    --------\n    isrealobj, iscomplex\n\n    Examples\n    --------\n    >>> np.iscomplexobj(1)\n    False\n    >>> np.iscomplexobj(1+0j)\n    True\n    >>> np.iscomplexobj([3, 1+0j, True])\n    True\n\n    '
    try:
        dtype = x.dtype
        type_ = dtype.type
    except AttributeError:
        type_ = asarray(x).dtype.type
    return issubclass(type_, _nx.complexfloating)

@array_function_dispatch(_is_type_dispatcher)
def isrealobj(x):
    if False:
        i = 10
        return i + 15
    "\n    Return True if x is a not complex type or an array of complex numbers.\n\n    The type of the input is checked, not the value. So even if the input\n    has an imaginary part equal to zero, `isrealobj` evaluates to False\n    if the data type is complex.\n\n    Parameters\n    ----------\n    x : any\n        The input can be of any type and shape.\n\n    Returns\n    -------\n    y : bool\n        The return value, False if `x` is of a complex type.\n\n    See Also\n    --------\n    iscomplexobj, isreal\n\n    Notes\n    -----\n    The function is only meant for arrays with numerical values but it\n    accepts all other objects. Since it assumes array input, the return\n    value of other objects may be True.\n\n    >>> np.isrealobj('A string')\n    True\n    >>> np.isrealobj(False)\n    True\n    >>> np.isrealobj(None)\n    True\n\n    Examples\n    --------\n    >>> np.isrealobj(1)\n    True\n    >>> np.isrealobj(1+0j)\n    False\n    >>> np.isrealobj([3, 1+0j, True])\n    False\n\n    "
    return not iscomplexobj(x)

def _getmaxmin(t):
    if False:
        while True:
            i = 10
    from numpy._core import getlimits
    f = getlimits.finfo(t)
    return (f.max, f.min)

def _nan_to_num_dispatcher(x, copy=None, nan=None, posinf=None, neginf=None):
    if False:
        for i in range(10):
            print('nop')
    return (x,)

@array_function_dispatch(_nan_to_num_dispatcher)
def nan_to_num(x, copy=True, nan=0.0, posinf=None, neginf=None):
    if False:
        print('Hello World!')
    '\n    Replace NaN with zero and infinity with large finite numbers (default\n    behaviour) or with the numbers defined by the user using the `nan`,\n    `posinf` and/or `neginf` keywords.\n\n    If `x` is inexact, NaN is replaced by zero or by the user defined value in\n    `nan` keyword, infinity is replaced by the largest finite floating point\n    values representable by ``x.dtype`` or by the user defined value in\n    `posinf` keyword and -infinity is replaced by the most negative finite\n    floating point values representable by ``x.dtype`` or by the user defined\n    value in `neginf` keyword.\n\n    For complex dtypes, the above is applied to each of the real and\n    imaginary components of `x` separately.\n\n    If `x` is not inexact, then no replacements are made.\n\n    Parameters\n    ----------\n    x : scalar or array_like\n        Input data.\n    copy : bool, optional\n        Whether to create a copy of `x` (True) or to replace values\n        in-place (False). The in-place operation only occurs if\n        casting to an array does not require a copy.\n        Default is True.\n\n        .. versionadded:: 1.13\n    nan : int, float, optional\n        Value to be used to fill NaN values. If no value is passed\n        then NaN values will be replaced with 0.0.\n\n        .. versionadded:: 1.17\n    posinf : int, float, optional\n        Value to be used to fill positive infinity values. If no value is\n        passed then positive infinity values will be replaced with a very\n        large number.\n\n        .. versionadded:: 1.17\n    neginf : int, float, optional\n        Value to be used to fill negative infinity values. If no value is\n        passed then negative infinity values will be replaced with a very\n        small (or negative) number.\n\n        .. versionadded:: 1.17\n\n\n\n    Returns\n    -------\n    out : ndarray\n        `x`, with the non-finite values replaced. If `copy` is False, this may\n        be `x` itself.\n\n    See Also\n    --------\n    isinf : Shows which elements are positive or negative infinity.\n    isneginf : Shows which elements are negative infinity.\n    isposinf : Shows which elements are positive infinity.\n    isnan : Shows which elements are Not a Number (NaN).\n    isfinite : Shows which elements are finite (not NaN, not infinity)\n\n    Notes\n    -----\n    NumPy uses the IEEE Standard for Binary Floating-Point for Arithmetic\n    (IEEE 754). This means that Not a Number is not equivalent to infinity.\n\n    Examples\n    --------\n    >>> np.nan_to_num(np.inf)\n    1.7976931348623157e+308\n    >>> np.nan_to_num(-np.inf)\n    -1.7976931348623157e+308\n    >>> np.nan_to_num(np.nan)\n    0.0\n    >>> x = np.array([np.inf, -np.inf, np.nan, -128, 128])\n    >>> np.nan_to_num(x)\n    array([ 1.79769313e+308, -1.79769313e+308,  0.00000000e+000, # may vary\n           -1.28000000e+002,  1.28000000e+002])\n    >>> np.nan_to_num(x, nan=-9999, posinf=33333333, neginf=33333333)\n    array([ 3.3333333e+07,  3.3333333e+07, -9.9990000e+03,\n           -1.2800000e+02,  1.2800000e+02])\n    >>> y = np.array([complex(np.inf, np.nan), np.nan, complex(np.nan, np.inf)])\n    array([  1.79769313e+308,  -1.79769313e+308,   0.00000000e+000, # may vary\n         -1.28000000e+002,   1.28000000e+002])\n    >>> np.nan_to_num(y)\n    array([  1.79769313e+308 +0.00000000e+000j, # may vary\n             0.00000000e+000 +0.00000000e+000j,\n             0.00000000e+000 +1.79769313e+308j])\n    >>> np.nan_to_num(y, nan=111111, posinf=222222)\n    array([222222.+111111.j, 111111.     +0.j, 111111.+222222.j])\n    '
    x = _nx.array(x, subok=True, copy=copy)
    xtype = x.dtype.type
    isscalar = x.ndim == 0
    if not issubclass(xtype, _nx.inexact):
        return x[()] if isscalar else x
    iscomplex = issubclass(xtype, _nx.complexfloating)
    dest = (x.real, x.imag) if iscomplex else (x,)
    (maxf, minf) = _getmaxmin(x.real.dtype)
    if posinf is not None:
        maxf = posinf
    if neginf is not None:
        minf = neginf
    for d in dest:
        idx_nan = isnan(d)
        idx_posinf = isposinf(d)
        idx_neginf = isneginf(d)
        _nx.copyto(d, nan, where=idx_nan)
        _nx.copyto(d, maxf, where=idx_posinf)
        _nx.copyto(d, minf, where=idx_neginf)
    return x[()] if isscalar else x

def _real_if_close_dispatcher(a, tol=None):
    if False:
        print('Hello World!')
    return (a,)

@array_function_dispatch(_real_if_close_dispatcher)
def real_if_close(a, tol=100):
    if False:
        return 10
    '\n    If input is complex with all imaginary parts close to zero, return\n    real parts.\n\n    "Close to zero" is defined as `tol` * (machine epsilon of the type for\n    `a`).\n\n    Parameters\n    ----------\n    a : array_like\n        Input array.\n    tol : float\n        Tolerance in machine epsilons for the complex part of the elements\n        in the array. If the tolerance is <=1, then the absolute tolerance\n        is used.\n\n    Returns\n    -------\n    out : ndarray\n        If `a` is real, the type of `a` is used for the output.  If `a`\n        has complex elements, the returned type is float.\n\n    See Also\n    --------\n    real, imag, angle\n\n    Notes\n    -----\n    Machine epsilon varies from machine to machine and between data types\n    but Python floats on most platforms have a machine epsilon equal to\n    2.2204460492503131e-16.  You can use \'np.finfo(float).eps\' to print\n    out the machine epsilon for floats.\n\n    Examples\n    --------\n    >>> np.finfo(float).eps\n    2.2204460492503131e-16 # may vary\n\n    >>> np.real_if_close([2.1 + 4e-14j, 5.2 + 3e-15j], tol=1000)\n    array([2.1, 5.2])\n    >>> np.real_if_close([2.1 + 4e-13j, 5.2 + 3e-15j], tol=1000)\n    array([2.1+4.e-13j, 5.2 + 3e-15j])\n\n    '
    a = asanyarray(a)
    type_ = a.dtype.type
    if not issubclass(type_, _nx.complexfloating):
        return a
    if tol > 1:
        f = getlimits.finfo(type_)
        tol = f.eps * tol
    if _nx.all(_nx.absolute(a.imag) < tol):
        a = a.real
    return a
_namefromtype = {'S1': 'character', '?': 'bool', 'b': 'signed char', 'B': 'unsigned char', 'h': 'short', 'H': 'unsigned short', 'i': 'integer', 'I': 'unsigned integer', 'l': 'long integer', 'L': 'unsigned long integer', 'q': 'long long integer', 'Q': 'unsigned long long integer', 'f': 'single precision', 'd': 'double precision', 'g': 'long precision', 'F': 'complex single precision', 'D': 'complex double precision', 'G': 'complex long double precision', 'S': 'string', 'U': 'unicode', 'V': 'void', 'O': 'object'}

@set_module('numpy')
def typename(char):
    if False:
        while True:
            i = 10
    "\n    Return a description for the given data type code.\n\n    Parameters\n    ----------\n    char : str\n        Data type code.\n\n    Returns\n    -------\n    out : str\n        Description of the input data type code.\n\n    See Also\n    --------\n    dtype\n\n    Examples\n    --------\n    >>> typechars = ['S1', '?', 'B', 'D', 'G', 'F', 'I', 'H', 'L', 'O', 'Q',\n    ...              'S', 'U', 'V', 'b', 'd', 'g', 'f', 'i', 'h', 'l', 'q']\n    >>> for typechar in typechars:\n    ...     print(typechar, ' : ', np.typename(typechar))\n    ...\n    S1  :  character\n    ?  :  bool\n    B  :  unsigned char\n    D  :  complex double precision\n    G  :  complex long double precision\n    F  :  complex single precision\n    I  :  unsigned integer\n    H  :  unsigned short\n    L  :  unsigned long integer\n    O  :  object\n    Q  :  unsigned long long integer\n    S  :  string\n    U  :  unicode\n    V  :  void\n    b  :  signed char\n    d  :  double precision\n    g  :  long precision\n    f  :  single precision\n    i  :  integer\n    h  :  short\n    l  :  long integer\n    q  :  long long integer\n\n    "
    return _namefromtype[char]
array_type = [[_nx.float16, _nx.float32, _nx.float64, _nx.longdouble], [None, _nx.complex64, _nx.complex128, _nx.clongdouble]]
array_precision = {_nx.float16: 0, _nx.float32: 1, _nx.float64: 2, _nx.longdouble: 3, _nx.complex64: 1, _nx.complex128: 2, _nx.clongdouble: 3}

def _common_type_dispatcher(*arrays):
    if False:
        i = 10
        return i + 15
    return arrays

@array_function_dispatch(_common_type_dispatcher)
def common_type(*arrays):
    if False:
        for i in range(10):
            print('nop')
    "\n    Return a scalar type which is common to the input arrays.\n\n    The return type will always be an inexact (i.e. floating point) scalar\n    type, even if all the arrays are integer arrays. If one of the inputs is\n    an integer array, the minimum precision type that is returned is a\n    64-bit floating point dtype.\n\n    All input arrays except int64 and uint64 can be safely cast to the\n    returned dtype without loss of information.\n\n    Parameters\n    ----------\n    array1, array2, ... : ndarrays\n        Input arrays.\n\n    Returns\n    -------\n    out : data type code\n        Data type code.\n\n    See Also\n    --------\n    dtype, mintypecode\n\n    Examples\n    --------\n    >>> np.common_type(np.arange(2, dtype=np.float32))\n    <class 'numpy.float32'>\n    >>> np.common_type(np.arange(2, dtype=np.float32), np.arange(2))\n    <class 'numpy.float64'>\n    >>> np.common_type(np.arange(4), np.array([45, 6.j]), np.array([45.0]))\n    <class 'numpy.complex128'>\n\n    "
    is_complex = False
    precision = 0
    for a in arrays:
        t = a.dtype.type
        if iscomplexobj(a):
            is_complex = True
        if issubclass(t, _nx.integer):
            p = 2
        else:
            p = array_precision.get(t, None)
            if p is None:
                raise TypeError("can't get common type for non-numeric array")
        precision = max(precision, p)
    if is_complex:
        return array_type[1][precision]
    else:
        return array_type[0][precision]