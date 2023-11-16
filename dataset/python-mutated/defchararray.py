"""
This module contains a set of functions for vectorized string
operations and methods.

.. note::
   The `chararray` class exists for backwards compatibility with
   Numarray, it is not recommended for new development. Starting from numpy
   1.4, if one needs arrays of strings, it is recommended to use arrays of
   `dtype` `object_`, `bytes_` or `str_`, and use the free functions
   in the `numpy.char` module for fast vectorized string operations.

Some methods will only be available if the corresponding string method is
available in your version of Python.

The preferred alias for `defchararray` is `numpy.char`.

"""
import functools
from .._utils import set_module
from .numerictypes import bytes_, str_, integer, int_, object_, bool_, character
from .numeric import ndarray, array as narray
from numpy._core.multiarray import _vec_string, compare_chararrays
from numpy._core import overrides
from numpy._utils import asbytes
import numpy
__all__ = ['equal', 'not_equal', 'greater_equal', 'less_equal', 'greater', 'less', 'str_len', 'add', 'multiply', 'mod', 'capitalize', 'center', 'count', 'decode', 'encode', 'endswith', 'expandtabs', 'find', 'index', 'isalnum', 'isalpha', 'isdigit', 'islower', 'isspace', 'istitle', 'isupper', 'join', 'ljust', 'lower', 'lstrip', 'partition', 'replace', 'rfind', 'rindex', 'rjust', 'rpartition', 'rsplit', 'rstrip', 'split', 'splitlines', 'startswith', 'strip', 'swapcase', 'title', 'translate', 'upper', 'zfill', 'isnumeric', 'isdecimal', 'array', 'asarray', 'compare_chararrays', 'chararray']
_globalvar = 0
array_function_dispatch = functools.partial(overrides.array_function_dispatch, module='numpy.char')

def _is_unicode(arr):
    if False:
        return 10
    'Returns True if arr is a string or a string array with a dtype that\n    represents a unicode string, otherwise returns False.\n\n    '
    if isinstance(arr, str) or issubclass(numpy.asarray(arr).dtype.type, str):
        return True
    return False

def _to_bytes_or_str_array(result, output_dtype_like=None):
    if False:
        while True:
            i = 10
    '\n    Helper function to cast a result back into an array\n    with the appropriate dtype if an object array must be used\n    as an intermediary.\n    '
    ret = numpy.asarray(result.tolist())
    dtype = getattr(output_dtype_like, 'dtype', None)
    if dtype is not None:
        return ret.astype(type(dtype)(_get_num_chars(ret)), copy=False)
    return ret

def _clean_args(*args):
    if False:
        i = 10
        return i + 15
    "\n    Helper function for delegating arguments to Python string\n    functions.\n\n    Many of the Python string operations that have optional arguments\n    do not use 'None' to indicate a default value.  In these cases,\n    we need to remove all None arguments, and those following them.\n    "
    newargs = []
    for chk in args:
        if chk is None:
            break
        newargs.append(chk)
    return newargs

def _get_num_chars(a):
    if False:
        i = 10
        return i + 15
    '\n    Helper function that returns the number of characters per field in\n    a string or unicode array.  This is to abstract out the fact that\n    for a unicode array this is itemsize / 4.\n    '
    if issubclass(a.dtype.type, str_):
        return a.itemsize // 4
    return a.itemsize

def _binary_op_dispatcher(x1, x2):
    if False:
        i = 10
        return i + 15
    return (x1, x2)

@array_function_dispatch(_binary_op_dispatcher)
def equal(x1, x2):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return (x1 == x2) element-wise.\n\n    Unlike `numpy.equal`, this comparison is performed by first\n    stripping whitespace characters from the end of the string.  This\n    behavior is provided for backward-compatibility with numarray.\n\n    Parameters\n    ----------\n    x1, x2 : array_like of str or unicode\n        Input arrays of the same shape.\n\n    Returns\n    -------\n    out : ndarray\n        Output array of bools.\n\n    See Also\n    --------\n    not_equal, greater_equal, less_equal, greater, less\n    '
    return compare_chararrays(x1, x2, '==', True)

@array_function_dispatch(_binary_op_dispatcher)
def not_equal(x1, x2):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return (x1 != x2) element-wise.\n\n    Unlike `numpy.not_equal`, this comparison is performed by first\n    stripping whitespace characters from the end of the string.  This\n    behavior is provided for backward-compatibility with numarray.\n\n    Parameters\n    ----------\n    x1, x2 : array_like of str or unicode\n        Input arrays of the same shape.\n\n    Returns\n    -------\n    out : ndarray\n        Output array of bools.\n\n    See Also\n    --------\n    equal, greater_equal, less_equal, greater, less\n    '
    return compare_chararrays(x1, x2, '!=', True)

@array_function_dispatch(_binary_op_dispatcher)
def greater_equal(x1, x2):
    if False:
        i = 10
        return i + 15
    '\n    Return (x1 >= x2) element-wise.\n\n    Unlike `numpy.greater_equal`, this comparison is performed by\n    first stripping whitespace characters from the end of the string.\n    This behavior is provided for backward-compatibility with\n    numarray.\n\n    Parameters\n    ----------\n    x1, x2 : array_like of str or unicode\n        Input arrays of the same shape.\n\n    Returns\n    -------\n    out : ndarray\n        Output array of bools.\n\n    See Also\n    --------\n    equal, not_equal, less_equal, greater, less\n    '
    return compare_chararrays(x1, x2, '>=', True)

@array_function_dispatch(_binary_op_dispatcher)
def less_equal(x1, x2):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return (x1 <= x2) element-wise.\n\n    Unlike `numpy.less_equal`, this comparison is performed by first\n    stripping whitespace characters from the end of the string.  This\n    behavior is provided for backward-compatibility with numarray.\n\n    Parameters\n    ----------\n    x1, x2 : array_like of str or unicode\n        Input arrays of the same shape.\n\n    Returns\n    -------\n    out : ndarray\n        Output array of bools.\n\n    See Also\n    --------\n    equal, not_equal, greater_equal, greater, less\n    '
    return compare_chararrays(x1, x2, '<=', True)

@array_function_dispatch(_binary_op_dispatcher)
def greater(x1, x2):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return (x1 > x2) element-wise.\n\n    Unlike `numpy.greater`, this comparison is performed by first\n    stripping whitespace characters from the end of the string.  This\n    behavior is provided for backward-compatibility with numarray.\n\n    Parameters\n    ----------\n    x1, x2 : array_like of str or unicode\n        Input arrays of the same shape.\n\n    Returns\n    -------\n    out : ndarray\n        Output array of bools.\n\n    See Also\n    --------\n    equal, not_equal, greater_equal, less_equal, less\n    '
    return compare_chararrays(x1, x2, '>', True)

@array_function_dispatch(_binary_op_dispatcher)
def less(x1, x2):
    if False:
        while True:
            i = 10
    '\n    Return (x1 < x2) element-wise.\n\n    Unlike `numpy.greater`, this comparison is performed by first\n    stripping whitespace characters from the end of the string.  This\n    behavior is provided for backward-compatibility with numarray.\n\n    Parameters\n    ----------\n    x1, x2 : array_like of str or unicode\n        Input arrays of the same shape.\n\n    Returns\n    -------\n    out : ndarray\n        Output array of bools.\n\n    See Also\n    --------\n    equal, not_equal, greater_equal, less_equal, greater\n    '
    return compare_chararrays(x1, x2, '<', True)

def _unary_op_dispatcher(a):
    if False:
        while True:
            i = 10
    return (a,)

@array_function_dispatch(_unary_op_dispatcher)
def str_len(a):
    if False:
        for i in range(10):
            print('nop')
    "\n    Return len(a) element-wise.\n\n    Parameters\n    ----------\n    a : array_like of str or unicode\n\n    Returns\n    -------\n    out : ndarray\n        Output array of integers\n\n    See Also\n    --------\n    len\n\n    Examples\n    --------\n    >>> a = np.array(['Grace Hopper Conference', 'Open Source Day'])\n    >>> np.char.str_len(a)\n    array([23, 15])\n    >>> a = np.array([u'Р', u'о'])\n    >>> np.char.str_len(a)\n    array([1, 1])\n    >>> a = np.array([['hello', 'world'], [u'Р', u'о']])\n    >>> np.char.str_len(a)\n    array([[5, 5], [1, 1]])\n    "
    return numpy._core.umath.str_len(a)

@array_function_dispatch(_binary_op_dispatcher)
def add(x1, x2):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return element-wise string concatenation for two arrays of str or unicode.\n\n    Arrays `x1` and `x2` must have the same shape.\n\n    Parameters\n    ----------\n    x1 : array_like of str or unicode\n        Input array.\n    x2 : array_like of str or unicode\n        Input array.\n\n    Returns\n    -------\n    add : ndarray\n        Output array of `bytes_` or `str_`, depending on input types\n        of the same shape as `x1` and `x2`.\n\n    '
    arr1 = numpy.asarray(x1)
    arr2 = numpy.asarray(x2)
    if type(arr1.dtype) != type(arr2.dtype):
        raise TypeError(f"np.char.add() requires both arrays of the same dtype kind, but got dtypes: '{arr1.dtype}' and '{arr2.dtype}' (the few cases where this used to work often lead to incorrect results).")
    return numpy.add(x1, x2)

def _multiply_dispatcher(a, i):
    if False:
        return 10
    return (a,)

@array_function_dispatch(_multiply_dispatcher)
def multiply(a, i):
    if False:
        i = 10
        return i + 15
    '\n    Return (a * i), that is string multiple concatenation,\n    element-wise.\n\n    Values in `i` of less than 0 are treated as 0 (which yields an\n    empty string).\n\n    Parameters\n    ----------\n    a : array_like of str or unicode\n\n    i : array_like of ints\n\n    Returns\n    -------\n    out : ndarray\n        Output array of str or unicode, depending on input types\n    \n    Examples\n    --------\n    >>> a = np.array(["a", "b", "c"])\n    >>> np.char.multiply(x, 3)\n    array([\'aaa\', \'bbb\', \'ccc\'], dtype=\'<U3\')\n    >>> i = np.array([1, 2, 3])\n    >>> np.char.multiply(a, i)\n    array([\'a\', \'bb\', \'ccc\'], dtype=\'<U3\')\n    >>> np.char.multiply(np.array([\'a\']), i)\n    array([\'a\', \'aa\', \'aaa\'], dtype=\'<U3\')\n    >>> a = np.array([\'a\', \'b\', \'c\', \'d\', \'e\', \'f\']).reshape((2, 3))\n    >>> np.char.multiply(a, 3)\n    array([[\'aaa\', \'bbb\', \'ccc\'],\n           [\'ddd\', \'eee\', \'fff\']], dtype=\'<U3\')\n    >>> np.char.multiply(a, i)\n    array([[\'a\', \'bb\', \'ccc\'],\n           [\'d\', \'ee\', \'fff\']], dtype=\'<U3\')\n    '
    a_arr = numpy.asarray(a)
    i_arr = numpy.asarray(i)
    if not issubclass(i_arr.dtype.type, integer):
        raise ValueError('Can only multiply by integers')
    out_size = _get_num_chars(a_arr) * max(int(i_arr.max()), 0)
    return _vec_string(a_arr, type(a_arr.dtype)(out_size), '__mul__', (i_arr,))

def _mod_dispatcher(a, values):
    if False:
        while True:
            i = 10
    return (a, values)

@array_function_dispatch(_mod_dispatcher)
def mod(a, values):
    if False:
        return 10
    '\n    Return (a % i), that is pre-Python 2.6 string formatting\n    (interpolation), element-wise for a pair of array_likes of str\n    or unicode.\n\n    Parameters\n    ----------\n    a : array_like of str or unicode\n\n    values : array_like of values\n       These values will be element-wise interpolated into the string.\n\n    Returns\n    -------\n    out : ndarray\n        Output array of str or unicode, depending on input types\n\n\n    '
    return _to_bytes_or_str_array(_vec_string(a, object_, '__mod__', (values,)), a)

@array_function_dispatch(_unary_op_dispatcher)
def capitalize(a):
    if False:
        for i in range(10):
            print('nop')
    "\n    Return a copy of `a` with only the first character of each element\n    capitalized.\n\n    Calls :meth:`str.capitalize` element-wise.\n\n    For 8-bit strings, this method is locale-dependent.\n\n    Parameters\n    ----------\n    a : array_like of str or unicode\n        Input array of strings to capitalize.\n\n    Returns\n    -------\n    out : ndarray\n        Output array of str or unicode, depending on input\n        types\n\n    See Also\n    --------\n    str.capitalize\n\n    Examples\n    --------\n    >>> c = np.array(['a1b2','1b2a','b2a1','2a1b'],'S4'); c\n    array(['a1b2', '1b2a', 'b2a1', '2a1b'],\n        dtype='|S4')\n    >>> np.char.capitalize(c)\n    array(['A1b2', '1b2a', 'B2a1', '2a1b'],\n        dtype='|S4')\n\n    "
    a_arr = numpy.asarray(a)
    return _vec_string(a_arr, a_arr.dtype, 'capitalize')

def _center_dispatcher(a, width, fillchar=None):
    if False:
        for i in range(10):
            print('nop')
    return (a,)

@array_function_dispatch(_center_dispatcher)
def center(a, width, fillchar=' '):
    if False:
        return 10
    "\n    Return a copy of `a` with its elements centered in a string of\n    length `width`.\n\n    Calls :meth:`str.center` element-wise.\n\n    Parameters\n    ----------\n    a : array_like of str or unicode\n\n    width : int\n        The length of the resulting strings\n    fillchar : str or unicode, optional\n        The padding character to use (default is space).\n\n    Returns\n    -------\n    out : ndarray\n        Output array of str or unicode, depending on input\n        types\n\n    See Also\n    --------\n    str.center\n    \n    Notes\n    -----\n    This function is intended to work with arrays of strings.  The\n    fill character is not applied to numeric types.\n\n    Examples\n    --------\n    >>> c = np.array(['a1b2','1b2a','b2a1','2a1b']); c\n    array(['a1b2', '1b2a', 'b2a1', '2a1b'], dtype='<U4')\n    >>> np.char.center(c, width=9)\n    array(['   a1b2  ', '   1b2a  ', '   b2a1  ', '   2a1b  '], dtype='<U9')\n    >>> np.char.center(c, width=9, fillchar='*')\n    array(['***a1b2**', '***1b2a**', '***b2a1**', '***2a1b**'], dtype='<U9')\n    >>> np.char.center(c, width=1)\n    array(['a', '1', 'b', '2'], dtype='<U1')\n\n    "
    a_arr = numpy.asarray(a)
    width_arr = numpy.asarray(width)
    size = int(numpy.max(width_arr.flat))
    if numpy.issubdtype(a_arr.dtype, numpy.bytes_):
        fillchar = asbytes(fillchar)
    return _vec_string(a_arr, type(a_arr.dtype)(size), 'center', (width_arr, fillchar))

def _count_dispatcher(a, sub, start=None, end=None):
    if False:
        print('Hello World!')
    return (a,)

@array_function_dispatch(_count_dispatcher)
def count(a, sub, start=0, end=None):
    if False:
        i = 10
        return i + 15
    "\n    Returns an array with the number of non-overlapping occurrences of\n    substring `sub` in the range [`start`, `end`].\n\n    Calls :meth:`str.count` element-wise.\n\n    Parameters\n    ----------\n    a : array_like of str or unicode\n\n    sub : str or unicode\n       The substring to search for.\n\n    start, end : int, optional\n       Optional arguments `start` and `end` are interpreted as slice\n       notation to specify the range in which to count.\n\n    Returns\n    -------\n    out : ndarray\n        Output array of ints.\n\n    See Also\n    --------\n    str.count\n\n    Examples\n    --------\n    >>> c = np.array(['aAaAaA', '  aA  ', 'abBABba'])\n    >>> c\n    array(['aAaAaA', '  aA  ', 'abBABba'], dtype='<U7')\n    >>> np.char.count(c, 'A')\n    array([3, 1, 1])\n    >>> np.char.count(c, 'aA')\n    array([3, 1, 0])\n    >>> np.char.count(c, 'A', start=1, end=4)\n    array([2, 1, 1])\n    >>> np.char.count(c, 'A', start=1, end=3)\n    array([1, 0, 0])\n\n    "
    end = end if end is not None else numpy.iinfo(numpy.int64).max
    return numpy._core.umath.count(a, sub, start, end)

def _code_dispatcher(a, encoding=None, errors=None):
    if False:
        for i in range(10):
            print('nop')
    return (a,)

@array_function_dispatch(_code_dispatcher)
def decode(a, encoding=None, errors=None):
    if False:
        i = 10
        return i + 15
    "\n    Calls :meth:`bytes.decode` element-wise.\n\n    The set of available codecs comes from the Python standard library,\n    and may be extended at runtime.  For more information, see the\n    :mod:`codecs` module.\n\n    Parameters\n    ----------\n    a : array_like of str or unicode\n\n    encoding : str, optional\n       The name of an encoding\n\n    errors : str, optional\n       Specifies how to handle encoding errors\n\n    Returns\n    -------\n    out : ndarray\n\n    See Also\n    --------\n    :py:meth:`bytes.decode`\n\n    Notes\n    -----\n    The type of the result will depend on the encoding specified.\n\n    Examples\n    --------\n    >>> c = np.array([b'\\x81\\xc1\\x81\\xc1\\x81\\xc1', b'@@\\x81\\xc1@@',\n    ...               b'\\x81\\x82\\xc2\\xc1\\xc2\\x82\\x81'])\n    >>> c\n    array([b'\\x81\\xc1\\x81\\xc1\\x81\\xc1', b'@@\\x81\\xc1@@',\n    ...    b'\\x81\\x82\\xc2\\xc1\\xc2\\x82\\x81'], dtype='|S7')\n    >>> np.char.decode(c, encoding='cp037')\n    array(['aAaAaA', '  aA  ', 'abBABba'], dtype='<U7')\n\n    "
    return _to_bytes_or_str_array(_vec_string(a, object_, 'decode', _clean_args(encoding, errors)))

@array_function_dispatch(_code_dispatcher)
def encode(a, encoding=None, errors=None):
    if False:
        while True:
            i = 10
    '\n    Calls :meth:`str.encode` element-wise.\n\n    The set of available codecs comes from the Python standard library,\n    and may be extended at runtime. For more information, see the\n    :mod:`codecs` module.\n\n    Parameters\n    ----------\n    a : array_like of str or unicode\n\n    encoding : str, optional\n       The name of an encoding\n\n    errors : str, optional\n       Specifies how to handle encoding errors\n\n    Returns\n    -------\n    out : ndarray\n\n    See Also\n    --------\n    str.encode\n\n    Notes\n    -----\n    The type of the result will depend on the encoding specified.\n\n    '
    return _to_bytes_or_str_array(_vec_string(a, object_, 'encode', _clean_args(encoding, errors)))

def _endswith_dispatcher(a, suffix, start=None, end=None):
    if False:
        i = 10
        return i + 15
    return (a,)

@array_function_dispatch(_endswith_dispatcher)
def endswith(a, suffix, start=0, end=None):
    if False:
        i = 10
        return i + 15
    "\n    Returns a boolean array which is `True` where the string element\n    in `a` ends with `suffix`, otherwise `False`.\n\n    Calls :meth:`str.endswith` element-wise.\n\n    Parameters\n    ----------\n    a : array_like of str or unicode\n\n    suffix : str\n\n    start, end : int, optional\n        With optional `start`, test beginning at that position. With\n        optional `end`, stop comparing at that position.\n\n    Returns\n    -------\n    out : ndarray\n        Outputs an array of bools.\n\n    See Also\n    --------\n    str.endswith\n\n    Examples\n    --------\n    >>> s = np.array(['foo', 'bar'])\n    >>> s[0] = 'foo'\n    >>> s[1] = 'bar'\n    >>> s\n    array(['foo', 'bar'], dtype='<U3')\n    >>> np.char.endswith(s, 'ar')\n    array([False,  True])\n    >>> np.char.endswith(s, 'a', start=1, end=2)\n    array([False,  True])\n\n    "
    end = end if end is not None else numpy.iinfo(numpy.int_).max
    return numpy._core.umath.endswith(a, suffix, start, end)

def _expandtabs_dispatcher(a, tabsize=None):
    if False:
        i = 10
        return i + 15
    return (a,)

@array_function_dispatch(_expandtabs_dispatcher)
def expandtabs(a, tabsize=8):
    if False:
        print('Hello World!')
    "\n    Return a copy of each string element where all tab characters are\n    replaced by one or more spaces.\n\n    Calls :meth:`str.expandtabs` element-wise.\n\n    Return a copy of each string element where all tab characters are\n    replaced by one or more spaces, depending on the current column\n    and the given `tabsize`. The column number is reset to zero after\n    each newline occurring in the string. This doesn't understand other\n    non-printing characters or escape sequences.\n\n    Parameters\n    ----------\n    a : array_like of str or unicode\n        Input array\n    tabsize : int, optional\n        Replace tabs with `tabsize` number of spaces.  If not given defaults\n        to 8 spaces.\n\n    Returns\n    -------\n    out : ndarray\n        Output array of str or unicode, depending on input type\n\n    See Also\n    --------\n    str.expandtabs\n\n    "
    return _to_bytes_or_str_array(_vec_string(a, object_, 'expandtabs', (tabsize,)), a)

@array_function_dispatch(_count_dispatcher)
def find(a, sub, start=0, end=None):
    if False:
        while True:
            i = 10
    '\n    For each element, return the lowest index in the string where\n    substring `sub` is found.\n\n    Calls :meth:`str.find` element-wise.\n\n    For each element, return the lowest index in the string where\n    substring `sub` is found, such that `sub` is contained in the\n    range [`start`, `end`].\n\n    Parameters\n    ----------\n    a : array_like of str or unicode\n\n    sub : str or unicode\n\n    start, end : int, optional\n        Optional arguments `start` and `end` are interpreted as in\n        slice notation.\n\n    Returns\n    -------\n    out : ndarray or int\n        Output array of ints.  Returns -1 if `sub` is not found.\n\n    See Also\n    --------\n    str.find\n\n    Examples\n    --------\n    >>> a = np.array(["NumPy is a Python library"])\n    >>> np.char.find(a, "Python", start=0, end=None)\n    array([11])\n\n    '
    end = end if end is not None else numpy.iinfo(numpy.int64).max
    return numpy._core.umath.find(a, sub, start, end)

@array_function_dispatch(_count_dispatcher)
def index(a, sub, start=0, end=None):
    if False:
        while True:
            i = 10
    '\n    Like `find`, but raises :exc:`ValueError` when the substring is not found.\n\n    Calls :meth:`str.index` element-wise.\n\n    Parameters\n    ----------\n    a : array_like of str or unicode\n\n    sub : str or unicode\n\n    start, end : int, optional\n\n    Returns\n    -------\n    out : ndarray\n        Output array of ints.  Returns -1 if `sub` is not found.\n\n    See Also\n    --------\n    find, str.find\n\n    Examples\n    --------\n    >>> a = np.array(["Computer Science"])\n    >>> np.char.index(a, "Science", start=0, end=None)\n    array([9])\n\n    '
    return _vec_string(a, int_, 'index', [sub, start] + _clean_args(end))

@array_function_dispatch(_unary_op_dispatcher)
def isalnum(a):
    if False:
        i = 10
        return i + 15
    '\n    Returns true for each element if all characters in the string are\n    alphanumeric and there is at least one character, false otherwise.\n\n    Calls :meth:`str.isalnum` element-wise.\n\n    For 8-bit strings, this method is locale-dependent.\n\n    Parameters\n    ----------\n    a : array_like of str or unicode\n\n    Returns\n    -------\n    out : ndarray\n        Output array of str or unicode, depending on input type\n\n    See Also\n    --------\n    str.isalnum\n    '
    return _vec_string(a, bool_, 'isalnum')

@array_function_dispatch(_unary_op_dispatcher)
def isalpha(a):
    if False:
        print('Hello World!')
    '\n    Returns true for each element if all characters in the string are\n    alphabetic and there is at least one character, false otherwise.\n\n    Calls :meth:`str.isalpha` element-wise.\n\n    For 8-bit strings, this method is locale-dependent.\n\n    Parameters\n    ----------\n    a : array_like of str or unicode\n\n    Returns\n    -------\n    out : ndarray\n        Output array of bools\n\n    See Also\n    --------\n    str.isalpha\n    '
    return numpy._core.umath.isalpha(a)

@array_function_dispatch(_unary_op_dispatcher)
def isdigit(a):
    if False:
        i = 10
        return i + 15
    "\n    Returns true for each element if all characters in the string are\n    digits and there is at least one character, false otherwise.\n\n    Calls :meth:`str.isdigit` element-wise.\n\n    For 8-bit strings, this method is locale-dependent.\n\n    Parameters\n    ----------\n    a : array_like of str or unicode\n\n    Returns\n    -------\n    out : ndarray\n        Output array of bools\n\n    See Also\n    --------\n    str.isdigit\n\n    Examples\n    --------\n    >>> a = np.array(['a', 'b', '0'])\n    >>> np.char.isdigit(a)\n    array([False, False,  True])\n    >>> a = np.array([['a', 'b', '0'], ['c', '1', '2']])\n    >>> np.char.isdigit(a)\n    array([[False, False,  True], [False,  True,  True]])\n    "
    return numpy._core.umath.isdigit(a)

@array_function_dispatch(_unary_op_dispatcher)
def islower(a):
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns true for each element if all cased characters in the\n    string are lowercase and there is at least one cased character,\n    false otherwise.\n\n    Calls :meth:`str.islower` element-wise.\n\n    For 8-bit strings, this method is locale-dependent.\n\n    Parameters\n    ----------\n    a : array_like of str or unicode\n\n    Returns\n    -------\n    out : ndarray\n        Output array of bools\n\n    See Also\n    --------\n    str.islower\n    '
    return _vec_string(a, bool_, 'islower')

@array_function_dispatch(_unary_op_dispatcher)
def isspace(a):
    if False:
        return 10
    '\n    Returns true for each element if there are only whitespace\n    characters in the string and there is at least one character,\n    false otherwise.\n\n    Calls :meth:`str.isspace` element-wise.\n\n    For 8-bit strings, this method is locale-dependent.\n\n    Parameters\n    ----------\n    a : array_like of str or unicode\n\n    Returns\n    -------\n    out : ndarray\n        Output array of bools\n\n    See Also\n    --------\n    str.isspace\n    '
    return numpy._core.umath.isspace(a)

@array_function_dispatch(_unary_op_dispatcher)
def istitle(a):
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns true for each element if the element is a titlecased\n    string and there is at least one character, false otherwise.\n\n    Call :meth:`str.istitle` element-wise.\n\n    For 8-bit strings, this method is locale-dependent.\n\n    Parameters\n    ----------\n    a : array_like of str or unicode\n\n    Returns\n    -------\n    out : ndarray\n        Output array of bools\n\n    See Also\n    --------\n    str.istitle\n    '
    return _vec_string(a, bool_, 'istitle')

@array_function_dispatch(_unary_op_dispatcher)
def isupper(a):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return true for each element if all cased characters in the\n    string are uppercase and there is at least one character, false\n    otherwise.\n\n    Call :meth:`str.isupper` element-wise.\n\n    For 8-bit strings, this method is locale-dependent.\n\n    Parameters\n    ----------\n    a : array_like of str or unicode\n\n    Returns\n    -------\n    out : ndarray\n        Output array of bools\n\n    See Also\n    --------\n    str.isupper\n\n    Examples\n    --------\n    >>> str = "GHC"\n    >>> np.char.isupper(str)\n    array(True)     \n    >>> a = np.array(["hello", "HELLO", "Hello"])\n    >>> np.char.isupper(a)\n    array([False,  True, False]) \n\n    '
    return _vec_string(a, bool_, 'isupper')

def _join_dispatcher(sep, seq):
    if False:
        print('Hello World!')
    return (sep, seq)

@array_function_dispatch(_join_dispatcher)
def join(sep, seq):
    if False:
        for i in range(10):
            print('nop')
    "\n    Return a string which is the concatenation of the strings in the\n    sequence `seq`.\n\n    Calls :meth:`str.join` element-wise.\n\n    Parameters\n    ----------\n    sep : array_like of str or unicode\n    seq : array_like of str or unicode\n\n    Returns\n    -------\n    out : ndarray\n        Output array of str or unicode, depending on input types\n\n    See Also\n    --------\n    str.join\n\n    Examples\n    --------\n    >>> np.char.join('-', 'osd')\n    array('o-s-d', dtype='<U5')\n\n    >>> np.char.join(['-', '.'], ['ghc', 'osd'])\n    array(['g-h-c', 'o.s.d'], dtype='<U5')\n\n    "
    return _to_bytes_or_str_array(_vec_string(sep, object_, 'join', (seq,)), seq)

def _just_dispatcher(a, width, fillchar=None):
    if False:
        print('Hello World!')
    return (a,)

@array_function_dispatch(_just_dispatcher)
def ljust(a, width, fillchar=' '):
    if False:
        while True:
            i = 10
    '\n    Return an array with the elements of `a` left-justified in a\n    string of length `width`.\n\n    Calls :meth:`str.ljust` element-wise.\n\n    Parameters\n    ----------\n    a : array_like of str or unicode\n\n    width : int\n        The length of the resulting strings\n    fillchar : str or unicode, optional\n        The character to use for padding\n\n    Returns\n    -------\n    out : ndarray\n        Output array of str or unicode, depending on input type\n\n    See Also\n    --------\n    str.ljust\n\n    '
    a_arr = numpy.asarray(a)
    width_arr = numpy.asarray(width)
    size = int(numpy.max(width_arr.flat))
    if numpy.issubdtype(a_arr.dtype, numpy.bytes_):
        fillchar = asbytes(fillchar)
    return _vec_string(a_arr, type(a_arr.dtype)(size), 'ljust', (width_arr, fillchar))

@array_function_dispatch(_unary_op_dispatcher)
def lower(a):
    if False:
        while True:
            i = 10
    "\n    Return an array with the elements converted to lowercase.\n\n    Call :meth:`str.lower` element-wise.\n\n    For 8-bit strings, this method is locale-dependent.\n\n    Parameters\n    ----------\n    a : array_like, {str, unicode}\n        Input array.\n\n    Returns\n    -------\n    out : ndarray, {str, unicode}\n        Output array of str or unicode, depending on input type\n\n    See Also\n    --------\n    str.lower\n\n    Examples\n    --------\n    >>> c = np.array(['A1B C', '1BCA', 'BCA1']); c\n    array(['A1B C', '1BCA', 'BCA1'], dtype='<U5')\n    >>> np.char.lower(c)\n    array(['a1b c', '1bca', 'bca1'], dtype='<U5')\n\n    "
    a_arr = numpy.asarray(a)
    return _vec_string(a_arr, a_arr.dtype, 'lower')

def _strip_dispatcher(a, chars=None):
    if False:
        i = 10
        return i + 15
    return (a,)

@array_function_dispatch(_strip_dispatcher)
def lstrip(a, chars=None):
    if False:
        return 10
    "\n    For each element in `a`, return a copy with the leading characters\n    removed.\n\n    Calls :meth:`str.lstrip` element-wise.\n\n    Parameters\n    ----------\n    a : array-like, {str, unicode}\n        Input array.\n\n    chars : {str, unicode}, optional\n        The `chars` argument is a string specifying the set of\n        characters to be removed. If omitted or None, the `chars`\n        argument defaults to removing whitespace. The `chars` argument\n        is not a prefix; rather, all combinations of its values are\n        stripped.\n\n    Returns\n    -------\n    out : ndarray, {str, unicode}\n        Output array of str or unicode, depending on input type\n\n    See Also\n    --------\n    str.lstrip\n\n    Examples\n    --------\n    >>> c = np.array(['aAaAaA', '  aA  ', 'abBABba'])\n    >>> c\n    array(['aAaAaA', '  aA  ', 'abBABba'], dtype='<U7')\n\n    The 'a' variable is unstripped from c[1] because whitespace leading.\n\n    >>> np.char.lstrip(c, 'a')\n    array(['AaAaA', '  aA  ', 'bBABba'], dtype='<U7')\n\n\n    >>> np.char.lstrip(c, 'A') # leaves c unchanged\n    array(['aAaAaA', '  aA  ', 'abBABba'], dtype='<U7')\n    >>> (np.char.lstrip(c, ' ') == np.char.lstrip(c, '')).all()\n    ... # XXX: is this a regression? This used to return True\n    ... # np.char.lstrip(c,'') does not modify c at all.\n    False\n    >>> (np.char.lstrip(c, ' ') == np.char.lstrip(c, None)).all()\n    True\n\n    "
    a_arr = numpy.asarray(a)
    return _vec_string(a_arr, a_arr.dtype, 'lstrip', (chars,))

def _partition_dispatcher(a, sep):
    if False:
        return 10
    return (a,)

@array_function_dispatch(_partition_dispatcher)
def partition(a, sep):
    if False:
        while True:
            i = 10
    '\n    Partition each element in `a` around `sep`.\n\n    Calls :meth:`str.partition` element-wise.\n\n    For each element in `a`, split the element as the first\n    occurrence of `sep`, and return 3 strings containing the part\n    before the separator, the separator itself, and the part after\n    the separator. If the separator is not found, return 3 strings\n    containing the string itself, followed by two empty strings.\n\n    Parameters\n    ----------\n    a : array_like, {str, unicode}\n        Input array\n    sep : {str, unicode}\n        Separator to split each string element in `a`.\n\n    Returns\n    -------\n    out : ndarray, {str, unicode}\n        Output array of str or unicode, depending on input type.\n        The output array will have an extra dimension with 3\n        elements per input element.\n\n    See Also\n    --------\n    str.partition\n\n    '
    return _to_bytes_or_str_array(_vec_string(a, object_, 'partition', (sep,)), a)

def _replace_dispatcher(a, old, new, count=None):
    if False:
        i = 10
        return i + 15
    return (a,)

@array_function_dispatch(_replace_dispatcher)
def replace(a, old, new, count=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    For each element in `a`, return a copy of the string with all\n    occurrences of substring `old` replaced by `new`.\n\n    Calls :meth:`str.replace` element-wise.\n\n    Parameters\n    ----------\n    a : array-like of str or unicode\n\n    old, new : str or unicode\n\n    count : int, optional\n        If the optional argument `count` is given, only the first\n        `count` occurrences are replaced.\n\n    Returns\n    -------\n    out : ndarray\n        Output array of str or unicode, depending on input type\n\n    See Also\n    --------\n    str.replace\n    \n    Examples\n    --------\n    >>> a = np.array(["That is a mango", "Monkeys eat mangos"])\n    >>> np.char.replace(a, \'mango\', \'banana\')\n    array([\'That is a banana\', \'Monkeys eat bananas\'], dtype=\'<U19\')\n\n    >>> a = np.array(["The dish is fresh", "This is it"])\n    >>> np.char.replace(a, \'is\', \'was\')\n    array([\'The dwash was fresh\', \'Thwas was it\'], dtype=\'<U19\')\n    '
    return _to_bytes_or_str_array(_vec_string(a, object_, 'replace', [old, new] + _clean_args(count)), a)

@array_function_dispatch(_count_dispatcher)
def rfind(a, sub, start=0, end=None):
    if False:
        print('Hello World!')
    '\n    For each element in `a`, return the highest index in the string\n    where substring `sub` is found, such that `sub` is contained\n    within [`start`, `end`].\n\n    Calls :meth:`str.rfind` element-wise.\n\n    Parameters\n    ----------\n    a : array-like of str or unicode\n\n    sub : str or unicode\n\n    start, end : int, optional\n        Optional arguments `start` and `end` are interpreted as in\n        slice notation.\n\n    Returns\n    -------\n    out : ndarray\n       Output array of ints.  Return -1 on failure.\n\n    See Also\n    --------\n    str.rfind\n\n    '
    end = end if end is not None else numpy.iinfo(numpy.int64).max
    return numpy._core.umath.rfind(a, sub, start, end)

@array_function_dispatch(_count_dispatcher)
def rindex(a, sub, start=0, end=None):
    if False:
        print('Hello World!')
    '\n    Like `rfind`, but raises :exc:`ValueError` when the substring `sub` is\n    not found.\n\n    Calls :meth:`str.rindex` element-wise.\n\n    Parameters\n    ----------\n    a : array-like of str or unicode\n\n    sub : str or unicode\n\n    start, end : int, optional\n\n    Returns\n    -------\n    out : ndarray\n       Output array of ints.\n\n    See Also\n    --------\n    rfind, str.rindex\n\n    '
    return _vec_string(a, int_, 'rindex', [sub, start] + _clean_args(end))

@array_function_dispatch(_just_dispatcher)
def rjust(a, width, fillchar=' '):
    if False:
        print('Hello World!')
    '\n    Return an array with the elements of `a` right-justified in a\n    string of length `width`.\n\n    Calls :meth:`str.rjust` element-wise.\n\n    Parameters\n    ----------\n    a : array_like of str or unicode\n\n    width : int\n        The length of the resulting strings\n    fillchar : str or unicode, optional\n        The character to use for padding\n\n    Returns\n    -------\n    out : ndarray\n        Output array of str or unicode, depending on input type\n\n    See Also\n    --------\n    str.rjust\n\n    '
    a_arr = numpy.asarray(a)
    width_arr = numpy.asarray(width)
    size = int(numpy.max(width_arr.flat))
    if numpy.issubdtype(a_arr.dtype, numpy.bytes_):
        fillchar = asbytes(fillchar)
    return _vec_string(a_arr, type(a_arr.dtype)(size), 'rjust', (width_arr, fillchar))

@array_function_dispatch(_partition_dispatcher)
def rpartition(a, sep):
    if False:
        print('Hello World!')
    '\n    Partition (split) each element around the right-most separator.\n\n    Calls :meth:`str.rpartition` element-wise.\n\n    For each element in `a`, split the element as the last\n    occurrence of `sep`, and return 3 strings containing the part\n    before the separator, the separator itself, and the part after\n    the separator. If the separator is not found, return 3 strings\n    containing the string itself, followed by two empty strings.\n\n    Parameters\n    ----------\n    a : array_like of str or unicode\n        Input array\n    sep : str or unicode\n        Right-most separator to split each element in array.\n\n    Returns\n    -------\n    out : ndarray\n        Output array of string or unicode, depending on input\n        type.  The output array will have an extra dimension with\n        3 elements per input element.\n\n    See Also\n    --------\n    str.rpartition\n\n    '
    return _to_bytes_or_str_array(_vec_string(a, object_, 'rpartition', (sep,)), a)

def _split_dispatcher(a, sep=None, maxsplit=None):
    if False:
        return 10
    return (a,)

@array_function_dispatch(_split_dispatcher)
def rsplit(a, sep=None, maxsplit=None):
    if False:
        return 10
    '\n    For each element in `a`, return a list of the words in the\n    string, using `sep` as the delimiter string.\n\n    Calls :meth:`str.rsplit` element-wise.\n\n    Except for splitting from the right, `rsplit`\n    behaves like `split`.\n\n    Parameters\n    ----------\n    a : array_like of str or unicode\n\n    sep : str or unicode, optional\n        If `sep` is not specified or None, any whitespace string\n        is a separator.\n    maxsplit : int, optional\n        If `maxsplit` is given, at most `maxsplit` splits are done,\n        the rightmost ones.\n\n    Returns\n    -------\n    out : ndarray\n       Array of list objects\n\n    See Also\n    --------\n    str.rsplit, split\n\n    '
    return _vec_string(a, object_, 'rsplit', [sep] + _clean_args(maxsplit))

def _strip_dispatcher(a, chars=None):
    if False:
        print('Hello World!')
    return (a,)

@array_function_dispatch(_strip_dispatcher)
def rstrip(a, chars=None):
    if False:
        return 10
    "\n    For each element in `a`, return a copy with the trailing\n    characters removed.\n\n    Calls :meth:`str.rstrip` element-wise.\n\n    Parameters\n    ----------\n    a : array-like of str or unicode\n\n    chars : str or unicode, optional\n       The `chars` argument is a string specifying the set of\n       characters to be removed. If omitted or None, the `chars`\n       argument defaults to removing whitespace. The `chars` argument\n       is not a suffix; rather, all combinations of its values are\n       stripped.\n\n    Returns\n    -------\n    out : ndarray\n        Output array of str or unicode, depending on input type\n\n    See Also\n    --------\n    str.rstrip\n\n    Examples\n    --------\n    >>> c = np.array(['aAaAaA', 'abBABba'], dtype='S7'); c\n    array(['aAaAaA', 'abBABba'],\n        dtype='|S7')\n    >>> np.char.rstrip(c, b'a')\n    array(['aAaAaA', 'abBABb'],\n        dtype='|S7')\n    >>> np.char.rstrip(c, b'A')\n    array(['aAaAa', 'abBABba'],\n        dtype='|S7')\n\n    "
    a_arr = numpy.asarray(a)
    return _vec_string(a_arr, a_arr.dtype, 'rstrip', (chars,))

@array_function_dispatch(_split_dispatcher)
def split(a, sep=None, maxsplit=None):
    if False:
        i = 10
        return i + 15
    '\n    For each element in `a`, return a list of the words in the\n    string, using `sep` as the delimiter string.\n\n    Calls :meth:`str.split` element-wise.\n\n    Parameters\n    ----------\n    a : array_like of str or unicode\n\n    sep : str or unicode, optional\n       If `sep` is not specified or None, any whitespace string is a\n       separator.\n\n    maxsplit : int, optional\n        If `maxsplit` is given, at most `maxsplit` splits are done.\n\n    Returns\n    -------\n    out : ndarray\n        Array of list objects\n\n    See Also\n    --------\n    str.split, rsplit\n\n    '
    return _vec_string(a, object_, 'split', [sep] + _clean_args(maxsplit))

def _splitlines_dispatcher(a, keepends=None):
    if False:
        while True:
            i = 10
    return (a,)

@array_function_dispatch(_splitlines_dispatcher)
def splitlines(a, keepends=None):
    if False:
        while True:
            i = 10
    '\n    For each element in `a`, return a list of the lines in the\n    element, breaking at line boundaries.\n\n    Calls :meth:`str.splitlines` element-wise.\n\n    Parameters\n    ----------\n    a : array_like of str or unicode\n\n    keepends : bool, optional\n        Line breaks are not included in the resulting list unless\n        keepends is given and true.\n\n    Returns\n    -------\n    out : ndarray\n        Array of list objects\n\n    See Also\n    --------\n    str.splitlines\n\n    '
    return _vec_string(a, object_, 'splitlines', _clean_args(keepends))

def _startswith_dispatcher(a, prefix, start=None, end=None):
    if False:
        while True:
            i = 10
    return (a,)

@array_function_dispatch(_startswith_dispatcher)
def startswith(a, prefix, start=0, end=None):
    if False:
        i = 10
        return i + 15
    '\n    Returns a boolean array which is `True` where the string element\n    in `a` starts with `prefix`, otherwise `False`.\n\n    Calls :meth:`str.startswith` element-wise.\n\n    Parameters\n    ----------\n    a : array_like of str or unicode\n\n    prefix : str\n\n    start, end : int, optional\n        With optional `start`, test beginning at that position. With\n        optional `end`, stop comparing at that position.\n\n    Returns\n    -------\n    out : ndarray\n        Array of booleans\n\n    See Also\n    --------\n    str.startswith\n\n    '
    end = end if end is not None else numpy.iinfo(numpy.int_).max
    return numpy._core.umath.startswith(a, prefix, start, end)

@array_function_dispatch(_strip_dispatcher)
def strip(a, chars=None):
    if False:
        return 10
    "\n    For each element in `a`, return a copy with the leading and\n    trailing characters removed.\n\n    Calls :meth:`str.strip` element-wise.\n\n    Parameters\n    ----------\n    a : array-like of str or unicode\n\n    chars : str or unicode, optional\n       The `chars` argument is a string specifying the set of\n       characters to be removed. If omitted or None, the `chars`\n       argument defaults to removing whitespace. The `chars` argument\n       is not a prefix or suffix; rather, all combinations of its\n       values are stripped.\n\n    Returns\n    -------\n    out : ndarray\n        Output array of str or unicode, depending on input type\n\n    See Also\n    --------\n    str.strip\n\n    Examples\n    --------\n    >>> c = np.array(['aAaAaA', '  aA  ', 'abBABba'])\n    >>> c\n    array(['aAaAaA', '  aA  ', 'abBABba'], dtype='<U7')\n    >>> np.char.strip(c)\n    array(['aAaAaA', 'aA', 'abBABba'], dtype='<U7')\n    >>> np.char.strip(c, 'a') # 'a' unstripped from c[1] because ws leads\n    array(['AaAaA', '  aA  ', 'bBABb'], dtype='<U7')\n    >>> np.char.strip(c, 'A') # 'A' unstripped from c[1] because ws trails\n    array(['aAaAa', '  aA  ', 'abBABba'], dtype='<U7')\n\n    "
    a_arr = numpy.asarray(a)
    return _vec_string(a_arr, a_arr.dtype, 'strip', _clean_args(chars))

@array_function_dispatch(_unary_op_dispatcher)
def swapcase(a):
    if False:
        i = 10
        return i + 15
    "\n    Return element-wise a copy of the string with\n    uppercase characters converted to lowercase and vice versa.\n\n    Calls :meth:`str.swapcase` element-wise.\n\n    For 8-bit strings, this method is locale-dependent.\n\n    Parameters\n    ----------\n    a : array_like, {str, unicode}\n        Input array.\n\n    Returns\n    -------\n    out : ndarray, {str, unicode}\n        Output array of str or unicode, depending on input type\n\n    See Also\n    --------\n    str.swapcase\n\n    Examples\n    --------\n    >>> c=np.array(['a1B c','1b Ca','b Ca1','cA1b'],'S5'); c\n    array(['a1B c', '1b Ca', 'b Ca1', 'cA1b'],\n        dtype='|S5')\n    >>> np.char.swapcase(c)\n    array(['A1b C', '1B cA', 'B cA1', 'Ca1B'],\n        dtype='|S5')\n\n    "
    a_arr = numpy.asarray(a)
    return _vec_string(a_arr, a_arr.dtype, 'swapcase')

@array_function_dispatch(_unary_op_dispatcher)
def title(a):
    if False:
        return 10
    "\n    Return element-wise title cased version of string or unicode.\n\n    Title case words start with uppercase characters, all remaining cased\n    characters are lowercase.\n\n    Calls :meth:`str.title` element-wise.\n\n    For 8-bit strings, this method is locale-dependent.\n\n    Parameters\n    ----------\n    a : array_like, {str, unicode}\n        Input array.\n\n    Returns\n    -------\n    out : ndarray\n        Output array of str or unicode, depending on input type\n\n    See Also\n    --------\n    str.title\n\n    Examples\n    --------\n    >>> c=np.array(['a1b c','1b ca','b ca1','ca1b'],'S5'); c\n    array(['a1b c', '1b ca', 'b ca1', 'ca1b'],\n        dtype='|S5')\n    >>> np.char.title(c)\n    array(['A1B C', '1B Ca', 'B Ca1', 'Ca1B'],\n        dtype='|S5')\n\n    "
    a_arr = numpy.asarray(a)
    return _vec_string(a_arr, a_arr.dtype, 'title')

def _translate_dispatcher(a, table, deletechars=None):
    if False:
        return 10
    return (a,)

@array_function_dispatch(_translate_dispatcher)
def translate(a, table, deletechars=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    For each element in `a`, return a copy of the string where all\n    characters occurring in the optional argument `deletechars` are\n    removed, and the remaining characters have been mapped through the\n    given translation table.\n\n    Calls :meth:`str.translate` element-wise.\n\n    Parameters\n    ----------\n    a : array-like of str or unicode\n\n    table : str of length 256\n\n    deletechars : str\n\n    Returns\n    -------\n    out : ndarray\n        Output array of str or unicode, depending on input type\n\n    See Also\n    --------\n    str.translate\n\n    '
    a_arr = numpy.asarray(a)
    if issubclass(a_arr.dtype.type, str_):
        return _vec_string(a_arr, a_arr.dtype, 'translate', (table,))
    else:
        return _vec_string(a_arr, a_arr.dtype, 'translate', [table] + _clean_args(deletechars))

@array_function_dispatch(_unary_op_dispatcher)
def upper(a):
    if False:
        while True:
            i = 10
    "\n    Return an array with the elements converted to uppercase.\n\n    Calls :meth:`str.upper` element-wise.\n\n    For 8-bit strings, this method is locale-dependent.\n\n    Parameters\n    ----------\n    a : array_like, {str, unicode}\n        Input array.\n\n    Returns\n    -------\n    out : ndarray, {str, unicode}\n        Output array of str or unicode, depending on input type\n\n    See Also\n    --------\n    str.upper\n\n    Examples\n    --------\n    >>> c = np.array(['a1b c', '1bca', 'bca1']); c\n    array(['a1b c', '1bca', 'bca1'], dtype='<U5')\n    >>> np.char.upper(c)\n    array(['A1B C', '1BCA', 'BCA1'], dtype='<U5')\n\n    "
    a_arr = numpy.asarray(a)
    return _vec_string(a_arr, a_arr.dtype, 'upper')

def _zfill_dispatcher(a, width):
    if False:
        i = 10
        return i + 15
    return (a,)

@array_function_dispatch(_zfill_dispatcher)
def zfill(a, width):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return the numeric string left-filled with zeros\n\n    Calls :meth:`str.zfill` element-wise.\n\n    Parameters\n    ----------\n    a : array_like, {str, unicode}\n        Input array.\n    width : int\n        Width of string to left-fill elements in `a`.\n\n    Returns\n    -------\n    out : ndarray, {str, unicode}\n        Output array of str or unicode, depending on input type\n\n    See Also\n    --------\n    str.zfill\n\n    '
    a_arr = numpy.asarray(a)
    width_arr = numpy.asarray(width)
    size = int(numpy.max(width_arr.flat))
    return _vec_string(a_arr, type(a_arr.dtype)(size), 'zfill', (width_arr,))

@array_function_dispatch(_unary_op_dispatcher)
def isnumeric(a):
    if False:
        return 10
    "\n    For each element, return True if there are only numeric\n    characters in the element.\n\n    Calls :meth:`str.isnumeric` element-wise.\n\n    Numeric characters include digit characters, and all characters\n    that have the Unicode numeric value property, e.g. ``U+2155,\n    VULGAR FRACTION ONE FIFTH``.\n\n    Parameters\n    ----------\n    a : array_like, unicode\n        Input array.\n\n    Returns\n    -------\n    out : ndarray, bool\n        Array of booleans of same shape as `a`.\n\n    See Also\n    --------\n    str.isnumeric\n\n    Examples\n    --------\n    >>> np.char.isnumeric(['123', '123abc', '9.0', '1/4', 'VIII'])\n    array([ True, False, False, False, False])\n\n    "
    return numpy._core.umath.isnumeric(a)

@array_function_dispatch(_unary_op_dispatcher)
def isdecimal(a):
    if False:
        for i in range(10):
            print('nop')
    "\n    For each element, return True if there are only decimal\n    characters in the element.\n\n    Calls :meth:`str.isdecimal` element-wise.\n\n    Decimal characters include digit characters, and all characters\n    that can be used to form decimal-radix numbers,\n    e.g. ``U+0660, ARABIC-INDIC DIGIT ZERO``.\n\n    Parameters\n    ----------\n    a : array_like, unicode\n        Input array.\n\n    Returns\n    -------\n    out : ndarray, bool\n        Array of booleans identical in shape to `a`.\n\n    See Also\n    --------\n    str.isdecimal\n\n    Examples\n    --------\n    >>> np.char.isdecimal(['12345', '4.99', '123ABC', ''])\n    array([ True, False, False, False])\n\n    "
    return numpy._core.umath.isdecimal(a)

@set_module('numpy.char')
class chararray(ndarray):
    """
    chararray(shape, itemsize=1, unicode=False, buffer=None, offset=0,
              strides=None, order=None)

    Provides a convenient view on arrays of string and unicode values.

    .. note::
       The `chararray` class exists for backwards compatibility with
       Numarray, it is not recommended for new development. Starting from numpy
       1.4, if one needs arrays of strings, it is recommended to use arrays of
       `dtype` `~numpy.object_`, `~numpy.bytes_` or `~numpy.str_`, and use
       the free functions in the `numpy.char` module for fast vectorized
       string operations.

    Versus a NumPy array of dtype `~numpy.bytes_` or `~numpy.str_`, this
    class adds the following functionality:

    1) values automatically have whitespace removed from the end
       when indexed

    2) comparison operators automatically remove whitespace from the
       end when comparing values

    3) vectorized string operations are provided as methods
       (e.g. `.endswith`) and infix operators (e.g. ``"+", "*", "%"``)

    chararrays should be created using `numpy.char.array` or
    `numpy.char.asarray`, rather than this constructor directly.

    This constructor creates the array, using `buffer` (with `offset`
    and `strides`) if it is not ``None``. If `buffer` is ``None``, then
    constructs a new array with `strides` in "C order", unless both
    ``len(shape) >= 2`` and ``order='F'``, in which case `strides`
    is in "Fortran order".

    Methods
    -------
    astype
    argsort
    copy
    count
    decode
    dump
    dumps
    encode
    endswith
    expandtabs
    fill
    find
    flatten
    getfield
    index
    isalnum
    isalpha
    isdecimal
    isdigit
    islower
    isnumeric
    isspace
    istitle
    isupper
    item
    join
    ljust
    lower
    lstrip
    nonzero
    put
    ravel
    repeat
    replace
    reshape
    resize
    rfind
    rindex
    rjust
    rsplit
    rstrip
    searchsorted
    setfield
    setflags
    sort
    split
    splitlines
    squeeze
    startswith
    strip
    swapaxes
    swapcase
    take
    title
    tofile
    tolist
    tostring
    translate
    transpose
    upper
    view
    zfill

    Parameters
    ----------
    shape : tuple
        Shape of the array.
    itemsize : int, optional
        Length of each array element, in number of characters. Default is 1.
    unicode : bool, optional
        Are the array elements of type unicode (True) or string (False).
        Default is False.
    buffer : object exposing the buffer interface or str, optional
        Memory address of the start of the array data.  Default is None,
        in which case a new array is created.
    offset : int, optional
        Fixed stride displacement from the beginning of an axis?
        Default is 0. Needs to be >=0.
    strides : array_like of ints, optional
        Strides for the array (see `~numpy.ndarray.strides` for
        full description). Default is None.
    order : {'C', 'F'}, optional
        The order in which the array data is stored in memory: 'C' ->
        "row major" order (the default), 'F' -> "column major"
        (Fortran) order.

    Examples
    --------
    >>> charar = np.char.chararray((3, 3))
    >>> charar[:] = 'a'
    >>> charar
    chararray([[b'a', b'a', b'a'],
               [b'a', b'a', b'a'],
               [b'a', b'a', b'a']], dtype='|S1')

    >>> charar = np.char.chararray(charar.shape, itemsize=5)
    >>> charar[:] = 'abc'
    >>> charar
    chararray([[b'abc', b'abc', b'abc'],
               [b'abc', b'abc', b'abc'],
               [b'abc', b'abc', b'abc']], dtype='|S5')

    """

    def __new__(subtype, shape, itemsize=1, unicode=False, buffer=None, offset=0, strides=None, order='C'):
        if False:
            i = 10
            return i + 15
        if unicode:
            dtype = str_
        else:
            dtype = bytes_
        itemsize = int(itemsize)
        if isinstance(buffer, str):
            filler = buffer
            buffer = None
        else:
            filler = None
        if buffer is None:
            self = ndarray.__new__(subtype, shape, (dtype, itemsize), order=order)
        else:
            self = ndarray.__new__(subtype, shape, (dtype, itemsize), buffer=buffer, offset=offset, strides=strides, order=order)
        if filler is not None:
            self[...] = filler
        return self

    def __array_wrap__(self, arr, context=None):
        if False:
            print('Hello World!')
        if arr.dtype.char in 'SUbc':
            return arr.view(type(self))
        return arr

    def __array_finalize__(self, obj):
        if False:
            while True:
                i = 10
        if self.dtype.char not in 'SUbc':
            raise ValueError('Can only create a chararray from string data.')

    def __getitem__(self, obj):
        if False:
            print('Hello World!')
        val = ndarray.__getitem__(self, obj)
        if isinstance(val, character):
            temp = val.rstrip()
            if len(temp) == 0:
                val = ''
            else:
                val = temp
        return val

    def __eq__(self, other):
        if False:
            print('Hello World!')
        '\n        Return (self == other) element-wise.\n\n        See Also\n        --------\n        equal\n        '
        return equal(self, other)

    def __ne__(self, other):
        if False:
            print('Hello World!')
        '\n        Return (self != other) element-wise.\n\n        See Also\n        --------\n        not_equal\n        '
        return not_equal(self, other)

    def __ge__(self, other):
        if False:
            print('Hello World!')
        '\n        Return (self >= other) element-wise.\n\n        See Also\n        --------\n        greater_equal\n        '
        return greater_equal(self, other)

    def __le__(self, other):
        if False:
            while True:
                i = 10
        '\n        Return (self <= other) element-wise.\n\n        See Also\n        --------\n        less_equal\n        '
        return less_equal(self, other)

    def __gt__(self, other):
        if False:
            i = 10
            return i + 15
        '\n        Return (self > other) element-wise.\n\n        See Also\n        --------\n        greater\n        '
        return greater(self, other)

    def __lt__(self, other):
        if False:
            while True:
                i = 10
        '\n        Return (self < other) element-wise.\n\n        See Also\n        --------\n        less\n        '
        return less(self, other)

    def __add__(self, other):
        if False:
            while True:
                i = 10
        '\n        Return (self + other), that is string concatenation,\n        element-wise for a pair of array_likes of str or unicode.\n\n        See Also\n        --------\n        add\n        '
        return add(self, other)

    def __radd__(self, other):
        if False:
            while True:
                i = 10
        '\n        Return (other + self), that is string concatenation,\n        element-wise for a pair of array_likes of `bytes_` or `str_`.\n\n        See Also\n        --------\n        add\n        '
        return add(other, self)

    def __mul__(self, i):
        if False:
            while True:
                i = 10
        '\n        Return (self * i), that is string multiple concatenation,\n        element-wise.\n\n        See Also\n        --------\n        multiply\n        '
        return asarray(multiply(self, i))

    def __rmul__(self, i):
        if False:
            print('Hello World!')
        '\n        Return (self * i), that is string multiple concatenation,\n        element-wise.\n\n        See Also\n        --------\n        multiply\n        '
        return asarray(multiply(self, i))

    def __mod__(self, i):
        if False:
            return 10
        '\n        Return (self % i), that is pre-Python 2.6 string formatting\n        (interpolation), element-wise for a pair of array_likes of `bytes_`\n        or `str_`.\n\n        See Also\n        --------\n        mod\n        '
        return asarray(mod(self, i))

    def __rmod__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return NotImplemented

    def argsort(self, axis=-1, kind=None, order=None):
        if False:
            return 10
        '\n        Return the indices that sort the array lexicographically.\n\n        For full documentation see `numpy.argsort`, for which this method is\n        in fact merely a "thin wrapper."\n\n        Examples\n        --------\n        >>> c = np.array([\'a1b c\', \'1b ca\', \'b ca1\', \'Ca1b\'], \'S5\')\n        >>> c = c.view(np.char.chararray); c\n        chararray([\'a1b c\', \'1b ca\', \'b ca1\', \'Ca1b\'],\n              dtype=\'|S5\')\n        >>> c[c.argsort()]\n        chararray([\'1b ca\', \'Ca1b\', \'a1b c\', \'b ca1\'],\n              dtype=\'|S5\')\n\n        '
        return self.__array__().argsort(axis, kind, order)
    argsort.__doc__ = ndarray.argsort.__doc__

    def capitalize(self):
        if False:
            while True:
                i = 10
        '\n        Return a copy of `self` with only the first character of each element\n        capitalized.\n\n        See Also\n        --------\n        char.capitalize\n\n        '
        return asarray(capitalize(self))

    def center(self, width, fillchar=' '):
        if False:
            return 10
        '\n        Return a copy of `self` with its elements centered in a\n        string of length `width`.\n\n        See Also\n        --------\n        center\n        '
        return asarray(center(self, width, fillchar))

    def count(self, sub, start=0, end=None):
        if False:
            return 10
        '\n        Returns an array with the number of non-overlapping occurrences of\n        substring `sub` in the range [`start`, `end`].\n\n        See Also\n        --------\n        char.count\n\n        '
        return count(self, sub, start, end)

    def decode(self, encoding=None, errors=None):
        if False:
            while True:
                i = 10
        '\n        Calls ``bytes.decode`` element-wise.\n\n        See Also\n        --------\n        char.decode\n\n        '
        return decode(self, encoding, errors)

    def encode(self, encoding=None, errors=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Calls :meth:`str.encode` element-wise.\n\n        See Also\n        --------\n        char.encode\n\n        '
        return encode(self, encoding, errors)

    def endswith(self, suffix, start=0, end=None):
        if False:
            while True:
                i = 10
        '\n        Returns a boolean array which is `True` where the string element\n        in `self` ends with `suffix`, otherwise `False`.\n\n        See Also\n        --------\n        char.endswith\n\n        '
        return endswith(self, suffix, start, end)

    def expandtabs(self, tabsize=8):
        if False:
            i = 10
            return i + 15
        '\n        Return a copy of each string element where all tab characters are\n        replaced by one or more spaces.\n\n        See Also\n        --------\n        char.expandtabs\n\n        '
        return asarray(expandtabs(self, tabsize))

    def find(self, sub, start=0, end=None):
        if False:
            return 10
        '\n        For each element, return the lowest index in the string where\n        substring `sub` is found.\n\n        See Also\n        --------\n        char.find\n\n        '
        return find(self, sub, start, end)

    def index(self, sub, start=0, end=None):
        if False:
            return 10
        '\n        Like `find`, but raises :exc:`ValueError` when the substring is not\n        found.\n\n        See Also\n        --------\n        char.index\n\n        '
        return index(self, sub, start, end)

    def isalnum(self):
        if False:
            return 10
        '\n        Returns true for each element if all characters in the string\n        are alphanumeric and there is at least one character, false\n        otherwise.\n\n        See Also\n        --------\n        char.isalnum\n\n        '
        return isalnum(self)

    def isalpha(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns true for each element if all characters in the string\n        are alphabetic and there is at least one character, false\n        otherwise.\n\n        See Also\n        --------\n        char.isalpha\n\n        '
        return isalpha(self)

    def isdigit(self):
        if False:
            return 10
        '\n        Returns true for each element if all characters in the string are\n        digits and there is at least one character, false otherwise.\n\n        See Also\n        --------\n        char.isdigit\n\n        '
        return isdigit(self)

    def islower(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns true for each element if all cased characters in the\n        string are lowercase and there is at least one cased character,\n        false otherwise.\n\n        See Also\n        --------\n        char.islower\n\n        '
        return islower(self)

    def isspace(self):
        if False:
            print('Hello World!')
        '\n        Returns true for each element if there are only whitespace\n        characters in the string and there is at least one character,\n        false otherwise.\n\n        See Also\n        --------\n        char.isspace\n\n        '
        return isspace(self)

    def istitle(self):
        if False:
            return 10
        '\n        Returns true for each element if the element is a titlecased\n        string and there is at least one character, false otherwise.\n\n        See Also\n        --------\n        char.istitle\n\n        '
        return istitle(self)

    def isupper(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns true for each element if all cased characters in the\n        string are uppercase and there is at least one character, false\n        otherwise.\n\n        See Also\n        --------\n        char.isupper\n\n        '
        return isupper(self)

    def join(self, seq):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return a string which is the concatenation of the strings in the\n        sequence `seq`.\n\n        See Also\n        --------\n        char.join\n\n        '
        return join(self, seq)

    def ljust(self, width, fillchar=' '):
        if False:
            while True:
                i = 10
        '\n        Return an array with the elements of `self` left-justified in a\n        string of length `width`.\n\n        See Also\n        --------\n        char.ljust\n\n        '
        return asarray(ljust(self, width, fillchar))

    def lower(self):
        if False:
            return 10
        '\n        Return an array with the elements of `self` converted to\n        lowercase.\n\n        See Also\n        --------\n        char.lower\n\n        '
        return asarray(lower(self))

    def lstrip(self, chars=None):
        if False:
            print('Hello World!')
        '\n        For each element in `self`, return a copy with the leading characters\n        removed.\n\n        See Also\n        --------\n        char.lstrip\n\n        '
        return asarray(lstrip(self, chars))

    def partition(self, sep):
        if False:
            print('Hello World!')
        '\n        Partition each element in `self` around `sep`.\n\n        See Also\n        --------\n        partition\n        '
        return asarray(partition(self, sep))

    def replace(self, old, new, count=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        For each element in `self`, return a copy of the string with all\n        occurrences of substring `old` replaced by `new`.\n\n        See Also\n        --------\n        char.replace\n\n        '
        return asarray(replace(self, old, new, count))

    def rfind(self, sub, start=0, end=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        For each element in `self`, return the highest index in the string\n        where substring `sub` is found, such that `sub` is contained\n        within [`start`, `end`].\n\n        See Also\n        --------\n        char.rfind\n\n        '
        return rfind(self, sub, start, end)

    def rindex(self, sub, start=0, end=None):
        if False:
            i = 10
            return i + 15
        '\n        Like `rfind`, but raises :exc:`ValueError` when the substring `sub` is\n        not found.\n\n        See Also\n        --------\n        char.rindex\n\n        '
        return rindex(self, sub, start, end)

    def rjust(self, width, fillchar=' '):
        if False:
            i = 10
            return i + 15
        '\n        Return an array with the elements of `self`\n        right-justified in a string of length `width`.\n\n        See Also\n        --------\n        char.rjust\n\n        '
        return asarray(rjust(self, width, fillchar))

    def rpartition(self, sep):
        if False:
            while True:
                i = 10
        '\n        Partition each element in `self` around `sep`.\n\n        See Also\n        --------\n        rpartition\n        '
        return asarray(rpartition(self, sep))

    def rsplit(self, sep=None, maxsplit=None):
        if False:
            i = 10
            return i + 15
        '\n        For each element in `self`, return a list of the words in\n        the string, using `sep` as the delimiter string.\n\n        See Also\n        --------\n        char.rsplit\n\n        '
        return rsplit(self, sep, maxsplit)

    def rstrip(self, chars=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        For each element in `self`, return a copy with the trailing\n        characters removed.\n\n        See Also\n        --------\n        char.rstrip\n\n        '
        return asarray(rstrip(self, chars))

    def split(self, sep=None, maxsplit=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        For each element in `self`, return a list of the words in the\n        string, using `sep` as the delimiter string.\n\n        See Also\n        --------\n        char.split\n\n        '
        return split(self, sep, maxsplit)

    def splitlines(self, keepends=None):
        if False:
            while True:
                i = 10
        '\n        For each element in `self`, return a list of the lines in the\n        element, breaking at line boundaries.\n\n        See Also\n        --------\n        char.splitlines\n\n        '
        return splitlines(self, keepends)

    def startswith(self, prefix, start=0, end=None):
        if False:
            return 10
        '\n        Returns a boolean array which is `True` where the string element\n        in `self` starts with `prefix`, otherwise `False`.\n\n        See Also\n        --------\n        char.startswith\n\n        '
        return startswith(self, prefix, start, end)

    def strip(self, chars=None):
        if False:
            return 10
        '\n        For each element in `self`, return a copy with the leading and\n        trailing characters removed.\n\n        See Also\n        --------\n        char.strip\n\n        '
        return asarray(strip(self, chars))

    def swapcase(self):
        if False:
            print('Hello World!')
        '\n        For each element in `self`, return a copy of the string with\n        uppercase characters converted to lowercase and vice versa.\n\n        See Also\n        --------\n        char.swapcase\n\n        '
        return asarray(swapcase(self))

    def title(self):
        if False:
            while True:
                i = 10
        '\n        For each element in `self`, return a titlecased version of the\n        string: words start with uppercase characters, all remaining cased\n        characters are lowercase.\n\n        See Also\n        --------\n        char.title\n\n        '
        return asarray(title(self))

    def translate(self, table, deletechars=None):
        if False:
            while True:
                i = 10
        '\n        For each element in `self`, return a copy of the string where\n        all characters occurring in the optional argument\n        `deletechars` are removed, and the remaining characters have\n        been mapped through the given translation table.\n\n        See Also\n        --------\n        char.translate\n\n        '
        return asarray(translate(self, table, deletechars))

    def upper(self):
        if False:
            print('Hello World!')
        '\n        Return an array with the elements of `self` converted to\n        uppercase.\n\n        See Also\n        --------\n        char.upper\n\n        '
        return asarray(upper(self))

    def zfill(self, width):
        if False:
            return 10
        '\n        Return the numeric string left-filled with zeros in a string of\n        length `width`.\n\n        See Also\n        --------\n        char.zfill\n\n        '
        return asarray(zfill(self, width))

    def isnumeric(self):
        if False:
            i = 10
            return i + 15
        '\n        For each element in `self`, return True if there are only\n        numeric characters in the element.\n\n        See Also\n        --------\n        char.isnumeric\n\n        '
        return isnumeric(self)

    def isdecimal(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        For each element in `self`, return True if there are only\n        decimal characters in the element.\n\n        See Also\n        --------\n        char.isdecimal\n\n        '
        return isdecimal(self)

@set_module('numpy.char')
def array(obj, itemsize=None, copy=True, unicode=None, order=None):
    if False:
        while True:
            i = 10
    "\n    Create a `~numpy.char.chararray`.\n\n    .. note::\n       This class is provided for numarray backward-compatibility.\n       New code (not concerned with numarray compatibility) should use\n       arrays of type `bytes_` or `str_` and use the free functions\n       in :mod:`numpy.char` for fast vectorized string operations instead.\n\n    Versus a NumPy array of dtype `bytes_` or `str_`, this\n    class adds the following functionality:\n\n    1) values automatically have whitespace removed from the end\n       when indexed\n\n    2) comparison operators automatically remove whitespace from the\n       end when comparing values\n\n    3) vectorized string operations are provided as methods\n       (e.g. `chararray.endswith <numpy.char.chararray.endswith>`)\n       and infix operators (e.g. ``+, *, %``)\n\n    Parameters\n    ----------\n    obj : array of str or unicode-like\n\n    itemsize : int, optional\n        `itemsize` is the number of characters per scalar in the\n        resulting array.  If `itemsize` is None, and `obj` is an\n        object array or a Python list, the `itemsize` will be\n        automatically determined.  If `itemsize` is provided and `obj`\n        is of type str or unicode, then the `obj` string will be\n        chunked into `itemsize` pieces.\n\n    copy : bool, optional\n        If true (default), then the object is copied.  Otherwise, a copy\n        will only be made if __array__ returns a copy, if obj is a\n        nested sequence, or if a copy is needed to satisfy any of the other\n        requirements (`itemsize`, unicode, `order`, etc.).\n\n    unicode : bool, optional\n        When true, the resulting `~numpy.char.chararray` can contain Unicode\n        characters, when false only 8-bit characters.  If unicode is\n        None and `obj` is one of the following:\n\n        - a `~numpy.char.chararray`,\n        - an ndarray of type `str_` or `unicode_`\n        - a Python str or unicode object,\n\n        then the unicode setting of the output array will be\n        automatically determined.\n\n    order : {'C', 'F', 'A'}, optional\n        Specify the order of the array.  If order is 'C' (default), then the\n        array will be in C-contiguous order (last-index varies the\n        fastest).  If order is 'F', then the returned array\n        will be in Fortran-contiguous order (first-index varies the\n        fastest).  If order is 'A', then the returned array may\n        be in any order (either C-, Fortran-contiguous, or even\n        discontiguous).\n    "
    if isinstance(obj, (bytes, str)):
        if unicode is None:
            if isinstance(obj, str):
                unicode = True
            else:
                unicode = False
        if itemsize is None:
            itemsize = len(obj)
        shape = len(obj) // itemsize
        return chararray(shape, itemsize=itemsize, unicode=unicode, buffer=obj, order=order)
    if isinstance(obj, (list, tuple)):
        obj = numpy.asarray(obj)
    if isinstance(obj, ndarray) and issubclass(obj.dtype.type, character):
        if not isinstance(obj, chararray):
            obj = obj.view(chararray)
        if itemsize is None:
            itemsize = obj.itemsize
            if issubclass(obj.dtype.type, str_):
                itemsize //= 4
        if unicode is None:
            if issubclass(obj.dtype.type, str_):
                unicode = True
            else:
                unicode = False
        if unicode:
            dtype = str_
        else:
            dtype = bytes_
        if order is not None:
            obj = numpy.asarray(obj, order=order)
        if copy or itemsize != obj.itemsize or (not unicode and isinstance(obj, str_)) or (unicode and isinstance(obj, bytes_)):
            obj = obj.astype((dtype, int(itemsize)))
        return obj
    if isinstance(obj, ndarray) and issubclass(obj.dtype.type, object):
        if itemsize is None:
            obj = obj.tolist()
    if unicode:
        dtype = str_
    else:
        dtype = bytes_
    if itemsize is None:
        val = narray(obj, dtype=dtype, order=order, subok=True)
    else:
        val = narray(obj, dtype=(dtype, itemsize), order=order, subok=True)
    return val.view(chararray)

@set_module('numpy.char')
def asarray(obj, itemsize=None, unicode=None, order=None):
    if False:
        i = 10
        return i + 15
    "\n    Convert the input to a `~numpy.char.chararray`, copying the data only if\n    necessary.\n\n    Versus a NumPy array of dtype `bytes_` or `str_`, this\n    class adds the following functionality:\n\n    1) values automatically have whitespace removed from the end\n       when indexed\n\n    2) comparison operators automatically remove whitespace from the\n       end when comparing values\n\n    3) vectorized string operations are provided as methods\n       (e.g. `chararray.endswith <numpy.char.chararray.endswith>`)\n       and infix operators (e.g. ``+``, ``*``, ``%``)\n\n    Parameters\n    ----------\n    obj : array of str or unicode-like\n\n    itemsize : int, optional\n        `itemsize` is the number of characters per scalar in the\n        resulting array.  If `itemsize` is None, and `obj` is an\n        object array or a Python list, the `itemsize` will be\n        automatically determined.  If `itemsize` is provided and `obj`\n        is of type str or unicode, then the `obj` string will be\n        chunked into `itemsize` pieces.\n\n    unicode : bool, optional\n        When true, the resulting `~numpy.char.chararray` can contain Unicode\n        characters, when false only 8-bit characters.  If unicode is\n        None and `obj` is one of the following:\n\n        - a `~numpy.char.chararray`,\n        - an ndarray of type `str_` or `unicode_`\n        - a Python str or unicode object,\n\n        then the unicode setting of the output array will be\n        automatically determined.\n\n    order : {'C', 'F'}, optional\n        Specify the order of the array.  If order is 'C' (default), then the\n        array will be in C-contiguous order (last-index varies the\n        fastest).  If order is 'F', then the returned array\n        will be in Fortran-contiguous order (first-index varies the\n        fastest).\n    "
    return array(obj, itemsize, copy=False, unicode=unicode, order=order)