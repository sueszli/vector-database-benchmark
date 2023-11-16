"""
Wrapper functions to more user-friendly calling of certain math functions
whose output data-type is different than the input data-type in certain
domains of the input.

For example, for functions like `log` with branch cuts, the versions in this
module provide the mathematically valid answers in the complex plane::

  >>> import math
  >>> np.emath.log(-math.exp(1)) == (1+1j*math.pi)
  True

Similarly, `sqrt`, other base logarithms, `power` and trig functions are
correctly handled.  See their respective docstrings for specific examples.

Functions
---------

.. autosummary::
   :toctree: generated/

   sqrt
   log
   log2
   logn
   log10
   power
   arccos
   arcsin
   arctanh

"""
import numpy._core.numeric as nx
import numpy._core.numerictypes as nt
from numpy._core.numeric import asarray, any
from numpy._core.overrides import array_function_dispatch
from numpy.lib._type_check_impl import isreal
__all__ = ['sqrt', 'log', 'log2', 'logn', 'log10', 'power', 'arccos', 'arcsin', 'arctanh']
_ln2 = nx.log(2.0)

def _tocomplex(arr):
    if False:
        i = 10
        return i + 15
    "Convert its input `arr` to a complex array.\n\n    The input is returned as a complex array of the smallest type that will fit\n    the original data: types like single, byte, short, etc. become csingle,\n    while others become cdouble.\n\n    A copy of the input is always made.\n\n    Parameters\n    ----------\n    arr : array\n\n    Returns\n    -------\n    array\n        An array with the same input data as the input but in complex form.\n\n    Examples\n    --------\n\n    First, consider an input of type short:\n\n    >>> a = np.array([1,2,3],np.short)\n\n    >>> ac = np.lib.scimath._tocomplex(a); ac\n    array([1.+0.j, 2.+0.j, 3.+0.j], dtype=complex64)\n\n    >>> ac.dtype\n    dtype('complex64')\n\n    If the input is of type double, the output is correspondingly of the\n    complex double type as well:\n\n    >>> b = np.array([1,2,3],np.double)\n\n    >>> bc = np.lib.scimath._tocomplex(b); bc\n    array([1.+0.j, 2.+0.j, 3.+0.j])\n\n    >>> bc.dtype\n    dtype('complex128')\n\n    Note that even if the input was complex to begin with, a copy is still\n    made, since the astype() method always copies:\n\n    >>> c = np.array([1,2,3],np.csingle)\n\n    >>> cc = np.lib.scimath._tocomplex(c); cc\n    array([1.+0.j,  2.+0.j,  3.+0.j], dtype=complex64)\n\n    >>> c *= 2; c\n    array([2.+0.j,  4.+0.j,  6.+0.j], dtype=complex64)\n\n    >>> cc\n    array([1.+0.j,  2.+0.j,  3.+0.j], dtype=complex64)\n    "
    if issubclass(arr.dtype.type, (nt.single, nt.byte, nt.short, nt.ubyte, nt.ushort, nt.csingle)):
        return arr.astype(nt.csingle)
    else:
        return arr.astype(nt.cdouble)

def _fix_real_lt_zero(x):
    if False:
        i = 10
        return i + 15
    'Convert `x` to complex if it has real, negative components.\n\n    Otherwise, output is just the array version of the input (via asarray).\n\n    Parameters\n    ----------\n    x : array_like\n\n    Returns\n    -------\n    array\n\n    Examples\n    --------\n    >>> np.lib.scimath._fix_real_lt_zero([1,2])\n    array([1, 2])\n\n    >>> np.lib.scimath._fix_real_lt_zero([-1,2])\n    array([-1.+0.j,  2.+0.j])\n\n    '
    x = asarray(x)
    if any(isreal(x) & (x < 0)):
        x = _tocomplex(x)
    return x

def _fix_int_lt_zero(x):
    if False:
        print('Hello World!')
    'Convert `x` to double if it has real, negative components.\n\n    Otherwise, output is just the array version of the input (via asarray).\n\n    Parameters\n    ----------\n    x : array_like\n\n    Returns\n    -------\n    array\n\n    Examples\n    --------\n    >>> np.lib.scimath._fix_int_lt_zero([1,2])\n    array([1, 2])\n\n    >>> np.lib.scimath._fix_int_lt_zero([-1,2])\n    array([-1.,  2.])\n    '
    x = asarray(x)
    if any(isreal(x) & (x < 0)):
        x = x * 1.0
    return x

def _fix_real_abs_gt_1(x):
    if False:
        for i in range(10):
            print('nop')
    'Convert `x` to complex if it has real components x_i with abs(x_i)>1.\n\n    Otherwise, output is just the array version of the input (via asarray).\n\n    Parameters\n    ----------\n    x : array_like\n\n    Returns\n    -------\n    array\n\n    Examples\n    --------\n    >>> np.lib.scimath._fix_real_abs_gt_1([0,1])\n    array([0, 1])\n\n    >>> np.lib.scimath._fix_real_abs_gt_1([0,2])\n    array([0.+0.j, 2.+0.j])\n    '
    x = asarray(x)
    if any(isreal(x) & (abs(x) > 1)):
        x = _tocomplex(x)
    return x

def _unary_dispatcher(x):
    if False:
        i = 10
        return i + 15
    return (x,)

@array_function_dispatch(_unary_dispatcher)
def sqrt(x):
    if False:
        print('Hello World!')
    '\n    Compute the square root of x.\n\n    For negative input elements, a complex value is returned\n    (unlike `numpy.sqrt` which returns NaN).\n\n    Parameters\n    ----------\n    x : array_like\n       The input value(s).\n\n    Returns\n    -------\n    out : ndarray or scalar\n       The square root of `x`. If `x` was a scalar, so is `out`,\n       otherwise an array is returned.\n\n    See Also\n    --------\n    numpy.sqrt\n\n    Examples\n    --------\n    For real, non-negative inputs this works just like `numpy.sqrt`:\n\n    >>> np.emath.sqrt(1)\n    1.0\n    >>> np.emath.sqrt([1, 4])\n    array([1.,  2.])\n\n    But it automatically handles negative inputs:\n\n    >>> np.emath.sqrt(-1)\n    1j\n    >>> np.emath.sqrt([-1,4])\n    array([0.+1.j, 2.+0.j])\n\n    Different results are expected because:\n    floating point 0.0 and -0.0 are distinct.\n\n    For more control, explicitly use complex() as follows:\n\n    >>> np.emath.sqrt(complex(-4.0, 0.0))\n    2j\n    >>> np.emath.sqrt(complex(-4.0, -0.0))\n    -2j\n    '
    x = _fix_real_lt_zero(x)
    return nx.sqrt(x)

@array_function_dispatch(_unary_dispatcher)
def log(x):
    if False:
        print('Hello World!')
    '\n    Compute the natural logarithm of `x`.\n\n    Return the "principal value" (for a description of this, see `numpy.log`)\n    of :math:`log_e(x)`. For real `x > 0`, this is a real number (``log(0)``\n    returns ``-inf`` and ``log(np.inf)`` returns ``inf``). Otherwise, the\n    complex principle value is returned.\n\n    Parameters\n    ----------\n    x : array_like\n       The value(s) whose log is (are) required.\n\n    Returns\n    -------\n    out : ndarray or scalar\n       The log of the `x` value(s). If `x` was a scalar, so is `out`,\n       otherwise an array is returned.\n\n    See Also\n    --------\n    numpy.log\n\n    Notes\n    -----\n    For a log() that returns ``NAN`` when real `x < 0`, use `numpy.log`\n    (note, however, that otherwise `numpy.log` and this `log` are identical,\n    i.e., both return ``-inf`` for `x = 0`, ``inf`` for `x = inf`, and,\n    notably, the complex principle value if ``x.imag != 0``).\n\n    Examples\n    --------\n    >>> np.emath.log(np.exp(1))\n    1.0\n\n    Negative arguments are handled "correctly" (recall that\n    ``exp(log(x)) == x`` does *not* hold for real ``x < 0``):\n\n    >>> np.emath.log(-np.exp(1)) == (1 + np.pi * 1j)\n    True\n\n    '
    x = _fix_real_lt_zero(x)
    return nx.log(x)

@array_function_dispatch(_unary_dispatcher)
def log10(x):
    if False:
        print('Hello World!')
    '\n    Compute the logarithm base 10 of `x`.\n\n    Return the "principal value" (for a description of this, see\n    `numpy.log10`) of :math:`log_{10}(x)`. For real `x > 0`, this\n    is a real number (``log10(0)`` returns ``-inf`` and ``log10(np.inf)``\n    returns ``inf``). Otherwise, the complex principle value is returned.\n\n    Parameters\n    ----------\n    x : array_like or scalar\n       The value(s) whose log base 10 is (are) required.\n\n    Returns\n    -------\n    out : ndarray or scalar\n       The log base 10 of the `x` value(s). If `x` was a scalar, so is `out`,\n       otherwise an array object is returned.\n\n    See Also\n    --------\n    numpy.log10\n\n    Notes\n    -----\n    For a log10() that returns ``NAN`` when real `x < 0`, use `numpy.log10`\n    (note, however, that otherwise `numpy.log10` and this `log10` are\n    identical, i.e., both return ``-inf`` for `x = 0`, ``inf`` for `x = inf`,\n    and, notably, the complex principle value if ``x.imag != 0``).\n\n    Examples\n    --------\n\n    (We set the printing precision so the example can be auto-tested)\n\n    >>> np.set_printoptions(precision=4)\n\n    >>> np.emath.log10(10**1)\n    1.0\n\n    >>> np.emath.log10([-10**1, -10**2, 10**2])\n    array([1.+1.3644j, 2.+1.3644j, 2.+0.j    ])\n\n    '
    x = _fix_real_lt_zero(x)
    return nx.log10(x)

def _logn_dispatcher(n, x):
    if False:
        print('Hello World!')
    return (n, x)

@array_function_dispatch(_logn_dispatcher)
def logn(n, x):
    if False:
        for i in range(10):
            print('nop')
    '\n    Take log base n of x.\n\n    If `x` contains negative inputs, the answer is computed and returned in the\n    complex domain.\n\n    Parameters\n    ----------\n    n : array_like\n       The integer base(s) in which the log is taken.\n    x : array_like\n       The value(s) whose log base `n` is (are) required.\n\n    Returns\n    -------\n    out : ndarray or scalar\n       The log base `n` of the `x` value(s). If `x` was a scalar, so is\n       `out`, otherwise an array is returned.\n\n    Examples\n    --------\n    >>> np.set_printoptions(precision=4)\n\n    >>> np.emath.logn(2, [4, 8])\n    array([2., 3.])\n    >>> np.emath.logn(2, [-4, -8, 8])\n    array([2.+4.5324j, 3.+4.5324j, 3.+0.j    ])\n\n    '
    x = _fix_real_lt_zero(x)
    n = _fix_real_lt_zero(n)
    return nx.log(x) / nx.log(n)

@array_function_dispatch(_unary_dispatcher)
def log2(x):
    if False:
        i = 10
        return i + 15
    '\n    Compute the logarithm base 2 of `x`.\n\n    Return the "principal value" (for a description of this, see\n    `numpy.log2`) of :math:`log_2(x)`. For real `x > 0`, this is\n    a real number (``log2(0)`` returns ``-inf`` and ``log2(np.inf)`` returns\n    ``inf``). Otherwise, the complex principle value is returned.\n\n    Parameters\n    ----------\n    x : array_like\n       The value(s) whose log base 2 is (are) required.\n\n    Returns\n    -------\n    out : ndarray or scalar\n       The log base 2 of the `x` value(s). If `x` was a scalar, so is `out`,\n       otherwise an array is returned.\n\n    See Also\n    --------\n    numpy.log2\n\n    Notes\n    -----\n    For a log2() that returns ``NAN`` when real `x < 0`, use `numpy.log2`\n    (note, however, that otherwise `numpy.log2` and this `log2` are\n    identical, i.e., both return ``-inf`` for `x = 0`, ``inf`` for `x = inf`,\n    and, notably, the complex principle value if ``x.imag != 0``).\n\n    Examples\n    --------\n    We set the printing precision so the example can be auto-tested:\n\n    >>> np.set_printoptions(precision=4)\n\n    >>> np.emath.log2(8)\n    3.0\n    >>> np.emath.log2([-4, -8, 8])\n    array([2.+4.5324j, 3.+4.5324j, 3.+0.j    ])\n\n    '
    x = _fix_real_lt_zero(x)
    return nx.log2(x)

def _power_dispatcher(x, p):
    if False:
        print('Hello World!')
    return (x, p)

@array_function_dispatch(_power_dispatcher)
def power(x, p):
    if False:
        while True:
            i = 10
    '\n    Return x to the power p, (x**p).\n\n    If `x` contains negative values, the output is converted to the\n    complex domain.\n\n    Parameters\n    ----------\n    x : array_like\n        The input value(s).\n    p : array_like of ints\n        The power(s) to which `x` is raised. If `x` contains multiple values,\n        `p` has to either be a scalar, or contain the same number of values\n        as `x`. In the latter case, the result is\n        ``x[0]**p[0], x[1]**p[1], ...``.\n\n    Returns\n    -------\n    out : ndarray or scalar\n        The result of ``x**p``. If `x` and `p` are scalars, so is `out`,\n        otherwise an array is returned.\n\n    See Also\n    --------\n    numpy.power\n\n    Examples\n    --------\n    >>> np.set_printoptions(precision=4)\n\n    >>> np.emath.power([2, 4], 2)\n    array([ 4, 16])\n    >>> np.emath.power([2, 4], -2)\n    array([0.25  ,  0.0625])\n    >>> np.emath.power([-2, 4], 2)\n    array([ 4.-0.j, 16.+0.j])\n\n    '
    x = _fix_real_lt_zero(x)
    p = _fix_int_lt_zero(p)
    return nx.power(x, p)

@array_function_dispatch(_unary_dispatcher)
def arccos(x):
    if False:
        for i in range(10):
            print('nop')
    '\n    Compute the inverse cosine of x.\n\n    Return the "principal value" (for a description of this, see\n    `numpy.arccos`) of the inverse cosine of `x`. For real `x` such that\n    `abs(x) <= 1`, this is a real number in the closed interval\n    :math:`[0, \\pi]`.  Otherwise, the complex principle value is returned.\n\n    Parameters\n    ----------\n    x : array_like or scalar\n       The value(s) whose arccos is (are) required.\n\n    Returns\n    -------\n    out : ndarray or scalar\n       The inverse cosine(s) of the `x` value(s). If `x` was a scalar, so\n       is `out`, otherwise an array object is returned.\n\n    See Also\n    --------\n    numpy.arccos\n\n    Notes\n    -----\n    For an arccos() that returns ``NAN`` when real `x` is not in the\n    interval ``[-1,1]``, use `numpy.arccos`.\n\n    Examples\n    --------\n    >>> np.set_printoptions(precision=4)\n\n    >>> np.emath.arccos(1) # a scalar is returned\n    0.0\n\n    >>> np.emath.arccos([1,2])\n    array([0.-0.j   , 0.-1.317j])\n\n    '
    x = _fix_real_abs_gt_1(x)
    return nx.arccos(x)

@array_function_dispatch(_unary_dispatcher)
def arcsin(x):
    if False:
        i = 10
        return i + 15
    '\n    Compute the inverse sine of x.\n\n    Return the "principal value" (for a description of this, see\n    `numpy.arcsin`) of the inverse sine of `x`. For real `x` such that\n    `abs(x) <= 1`, this is a real number in the closed interval\n    :math:`[-\\pi/2, \\pi/2]`.  Otherwise, the complex principle value is\n    returned.\n\n    Parameters\n    ----------\n    x : array_like or scalar\n       The value(s) whose arcsin is (are) required.\n\n    Returns\n    -------\n    out : ndarray or scalar\n       The inverse sine(s) of the `x` value(s). If `x` was a scalar, so\n       is `out`, otherwise an array object is returned.\n\n    See Also\n    --------\n    numpy.arcsin\n\n    Notes\n    -----\n    For an arcsin() that returns ``NAN`` when real `x` is not in the\n    interval ``[-1,1]``, use `numpy.arcsin`.\n\n    Examples\n    --------\n    >>> np.set_printoptions(precision=4)\n\n    >>> np.emath.arcsin(0)\n    0.0\n\n    >>> np.emath.arcsin([0,1])\n    array([0.    , 1.5708])\n\n    '
    x = _fix_real_abs_gt_1(x)
    return nx.arcsin(x)

@array_function_dispatch(_unary_dispatcher)
def arctanh(x):
    if False:
        return 10
    '\n    Compute the inverse hyperbolic tangent of `x`.\n\n    Return the "principal value" (for a description of this, see\n    `numpy.arctanh`) of ``arctanh(x)``. For real `x` such that\n    ``abs(x) < 1``, this is a real number.  If `abs(x) > 1`, or if `x` is\n    complex, the result is complex. Finally, `x = 1` returns``inf`` and\n    ``x=-1`` returns ``-inf``.\n\n    Parameters\n    ----------\n    x : array_like\n       The value(s) whose arctanh is (are) required.\n\n    Returns\n    -------\n    out : ndarray or scalar\n       The inverse hyperbolic tangent(s) of the `x` value(s). If `x` was\n       a scalar so is `out`, otherwise an array is returned.\n\n\n    See Also\n    --------\n    numpy.arctanh\n\n    Notes\n    -----\n    For an arctanh() that returns ``NAN`` when real `x` is not in the\n    interval ``(-1,1)``, use `numpy.arctanh` (this latter, however, does\n    return +/-inf for ``x = +/-1``).\n\n    Examples\n    --------\n    >>> np.set_printoptions(precision=4)\n\n    >>> from numpy.testing import suppress_warnings\n    >>> with suppress_warnings() as sup:\n    ...     sup.filter(RuntimeWarning)\n    ...     np.emath.arctanh(np.eye(2))\n    array([[inf,  0.],\n           [ 0., inf]])\n    >>> np.emath.arctanh([1j])\n    array([0.+0.7854j])\n\n    '
    x = _fix_real_abs_gt_1(x)
    return nx.arctanh(x)