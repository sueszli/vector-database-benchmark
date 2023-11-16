"""
==================================================
Laguerre Series (:mod:`numpy.polynomial.laguerre`)
==================================================

This module provides a number of objects (mostly functions) useful for
dealing with Laguerre series, including a `Laguerre` class that
encapsulates the usual arithmetic operations.  (General information
on how this module represents and works with such polynomials is in the
docstring for its "parent" sub-package, `numpy.polynomial`).

Classes
-------
.. autosummary::
   :toctree: generated/

   Laguerre

Constants
---------
.. autosummary::
   :toctree: generated/

   lagdomain
   lagzero
   lagone
   lagx

Arithmetic
----------
.. autosummary::
   :toctree: generated/

   lagadd
   lagsub
   lagmulx
   lagmul
   lagdiv
   lagpow
   lagval
   lagval2d
   lagval3d
   laggrid2d
   laggrid3d

Calculus
--------
.. autosummary::
   :toctree: generated/

   lagder
   lagint

Misc Functions
--------------
.. autosummary::
   :toctree: generated/

   lagfromroots
   lagroots
   lagvander
   lagvander2d
   lagvander3d
   laggauss
   lagweight
   lagcompanion
   lagfit
   lagtrim
   lagline
   lag2poly
   poly2lag

See also
--------
`numpy.polynomial`

"""
import numpy as np
import numpy.linalg as la
from numpy.lib.array_utils import normalize_axis_index
from . import polyutils as pu
from ._polybase import ABCPolyBase
__all__ = ['lagzero', 'lagone', 'lagx', 'lagdomain', 'lagline', 'lagadd', 'lagsub', 'lagmulx', 'lagmul', 'lagdiv', 'lagpow', 'lagval', 'lagder', 'lagint', 'lag2poly', 'poly2lag', 'lagfromroots', 'lagvander', 'lagfit', 'lagtrim', 'lagroots', 'Laguerre', 'lagval2d', 'lagval3d', 'laggrid2d', 'laggrid3d', 'lagvander2d', 'lagvander3d', 'lagcompanion', 'laggauss', 'lagweight']
lagtrim = pu.trimcoef

def poly2lag(pol):
    if False:
        for i in range(10):
            print('nop')
    '\n    poly2lag(pol)\n\n    Convert a polynomial to a Laguerre series.\n\n    Convert an array representing the coefficients of a polynomial (relative\n    to the "standard" basis) ordered from lowest degree to highest, to an\n    array of the coefficients of the equivalent Laguerre series, ordered\n    from lowest to highest degree.\n\n    Parameters\n    ----------\n    pol : array_like\n        1-D array containing the polynomial coefficients\n\n    Returns\n    -------\n    c : ndarray\n        1-D array containing the coefficients of the equivalent Laguerre\n        series.\n\n    See Also\n    --------\n    lag2poly\n\n    Notes\n    -----\n    The easy way to do conversions between polynomial basis sets\n    is to use the convert method of a class instance.\n\n    Examples\n    --------\n    >>> from numpy.polynomial.laguerre import poly2lag\n    >>> poly2lag(np.arange(4))\n    array([ 23., -63.,  58., -18.])\n\n    '
    [pol] = pu.as_series([pol])
    res = 0
    for p in pol[::-1]:
        res = lagadd(lagmulx(res), p)
    return res

def lag2poly(c):
    if False:
        i = 10
        return i + 15
    '\n    Convert a Laguerre series to a polynomial.\n\n    Convert an array representing the coefficients of a Laguerre series,\n    ordered from lowest degree to highest, to an array of the coefficients\n    of the equivalent polynomial (relative to the "standard" basis) ordered\n    from lowest to highest degree.\n\n    Parameters\n    ----------\n    c : array_like\n        1-D array containing the Laguerre series coefficients, ordered\n        from lowest order term to highest.\n\n    Returns\n    -------\n    pol : ndarray\n        1-D array containing the coefficients of the equivalent polynomial\n        (relative to the "standard" basis) ordered from lowest order term\n        to highest.\n\n    See Also\n    --------\n    poly2lag\n\n    Notes\n    -----\n    The easy way to do conversions between polynomial basis sets\n    is to use the convert method of a class instance.\n\n    Examples\n    --------\n    >>> from numpy.polynomial.laguerre import lag2poly\n    >>> lag2poly([ 23., -63.,  58., -18.])\n    array([0., 1., 2., 3.])\n\n    '
    from .polynomial import polyadd, polysub, polymulx
    [c] = pu.as_series([c])
    n = len(c)
    if n == 1:
        return c
    else:
        c0 = c[-2]
        c1 = c[-1]
        for i in range(n - 1, 1, -1):
            tmp = c0
            c0 = polysub(c[i - 2], c1 * (i - 1) / i)
            c1 = polyadd(tmp, polysub((2 * i - 1) * c1, polymulx(c1)) / i)
        return polyadd(c0, polysub(c1, polymulx(c1)))
lagdomain = np.array([0, 1])
lagzero = np.array([0])
lagone = np.array([1])
lagx = np.array([1, -1])

def lagline(off, scl):
    if False:
        print('Hello World!')
    "\n    Laguerre series whose graph is a straight line.\n\n    Parameters\n    ----------\n    off, scl : scalars\n        The specified line is given by ``off + scl*x``.\n\n    Returns\n    -------\n    y : ndarray\n        This module's representation of the Laguerre series for\n        ``off + scl*x``.\n\n    See Also\n    --------\n    numpy.polynomial.polynomial.polyline\n    numpy.polynomial.chebyshev.chebline\n    numpy.polynomial.legendre.legline\n    numpy.polynomial.hermite.hermline\n    numpy.polynomial.hermite_e.hermeline\n\n    Examples\n    --------\n    >>> from numpy.polynomial.laguerre import lagline, lagval\n    >>> lagval(0,lagline(3, 2))\n    3.0\n    >>> lagval(1,lagline(3, 2))\n    5.0\n\n    "
    if scl != 0:
        return np.array([off + scl, -scl])
    else:
        return np.array([off])

def lagfromroots(roots):
    if False:
        print('Hello World!')
    '\n    Generate a Laguerre series with given roots.\n\n    The function returns the coefficients of the polynomial\n\n    .. math:: p(x) = (x - r_0) * (x - r_1) * ... * (x - r_n),\n\n    in Laguerre form, where the :math:`r_n` are the roots specified in `roots`.\n    If a zero has multiplicity n, then it must appear in `roots` n times.\n    For instance, if 2 is a root of multiplicity three and 3 is a root of\n    multiplicity 2, then `roots` looks something like [2, 2, 2, 3, 3]. The\n    roots can appear in any order.\n\n    If the returned coefficients are `c`, then\n\n    .. math:: p(x) = c_0 + c_1 * L_1(x) + ... +  c_n * L_n(x)\n\n    The coefficient of the last term is not generally 1 for monic\n    polynomials in Laguerre form.\n\n    Parameters\n    ----------\n    roots : array_like\n        Sequence containing the roots.\n\n    Returns\n    -------\n    out : ndarray\n        1-D array of coefficients.  If all roots are real then `out` is a\n        real array, if some of the roots are complex, then `out` is complex\n        even if all the coefficients in the result are real (see Examples\n        below).\n\n    See Also\n    --------\n    numpy.polynomial.polynomial.polyfromroots\n    numpy.polynomial.legendre.legfromroots\n    numpy.polynomial.chebyshev.chebfromroots\n    numpy.polynomial.hermite.hermfromroots\n    numpy.polynomial.hermite_e.hermefromroots\n\n    Examples\n    --------\n    >>> from numpy.polynomial.laguerre import lagfromroots, lagval\n    >>> coef = lagfromroots((-1, 0, 1))\n    >>> lagval((-1, 0, 1), coef)\n    array([0.,  0.,  0.])\n    >>> coef = lagfromroots((-1j, 1j))\n    >>> lagval((-1j, 1j), coef)\n    array([0.+0.j, 0.+0.j])\n\n    '
    return pu._fromroots(lagline, lagmul, roots)

def lagadd(c1, c2):
    if False:
        print('Hello World!')
    '\n    Add one Laguerre series to another.\n\n    Returns the sum of two Laguerre series `c1` + `c2`.  The arguments\n    are sequences of coefficients ordered from lowest order term to\n    highest, i.e., [1,2,3] represents the series ``P_0 + 2*P_1 + 3*P_2``.\n\n    Parameters\n    ----------\n    c1, c2 : array_like\n        1-D arrays of Laguerre series coefficients ordered from low to\n        high.\n\n    Returns\n    -------\n    out : ndarray\n        Array representing the Laguerre series of their sum.\n\n    See Also\n    --------\n    lagsub, lagmulx, lagmul, lagdiv, lagpow\n\n    Notes\n    -----\n    Unlike multiplication, division, etc., the sum of two Laguerre series\n    is a Laguerre series (without having to "reproject" the result onto\n    the basis set) so addition, just like that of "standard" polynomials,\n    is simply "component-wise."\n\n    Examples\n    --------\n    >>> from numpy.polynomial.laguerre import lagadd\n    >>> lagadd([1, 2, 3], [1, 2, 3, 4])\n    array([2.,  4.,  6.,  4.])\n\n\n    '
    return pu._add(c1, c2)

def lagsub(c1, c2):
    if False:
        return 10
    '\n    Subtract one Laguerre series from another.\n\n    Returns the difference of two Laguerre series `c1` - `c2`.  The\n    sequences of coefficients are from lowest order term to highest, i.e.,\n    [1,2,3] represents the series ``P_0 + 2*P_1 + 3*P_2``.\n\n    Parameters\n    ----------\n    c1, c2 : array_like\n        1-D arrays of Laguerre series coefficients ordered from low to\n        high.\n\n    Returns\n    -------\n    out : ndarray\n        Of Laguerre series coefficients representing their difference.\n\n    See Also\n    --------\n    lagadd, lagmulx, lagmul, lagdiv, lagpow\n\n    Notes\n    -----\n    Unlike multiplication, division, etc., the difference of two Laguerre\n    series is a Laguerre series (without having to "reproject" the result\n    onto the basis set) so subtraction, just like that of "standard"\n    polynomials, is simply "component-wise."\n\n    Examples\n    --------\n    >>> from numpy.polynomial.laguerre import lagsub\n    >>> lagsub([1, 2, 3, 4], [1, 2, 3])\n    array([0.,  0.,  0.,  4.])\n\n    '
    return pu._sub(c1, c2)

def lagmulx(c):
    if False:
        i = 10
        return i + 15
    'Multiply a Laguerre series by x.\n\n    Multiply the Laguerre series `c` by x, where x is the independent\n    variable.\n\n\n    Parameters\n    ----------\n    c : array_like\n        1-D array of Laguerre series coefficients ordered from low to\n        high.\n\n    Returns\n    -------\n    out : ndarray\n        Array representing the result of the multiplication.\n\n    See Also\n    --------\n    lagadd, lagsub, lagmul, lagdiv, lagpow\n\n    Notes\n    -----\n    The multiplication uses the recursion relationship for Laguerre\n    polynomials in the form\n\n    .. math::\n\n        xP_i(x) = (-(i + 1)*P_{i + 1}(x) + (2i + 1)P_{i}(x) - iP_{i - 1}(x))\n\n    Examples\n    --------\n    >>> from numpy.polynomial.laguerre import lagmulx\n    >>> lagmulx([1, 2, 3])\n    array([-1.,  -1.,  11.,  -9.])\n\n    '
    [c] = pu.as_series([c])
    if len(c) == 1 and c[0] == 0:
        return c
    prd = np.empty(len(c) + 1, dtype=c.dtype)
    prd[0] = c[0]
    prd[1] = -c[0]
    for i in range(1, len(c)):
        prd[i + 1] = -c[i] * (i + 1)
        prd[i] += c[i] * (2 * i + 1)
        prd[i - 1] -= c[i] * i
    return prd

def lagmul(c1, c2):
    if False:
        for i in range(10):
            print('nop')
    '\n    Multiply one Laguerre series by another.\n\n    Returns the product of two Laguerre series `c1` * `c2`.  The arguments\n    are sequences of coefficients, from lowest order "term" to highest,\n    e.g., [1,2,3] represents the series ``P_0 + 2*P_1 + 3*P_2``.\n\n    Parameters\n    ----------\n    c1, c2 : array_like\n        1-D arrays of Laguerre series coefficients ordered from low to\n        high.\n\n    Returns\n    -------\n    out : ndarray\n        Of Laguerre series coefficients representing their product.\n\n    See Also\n    --------\n    lagadd, lagsub, lagmulx, lagdiv, lagpow\n\n    Notes\n    -----\n    In general, the (polynomial) product of two C-series results in terms\n    that are not in the Laguerre polynomial basis set.  Thus, to express\n    the product as a Laguerre series, it is necessary to "reproject" the\n    product onto said basis set, which may produce "unintuitive" (but\n    correct) results; see Examples section below.\n\n    Examples\n    --------\n    >>> from numpy.polynomial.laguerre import lagmul\n    >>> lagmul([1, 2, 3], [0, 1, 2])\n    array([  8., -13.,  38., -51.,  36.])\n\n    '
    [c1, c2] = pu.as_series([c1, c2])
    if len(c1) > len(c2):
        c = c2
        xs = c1
    else:
        c = c1
        xs = c2
    if len(c) == 1:
        c0 = c[0] * xs
        c1 = 0
    elif len(c) == 2:
        c0 = c[0] * xs
        c1 = c[1] * xs
    else:
        nd = len(c)
        c0 = c[-2] * xs
        c1 = c[-1] * xs
        for i in range(3, len(c) + 1):
            tmp = c0
            nd = nd - 1
            c0 = lagsub(c[-i] * xs, c1 * (nd - 1) / nd)
            c1 = lagadd(tmp, lagsub((2 * nd - 1) * c1, lagmulx(c1)) / nd)
    return lagadd(c0, lagsub(c1, lagmulx(c1)))

def lagdiv(c1, c2):
    if False:
        print('Hello World!')
    '\n    Divide one Laguerre series by another.\n\n    Returns the quotient-with-remainder of two Laguerre series\n    `c1` / `c2`.  The arguments are sequences of coefficients from lowest\n    order "term" to highest, e.g., [1,2,3] represents the series\n    ``P_0 + 2*P_1 + 3*P_2``.\n\n    Parameters\n    ----------\n    c1, c2 : array_like\n        1-D arrays of Laguerre series coefficients ordered from low to\n        high.\n\n    Returns\n    -------\n    [quo, rem] : ndarrays\n        Of Laguerre series coefficients representing the quotient and\n        remainder.\n\n    See Also\n    --------\n    lagadd, lagsub, lagmulx, lagmul, lagpow\n\n    Notes\n    -----\n    In general, the (polynomial) division of one Laguerre series by another\n    results in quotient and remainder terms that are not in the Laguerre\n    polynomial basis set.  Thus, to express these results as a Laguerre\n    series, it is necessary to "reproject" the results onto the Laguerre\n    basis set, which may produce "unintuitive" (but correct) results; see\n    Examples section below.\n\n    Examples\n    --------\n    >>> from numpy.polynomial.laguerre import lagdiv\n    >>> lagdiv([  8., -13.,  38., -51.,  36.], [0, 1, 2])\n    (array([1., 2., 3.]), array([0.]))\n    >>> lagdiv([  9., -12.,  38., -51.,  36.], [0, 1, 2])\n    (array([1., 2., 3.]), array([1., 1.]))\n\n    '
    return pu._div(lagmul, c1, c2)

def lagpow(c, pow, maxpower=16):
    if False:
        while True:
            i = 10
    'Raise a Laguerre series to a power.\n\n    Returns the Laguerre series `c` raised to the power `pow`. The\n    argument `c` is a sequence of coefficients ordered from low to high.\n    i.e., [1,2,3] is the series  ``P_0 + 2*P_1 + 3*P_2.``\n\n    Parameters\n    ----------\n    c : array_like\n        1-D array of Laguerre series coefficients ordered from low to\n        high.\n    pow : integer\n        Power to which the series will be raised\n    maxpower : integer, optional\n        Maximum power allowed. This is mainly to limit growth of the series\n        to unmanageable size. Default is 16\n\n    Returns\n    -------\n    coef : ndarray\n        Laguerre series of power.\n\n    See Also\n    --------\n    lagadd, lagsub, lagmulx, lagmul, lagdiv\n\n    Examples\n    --------\n    >>> from numpy.polynomial.laguerre import lagpow\n    >>> lagpow([1, 2, 3], 2)\n    array([ 14., -16.,  56., -72.,  54.])\n\n    '
    return pu._pow(lagmul, c, pow, maxpower)

def lagder(c, m=1, scl=1, axis=0):
    if False:
        for i in range(10):
            print('nop')
    '\n    Differentiate a Laguerre series.\n\n    Returns the Laguerre series coefficients `c` differentiated `m` times\n    along `axis`.  At each iteration the result is multiplied by `scl` (the\n    scaling factor is for use in a linear change of variable). The argument\n    `c` is an array of coefficients from low to high degree along each\n    axis, e.g., [1,2,3] represents the series ``1*L_0 + 2*L_1 + 3*L_2``\n    while [[1,2],[1,2]] represents ``1*L_0(x)*L_0(y) + 1*L_1(x)*L_0(y) +\n    2*L_0(x)*L_1(y) + 2*L_1(x)*L_1(y)`` if axis=0 is ``x`` and axis=1 is\n    ``y``.\n\n    Parameters\n    ----------\n    c : array_like\n        Array of Laguerre series coefficients. If `c` is multidimensional\n        the different axis correspond to different variables with the\n        degree in each axis given by the corresponding index.\n    m : int, optional\n        Number of derivatives taken, must be non-negative. (Default: 1)\n    scl : scalar, optional\n        Each differentiation is multiplied by `scl`.  The end result is\n        multiplication by ``scl**m``.  This is for use in a linear change of\n        variable. (Default: 1)\n    axis : int, optional\n        Axis over which the derivative is taken. (Default: 0).\n\n        .. versionadded:: 1.7.0\n\n    Returns\n    -------\n    der : ndarray\n        Laguerre series of the derivative.\n\n    See Also\n    --------\n    lagint\n\n    Notes\n    -----\n    In general, the result of differentiating a Laguerre series does not\n    resemble the same operation on a power series. Thus the result of this\n    function may be "unintuitive," albeit correct; see Examples section\n    below.\n\n    Examples\n    --------\n    >>> from numpy.polynomial.laguerre import lagder\n    >>> lagder([ 1.,  1.,  1., -3.])\n    array([1.,  2.,  3.])\n    >>> lagder([ 1.,  0.,  0., -4.,  3.], m=2)\n    array([1.,  2.,  3.])\n\n    '
    c = np.array(c, ndmin=1, copy=True)
    if c.dtype.char in '?bBhHiIlLqQpP':
        c = c.astype(np.double)
    cnt = pu._as_int(m, 'the order of derivation')
    iaxis = pu._as_int(axis, 'the axis')
    if cnt < 0:
        raise ValueError('The order of derivation must be non-negative')
    iaxis = normalize_axis_index(iaxis, c.ndim)
    if cnt == 0:
        return c
    c = np.moveaxis(c, iaxis, 0)
    n = len(c)
    if cnt >= n:
        c = c[:1] * 0
    else:
        for i in range(cnt):
            n = n - 1
            c *= scl
            der = np.empty((n,) + c.shape[1:], dtype=c.dtype)
            for j in range(n, 1, -1):
                der[j - 1] = -c[j]
                c[j - 1] += c[j]
            der[0] = -c[1]
            c = der
    c = np.moveaxis(c, 0, iaxis)
    return c

def lagint(c, m=1, k=[], lbnd=0, scl=1, axis=0):
    if False:
        for i in range(10):
            print('nop')
    '\n    Integrate a Laguerre series.\n\n    Returns the Laguerre series coefficients `c` integrated `m` times from\n    `lbnd` along `axis`. At each iteration the resulting series is\n    **multiplied** by `scl` and an integration constant, `k`, is added.\n    The scaling factor is for use in a linear change of variable.  ("Buyer\n    beware": note that, depending on what one is doing, one may want `scl`\n    to be the reciprocal of what one might expect; for more information,\n    see the Notes section below.)  The argument `c` is an array of\n    coefficients from low to high degree along each axis, e.g., [1,2,3]\n    represents the series ``L_0 + 2*L_1 + 3*L_2`` while [[1,2],[1,2]]\n    represents ``1*L_0(x)*L_0(y) + 1*L_1(x)*L_0(y) + 2*L_0(x)*L_1(y) +\n    2*L_1(x)*L_1(y)`` if axis=0 is ``x`` and axis=1 is ``y``.\n\n\n    Parameters\n    ----------\n    c : array_like\n        Array of Laguerre series coefficients. If `c` is multidimensional\n        the different axis correspond to different variables with the\n        degree in each axis given by the corresponding index.\n    m : int, optional\n        Order of integration, must be positive. (Default: 1)\n    k : {[], list, scalar}, optional\n        Integration constant(s).  The value of the first integral at\n        ``lbnd`` is the first value in the list, the value of the second\n        integral at ``lbnd`` is the second value, etc.  If ``k == []`` (the\n        default), all constants are set to zero.  If ``m == 1``, a single\n        scalar can be given instead of a list.\n    lbnd : scalar, optional\n        The lower bound of the integral. (Default: 0)\n    scl : scalar, optional\n        Following each integration the result is *multiplied* by `scl`\n        before the integration constant is added. (Default: 1)\n    axis : int, optional\n        Axis over which the integral is taken. (Default: 0).\n\n        .. versionadded:: 1.7.0\n\n    Returns\n    -------\n    S : ndarray\n        Laguerre series coefficients of the integral.\n\n    Raises\n    ------\n    ValueError\n        If ``m < 0``, ``len(k) > m``, ``np.ndim(lbnd) != 0``, or\n        ``np.ndim(scl) != 0``.\n\n    See Also\n    --------\n    lagder\n\n    Notes\n    -----\n    Note that the result of each integration is *multiplied* by `scl`.\n    Why is this important to note?  Say one is making a linear change of\n    variable :math:`u = ax + b` in an integral relative to `x`.  Then\n    :math:`dx = du/a`, so one will need to set `scl` equal to\n    :math:`1/a` - perhaps not what one would have first thought.\n\n    Also note that, in general, the result of integrating a C-series needs\n    to be "reprojected" onto the C-series basis set.  Thus, typically,\n    the result of this function is "unintuitive," albeit correct; see\n    Examples section below.\n\n    Examples\n    --------\n    >>> from numpy.polynomial.laguerre import lagint\n    >>> lagint([1,2,3])\n    array([ 1.,  1.,  1., -3.])\n    >>> lagint([1,2,3], m=2)\n    array([ 1.,  0.,  0., -4.,  3.])\n    >>> lagint([1,2,3], k=1)\n    array([ 2.,  1.,  1., -3.])\n    >>> lagint([1,2,3], lbnd=-1)\n    array([11.5,  1. ,  1. , -3. ])\n    >>> lagint([1,2], m=2, k=[1,2], lbnd=-1)\n    array([ 11.16666667,  -5.        ,  -3.        ,   2.        ]) # may vary\n\n    '
    c = np.array(c, ndmin=1, copy=True)
    if c.dtype.char in '?bBhHiIlLqQpP':
        c = c.astype(np.double)
    if not np.iterable(k):
        k = [k]
    cnt = pu._as_int(m, 'the order of integration')
    iaxis = pu._as_int(axis, 'the axis')
    if cnt < 0:
        raise ValueError('The order of integration must be non-negative')
    if len(k) > cnt:
        raise ValueError('Too many integration constants')
    if np.ndim(lbnd) != 0:
        raise ValueError('lbnd must be a scalar.')
    if np.ndim(scl) != 0:
        raise ValueError('scl must be a scalar.')
    iaxis = normalize_axis_index(iaxis, c.ndim)
    if cnt == 0:
        return c
    c = np.moveaxis(c, iaxis, 0)
    k = list(k) + [0] * (cnt - len(k))
    for i in range(cnt):
        n = len(c)
        c *= scl
        if n == 1 and np.all(c[0] == 0):
            c[0] += k[i]
        else:
            tmp = np.empty((n + 1,) + c.shape[1:], dtype=c.dtype)
            tmp[0] = c[0]
            tmp[1] = -c[0]
            for j in range(1, n):
                tmp[j] += c[j]
                tmp[j + 1] = -c[j]
            tmp[0] += k[i] - lagval(lbnd, tmp)
            c = tmp
    c = np.moveaxis(c, 0, iaxis)
    return c

def lagval(x, c, tensor=True):
    if False:
        return 10
    '\n    Evaluate a Laguerre series at points x.\n\n    If `c` is of length `n + 1`, this function returns the value:\n\n    .. math:: p(x) = c_0 * L_0(x) + c_1 * L_1(x) + ... + c_n * L_n(x)\n\n    The parameter `x` is converted to an array only if it is a tuple or a\n    list, otherwise it is treated as a scalar. In either case, either `x`\n    or its elements must support multiplication and addition both with\n    themselves and with the elements of `c`.\n\n    If `c` is a 1-D array, then `p(x)` will have the same shape as `x`.  If\n    `c` is multidimensional, then the shape of the result depends on the\n    value of `tensor`. If `tensor` is true the shape will be c.shape[1:] +\n    x.shape. If `tensor` is false the shape will be c.shape[1:]. Note that\n    scalars have shape (,).\n\n    Trailing zeros in the coefficients will be used in the evaluation, so\n    they should be avoided if efficiency is a concern.\n\n    Parameters\n    ----------\n    x : array_like, compatible object\n        If `x` is a list or tuple, it is converted to an ndarray, otherwise\n        it is left unchanged and treated as a scalar. In either case, `x`\n        or its elements must support addition and multiplication with\n        themselves and with the elements of `c`.\n    c : array_like\n        Array of coefficients ordered so that the coefficients for terms of\n        degree n are contained in c[n]. If `c` is multidimensional the\n        remaining indices enumerate multiple polynomials. In the two\n        dimensional case the coefficients may be thought of as stored in\n        the columns of `c`.\n    tensor : boolean, optional\n        If True, the shape of the coefficient array is extended with ones\n        on the right, one for each dimension of `x`. Scalars have dimension 0\n        for this action. The result is that every column of coefficients in\n        `c` is evaluated for every element of `x`. If False, `x` is broadcast\n        over the columns of `c` for the evaluation.  This keyword is useful\n        when `c` is multidimensional. The default value is True.\n\n        .. versionadded:: 1.7.0\n\n    Returns\n    -------\n    values : ndarray, algebra_like\n        The shape of the return value is described above.\n\n    See Also\n    --------\n    lagval2d, laggrid2d, lagval3d, laggrid3d\n\n    Notes\n    -----\n    The evaluation uses Clenshaw recursion, aka synthetic division.\n\n    Examples\n    --------\n    >>> from numpy.polynomial.laguerre import lagval\n    >>> coef = [1,2,3]\n    >>> lagval(1, coef)\n    -0.5\n    >>> lagval([[1,2],[3,4]], coef)\n    array([[-0.5, -4. ],\n           [-4.5, -2. ]])\n\n    '
    c = np.array(c, ndmin=1, copy=False)
    if c.dtype.char in '?bBhHiIlLqQpP':
        c = c.astype(np.double)
    if isinstance(x, (tuple, list)):
        x = np.asarray(x)
    if isinstance(x, np.ndarray) and tensor:
        c = c.reshape(c.shape + (1,) * x.ndim)
    if len(c) == 1:
        c0 = c[0]
        c1 = 0
    elif len(c) == 2:
        c0 = c[0]
        c1 = c[1]
    else:
        nd = len(c)
        c0 = c[-2]
        c1 = c[-1]
        for i in range(3, len(c) + 1):
            tmp = c0
            nd = nd - 1
            c0 = c[-i] - c1 * (nd - 1) / nd
            c1 = tmp + c1 * (2 * nd - 1 - x) / nd
    return c0 + c1 * (1 - x)

def lagval2d(x, y, c):
    if False:
        print('Hello World!')
    "\n    Evaluate a 2-D Laguerre series at points (x, y).\n\n    This function returns the values:\n\n    .. math:: p(x,y) = \\sum_{i,j} c_{i,j} * L_i(x) * L_j(y)\n\n    The parameters `x` and `y` are converted to arrays only if they are\n    tuples or a lists, otherwise they are treated as a scalars and they\n    must have the same shape after conversion. In either case, either `x`\n    and `y` or their elements must support multiplication and addition both\n    with themselves and with the elements of `c`.\n\n    If `c` is a 1-D array a one is implicitly appended to its shape to make\n    it 2-D. The shape of the result will be c.shape[2:] + x.shape.\n\n    Parameters\n    ----------\n    x, y : array_like, compatible objects\n        The two dimensional series is evaluated at the points `(x, y)`,\n        where `x` and `y` must have the same shape. If `x` or `y` is a list\n        or tuple, it is first converted to an ndarray, otherwise it is left\n        unchanged and if it isn't an ndarray it is treated as a scalar.\n    c : array_like\n        Array of coefficients ordered so that the coefficient of the term\n        of multi-degree i,j is contained in ``c[i,j]``. If `c` has\n        dimension greater than two the remaining indices enumerate multiple\n        sets of coefficients.\n\n    Returns\n    -------\n    values : ndarray, compatible object\n        The values of the two dimensional polynomial at points formed with\n        pairs of corresponding values from `x` and `y`.\n\n    See Also\n    --------\n    lagval, laggrid2d, lagval3d, laggrid3d\n\n    Notes\n    -----\n\n    .. versionadded:: 1.7.0\n\n    "
    return pu._valnd(lagval, c, x, y)

def laggrid2d(x, y, c):
    if False:
        print('Hello World!')
    "\n    Evaluate a 2-D Laguerre series on the Cartesian product of x and y.\n\n    This function returns the values:\n\n    .. math:: p(a,b) = \\sum_{i,j} c_{i,j} * L_i(a) * L_j(b)\n\n    where the points `(a, b)` consist of all pairs formed by taking\n    `a` from `x` and `b` from `y`. The resulting points form a grid with\n    `x` in the first dimension and `y` in the second.\n\n    The parameters `x` and `y` are converted to arrays only if they are\n    tuples or a lists, otherwise they are treated as a scalars. In either\n    case, either `x` and `y` or their elements must support multiplication\n    and addition both with themselves and with the elements of `c`.\n\n    If `c` has fewer than two dimensions, ones are implicitly appended to\n    its shape to make it 2-D. The shape of the result will be c.shape[2:] +\n    x.shape + y.shape.\n\n    Parameters\n    ----------\n    x, y : array_like, compatible objects\n        The two dimensional series is evaluated at the points in the\n        Cartesian product of `x` and `y`.  If `x` or `y` is a list or\n        tuple, it is first converted to an ndarray, otherwise it is left\n        unchanged and, if it isn't an ndarray, it is treated as a scalar.\n    c : array_like\n        Array of coefficients ordered so that the coefficient of the term of\n        multi-degree i,j is contained in `c[i,j]`. If `c` has dimension\n        greater than two the remaining indices enumerate multiple sets of\n        coefficients.\n\n    Returns\n    -------\n    values : ndarray, compatible object\n        The values of the two dimensional Chebyshev series at points in the\n        Cartesian product of `x` and `y`.\n\n    See Also\n    --------\n    lagval, lagval2d, lagval3d, laggrid3d\n\n    Notes\n    -----\n\n    .. versionadded:: 1.7.0\n\n    "
    return pu._gridnd(lagval, c, x, y)

def lagval3d(x, y, z, c):
    if False:
        print('Hello World!')
    "\n    Evaluate a 3-D Laguerre series at points (x, y, z).\n\n    This function returns the values:\n\n    .. math:: p(x,y,z) = \\sum_{i,j,k} c_{i,j,k} * L_i(x) * L_j(y) * L_k(z)\n\n    The parameters `x`, `y`, and `z` are converted to arrays only if\n    they are tuples or a lists, otherwise they are treated as a scalars and\n    they must have the same shape after conversion. In either case, either\n    `x`, `y`, and `z` or their elements must support multiplication and\n    addition both with themselves and with the elements of `c`.\n\n    If `c` has fewer than 3 dimensions, ones are implicitly appended to its\n    shape to make it 3-D. The shape of the result will be c.shape[3:] +\n    x.shape.\n\n    Parameters\n    ----------\n    x, y, z : array_like, compatible object\n        The three dimensional series is evaluated at the points\n        `(x, y, z)`, where `x`, `y`, and `z` must have the same shape.  If\n        any of `x`, `y`, or `z` is a list or tuple, it is first converted\n        to an ndarray, otherwise it is left unchanged and if it isn't an\n        ndarray it is  treated as a scalar.\n    c : array_like\n        Array of coefficients ordered so that the coefficient of the term of\n        multi-degree i,j,k is contained in ``c[i,j,k]``. If `c` has dimension\n        greater than 3 the remaining indices enumerate multiple sets of\n        coefficients.\n\n    Returns\n    -------\n    values : ndarray, compatible object\n        The values of the multidimensional polynomial on points formed with\n        triples of corresponding values from `x`, `y`, and `z`.\n\n    See Also\n    --------\n    lagval, lagval2d, laggrid2d, laggrid3d\n\n    Notes\n    -----\n\n    .. versionadded:: 1.7.0\n\n    "
    return pu._valnd(lagval, c, x, y, z)

def laggrid3d(x, y, z, c):
    if False:
        return 10
    "\n    Evaluate a 3-D Laguerre series on the Cartesian product of x, y, and z.\n\n    This function returns the values:\n\n    .. math:: p(a,b,c) = \\sum_{i,j,k} c_{i,j,k} * L_i(a) * L_j(b) * L_k(c)\n\n    where the points `(a, b, c)` consist of all triples formed by taking\n    `a` from `x`, `b` from `y`, and `c` from `z`. The resulting points form\n    a grid with `x` in the first dimension, `y` in the second, and `z` in\n    the third.\n\n    The parameters `x`, `y`, and `z` are converted to arrays only if they\n    are tuples or a lists, otherwise they are treated as a scalars. In\n    either case, either `x`, `y`, and `z` or their elements must support\n    multiplication and addition both with themselves and with the elements\n    of `c`.\n\n    If `c` has fewer than three dimensions, ones are implicitly appended to\n    its shape to make it 3-D. The shape of the result will be c.shape[3:] +\n    x.shape + y.shape + z.shape.\n\n    Parameters\n    ----------\n    x, y, z : array_like, compatible objects\n        The three dimensional series is evaluated at the points in the\n        Cartesian product of `x`, `y`, and `z`.  If `x`, `y`, or `z` is a\n        list or tuple, it is first converted to an ndarray, otherwise it is\n        left unchanged and, if it isn't an ndarray, it is treated as a\n        scalar.\n    c : array_like\n        Array of coefficients ordered so that the coefficients for terms of\n        degree i,j are contained in ``c[i,j]``. If `c` has dimension\n        greater than two the remaining indices enumerate multiple sets of\n        coefficients.\n\n    Returns\n    -------\n    values : ndarray, compatible object\n        The values of the two dimensional polynomial at points in the Cartesian\n        product of `x` and `y`.\n\n    See Also\n    --------\n    lagval, lagval2d, laggrid2d, lagval3d\n\n    Notes\n    -----\n\n    .. versionadded:: 1.7.0\n\n    "
    return pu._gridnd(lagval, c, x, y, z)

def lagvander(x, deg):
    if False:
        while True:
            i = 10
    'Pseudo-Vandermonde matrix of given degree.\n\n    Returns the pseudo-Vandermonde matrix of degree `deg` and sample points\n    `x`. The pseudo-Vandermonde matrix is defined by\n\n    .. math:: V[..., i] = L_i(x)\n\n    where `0 <= i <= deg`. The leading indices of `V` index the elements of\n    `x` and the last index is the degree of the Laguerre polynomial.\n\n    If `c` is a 1-D array of coefficients of length `n + 1` and `V` is the\n    array ``V = lagvander(x, n)``, then ``np.dot(V, c)`` and\n    ``lagval(x, c)`` are the same up to roundoff. This equivalence is\n    useful both for least squares fitting and for the evaluation of a large\n    number of Laguerre series of the same degree and sample points.\n\n    Parameters\n    ----------\n    x : array_like\n        Array of points. The dtype is converted to float64 or complex128\n        depending on whether any of the elements are complex. If `x` is\n        scalar it is converted to a 1-D array.\n    deg : int\n        Degree of the resulting matrix.\n\n    Returns\n    -------\n    vander : ndarray\n        The pseudo-Vandermonde matrix. The shape of the returned matrix is\n        ``x.shape + (deg + 1,)``, where The last index is the degree of the\n        corresponding Laguerre polynomial.  The dtype will be the same as\n        the converted `x`.\n\n    Examples\n    --------\n    >>> from numpy.polynomial.laguerre import lagvander\n    >>> x = np.array([0, 1, 2])\n    >>> lagvander(x, 3)\n    array([[ 1.        ,  1.        ,  1.        ,  1.        ],\n           [ 1.        ,  0.        , -0.5       , -0.66666667],\n           [ 1.        , -1.        , -1.        , -0.33333333]])\n\n    '
    ideg = pu._as_int(deg, 'deg')
    if ideg < 0:
        raise ValueError('deg must be non-negative')
    x = np.array(x, copy=False, ndmin=1) + 0.0
    dims = (ideg + 1,) + x.shape
    dtyp = x.dtype
    v = np.empty(dims, dtype=dtyp)
    v[0] = x * 0 + 1
    if ideg > 0:
        v[1] = 1 - x
        for i in range(2, ideg + 1):
            v[i] = (v[i - 1] * (2 * i - 1 - x) - v[i - 2] * (i - 1)) / i
    return np.moveaxis(v, 0, -1)

def lagvander2d(x, y, deg):
    if False:
        print('Hello World!')
    'Pseudo-Vandermonde matrix of given degrees.\n\n    Returns the pseudo-Vandermonde matrix of degrees `deg` and sample\n    points `(x, y)`. The pseudo-Vandermonde matrix is defined by\n\n    .. math:: V[..., (deg[1] + 1)*i + j] = L_i(x) * L_j(y),\n\n    where `0 <= i <= deg[0]` and `0 <= j <= deg[1]`. The leading indices of\n    `V` index the points `(x, y)` and the last index encodes the degrees of\n    the Laguerre polynomials.\n\n    If ``V = lagvander2d(x, y, [xdeg, ydeg])``, then the columns of `V`\n    correspond to the elements of a 2-D coefficient array `c` of shape\n    (xdeg + 1, ydeg + 1) in the order\n\n    .. math:: c_{00}, c_{01}, c_{02} ... , c_{10}, c_{11}, c_{12} ...\n\n    and ``np.dot(V, c.flat)`` and ``lagval2d(x, y, c)`` will be the same\n    up to roundoff. This equivalence is useful both for least squares\n    fitting and for the evaluation of a large number of 2-D Laguerre\n    series of the same degrees and sample points.\n\n    Parameters\n    ----------\n    x, y : array_like\n        Arrays of point coordinates, all of the same shape. The dtypes\n        will be converted to either float64 or complex128 depending on\n        whether any of the elements are complex. Scalars are converted to\n        1-D arrays.\n    deg : list of ints\n        List of maximum degrees of the form [x_deg, y_deg].\n\n    Returns\n    -------\n    vander2d : ndarray\n        The shape of the returned matrix is ``x.shape + (order,)``, where\n        :math:`order = (deg[0]+1)*(deg[1]+1)`.  The dtype will be the same\n        as the converted `x` and `y`.\n\n    See Also\n    --------\n    lagvander, lagvander3d, lagval2d, lagval3d\n\n    Notes\n    -----\n\n    .. versionadded:: 1.7.0\n\n    '
    return pu._vander_nd_flat((lagvander, lagvander), (x, y), deg)

def lagvander3d(x, y, z, deg):
    if False:
        for i in range(10):
            print('nop')
    'Pseudo-Vandermonde matrix of given degrees.\n\n    Returns the pseudo-Vandermonde matrix of degrees `deg` and sample\n    points `(x, y, z)`. If `l, m, n` are the given degrees in `x, y, z`,\n    then The pseudo-Vandermonde matrix is defined by\n\n    .. math:: V[..., (m+1)(n+1)i + (n+1)j + k] = L_i(x)*L_j(y)*L_k(z),\n\n    where `0 <= i <= l`, `0 <= j <= m`, and `0 <= j <= n`.  The leading\n    indices of `V` index the points `(x, y, z)` and the last index encodes\n    the degrees of the Laguerre polynomials.\n\n    If ``V = lagvander3d(x, y, z, [xdeg, ydeg, zdeg])``, then the columns\n    of `V` correspond to the elements of a 3-D coefficient array `c` of\n    shape (xdeg + 1, ydeg + 1, zdeg + 1) in the order\n\n    .. math:: c_{000}, c_{001}, c_{002},... , c_{010}, c_{011}, c_{012},...\n\n    and  ``np.dot(V, c.flat)`` and ``lagval3d(x, y, z, c)`` will be the\n    same up to roundoff. This equivalence is useful both for least squares\n    fitting and for the evaluation of a large number of 3-D Laguerre\n    series of the same degrees and sample points.\n\n    Parameters\n    ----------\n    x, y, z : array_like\n        Arrays of point coordinates, all of the same shape. The dtypes will\n        be converted to either float64 or complex128 depending on whether\n        any of the elements are complex. Scalars are converted to 1-D\n        arrays.\n    deg : list of ints\n        List of maximum degrees of the form [x_deg, y_deg, z_deg].\n\n    Returns\n    -------\n    vander3d : ndarray\n        The shape of the returned matrix is ``x.shape + (order,)``, where\n        :math:`order = (deg[0]+1)*(deg[1]+1)*(deg[2]+1)`.  The dtype will\n        be the same as the converted `x`, `y`, and `z`.\n\n    See Also\n    --------\n    lagvander, lagvander3d, lagval2d, lagval3d\n\n    Notes\n    -----\n\n    .. versionadded:: 1.7.0\n\n    '
    return pu._vander_nd_flat((lagvander, lagvander, lagvander), (x, y, z), deg)

def lagfit(x, y, deg, rcond=None, full=False, w=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Least squares fit of Laguerre series to data.\n\n    Return the coefficients of a Laguerre series of degree `deg` that is the\n    least squares fit to the data values `y` given at points `x`. If `y` is\n    1-D the returned coefficients will also be 1-D. If `y` is 2-D multiple\n    fits are done, one for each column of `y`, and the resulting\n    coefficients are stored in the corresponding columns of a 2-D return.\n    The fitted polynomial(s) are in the form\n\n    .. math::  p(x) = c_0 + c_1 * L_1(x) + ... + c_n * L_n(x),\n\n    where ``n`` is `deg`.\n\n    Parameters\n    ----------\n    x : array_like, shape (M,)\n        x-coordinates of the M sample points ``(x[i], y[i])``.\n    y : array_like, shape (M,) or (M, K)\n        y-coordinates of the sample points. Several data sets of sample\n        points sharing the same x-coordinates can be fitted at once by\n        passing in a 2D-array that contains one dataset per column.\n    deg : int or 1-D array_like\n        Degree(s) of the fitting polynomials. If `deg` is a single integer\n        all terms up to and including the `deg`\'th term are included in the\n        fit. For NumPy versions >= 1.11.0 a list of integers specifying the\n        degrees of the terms to include may be used instead.\n    rcond : float, optional\n        Relative condition number of the fit. Singular values smaller than\n        this relative to the largest singular value will be ignored. The\n        default value is len(x)*eps, where eps is the relative precision of\n        the float type, about 2e-16 in most cases.\n    full : bool, optional\n        Switch determining nature of return value. When it is False (the\n        default) just the coefficients are returned, when True diagnostic\n        information from the singular value decomposition is also returned.\n    w : array_like, shape (`M`,), optional\n        Weights. If not None, the weight ``w[i]`` applies to the unsquared\n        residual ``y[i] - y_hat[i]`` at ``x[i]``. Ideally the weights are\n        chosen so that the errors of the products ``w[i]*y[i]`` all have the\n        same variance.  When using inverse-variance weighting, use\n        ``w[i] = 1/sigma(y[i])``.  The default value is None.\n\n    Returns\n    -------\n    coef : ndarray, shape (M,) or (M, K)\n        Laguerre coefficients ordered from low to high. If `y` was 2-D,\n        the coefficients for the data in column *k*  of `y` are in column\n        *k*.\n\n    [residuals, rank, singular_values, rcond] : list\n        These values are only returned if ``full == True``\n\n        - residuals -- sum of squared residuals of the least squares fit\n        - rank -- the numerical rank of the scaled Vandermonde matrix\n        - singular_values -- singular values of the scaled Vandermonde matrix\n        - rcond -- value of `rcond`.\n\n        For more details, see `numpy.linalg.lstsq`.\n\n    Warns\n    -----\n    RankWarning\n        The rank of the coefficient matrix in the least-squares fit is\n        deficient. The warning is only raised if ``full == False``.  The\n        warnings can be turned off by\n\n        >>> import warnings\n        >>> warnings.simplefilter(\'ignore\', np.exceptions.RankWarning)\n\n    See Also\n    --------\n    numpy.polynomial.polynomial.polyfit\n    numpy.polynomial.legendre.legfit\n    numpy.polynomial.chebyshev.chebfit\n    numpy.polynomial.hermite.hermfit\n    numpy.polynomial.hermite_e.hermefit\n    lagval : Evaluates a Laguerre series.\n    lagvander : pseudo Vandermonde matrix of Laguerre series.\n    lagweight : Laguerre weight function.\n    numpy.linalg.lstsq : Computes a least-squares fit from the matrix.\n    scipy.interpolate.UnivariateSpline : Computes spline fits.\n\n    Notes\n    -----\n    The solution is the coefficients of the Laguerre series ``p`` that\n    minimizes the sum of the weighted squared errors\n\n    .. math:: E = \\sum_j w_j^2 * |y_j - p(x_j)|^2,\n\n    where the :math:`w_j` are the weights. This problem is solved by\n    setting up as the (typically) overdetermined matrix equation\n\n    .. math:: V(x) * c = w * y,\n\n    where ``V`` is the weighted pseudo Vandermonde matrix of `x`, ``c`` are the\n    coefficients to be solved for, `w` are the weights, and `y` are the\n    observed values.  This equation is then solved using the singular value\n    decomposition of ``V``.\n\n    If some of the singular values of `V` are so small that they are\n    neglected, then a `RankWarning` will be issued. This means that the\n    coefficient values may be poorly determined. Using a lower order fit\n    will usually get rid of the warning.  The `rcond` parameter can also be\n    set to a value smaller than its default, but the resulting fit may be\n    spurious and have large contributions from roundoff error.\n\n    Fits using Laguerre series are probably most useful when the data can\n    be approximated by ``sqrt(w(x)) * p(x)``, where ``w(x)`` is the Laguerre\n    weight. In that case the weight ``sqrt(w(x[i]))`` should be used\n    together with data values ``y[i]/sqrt(w(x[i]))``. The weight function is\n    available as `lagweight`.\n\n    References\n    ----------\n    .. [1] Wikipedia, "Curve fitting",\n           https://en.wikipedia.org/wiki/Curve_fitting\n\n    Examples\n    --------\n    >>> from numpy.polynomial.laguerre import lagfit, lagval\n    >>> x = np.linspace(0, 10)\n    >>> err = np.random.randn(len(x))/10\n    >>> y = lagval(x, [1, 2, 3]) + err\n    >>> lagfit(x, y, 2)\n    array([ 0.96971004,  2.00193749,  3.00288744]) # may vary\n\n    '
    return pu._fit(lagvander, x, y, deg, rcond, full, w)

def lagcompanion(c):
    if False:
        print('Hello World!')
    '\n    Return the companion matrix of c.\n\n    The usual companion matrix of the Laguerre polynomials is already\n    symmetric when `c` is a basis Laguerre polynomial, so no scaling is\n    applied.\n\n    Parameters\n    ----------\n    c : array_like\n        1-D array of Laguerre series coefficients ordered from low to high\n        degree.\n\n    Returns\n    -------\n    mat : ndarray\n        Companion matrix of dimensions (deg, deg).\n\n    Notes\n    -----\n\n    .. versionadded:: 1.7.0\n\n    '
    [c] = pu.as_series([c])
    if len(c) < 2:
        raise ValueError('Series must have maximum degree of at least 1.')
    if len(c) == 2:
        return np.array([[1 + c[0] / c[1]]])
    n = len(c) - 1
    mat = np.zeros((n, n), dtype=c.dtype)
    top = mat.reshape(-1)[1::n + 1]
    mid = mat.reshape(-1)[0::n + 1]
    bot = mat.reshape(-1)[n::n + 1]
    top[...] = -np.arange(1, n)
    mid[...] = 2.0 * np.arange(n) + 1.0
    bot[...] = top
    mat[:, -1] += c[:-1] / c[-1] * n
    return mat

def lagroots(c):
    if False:
        while True:
            i = 10
    '\n    Compute the roots of a Laguerre series.\n\n    Return the roots (a.k.a. "zeros") of the polynomial\n\n    .. math:: p(x) = \\sum_i c[i] * L_i(x).\n\n    Parameters\n    ----------\n    c : 1-D array_like\n        1-D array of coefficients.\n\n    Returns\n    -------\n    out : ndarray\n        Array of the roots of the series. If all the roots are real,\n        then `out` is also real, otherwise it is complex.\n\n    See Also\n    --------\n    numpy.polynomial.polynomial.polyroots\n    numpy.polynomial.legendre.legroots\n    numpy.polynomial.chebyshev.chebroots\n    numpy.polynomial.hermite.hermroots\n    numpy.polynomial.hermite_e.hermeroots\n\n    Notes\n    -----\n    The root estimates are obtained as the eigenvalues of the companion\n    matrix, Roots far from the origin of the complex plane may have large\n    errors due to the numerical instability of the series for such\n    values. Roots with multiplicity greater than 1 will also show larger\n    errors as the value of the series near such points is relatively\n    insensitive to errors in the roots. Isolated roots near the origin can\n    be improved by a few iterations of Newton\'s method.\n\n    The Laguerre series basis polynomials aren\'t powers of `x` so the\n    results of this function may seem unintuitive.\n\n    Examples\n    --------\n    >>> from numpy.polynomial.laguerre import lagroots, lagfromroots\n    >>> coef = lagfromroots([0, 1, 2])\n    >>> coef\n    array([  2.,  -8.,  12.,  -6.])\n    >>> lagroots(coef)\n    array([-4.4408921e-16,  1.0000000e+00,  2.0000000e+00])\n\n    '
    [c] = pu.as_series([c])
    if len(c) <= 1:
        return np.array([], dtype=c.dtype)
    if len(c) == 2:
        return np.array([1 + c[0] / c[1]])
    m = lagcompanion(c)[::-1, ::-1]
    r = la.eigvals(m)
    r.sort()
    return r

def laggauss(deg):
    if False:
        while True:
            i = 10
    "\n    Gauss-Laguerre quadrature.\n\n    Computes the sample points and weights for Gauss-Laguerre quadrature.\n    These sample points and weights will correctly integrate polynomials of\n    degree :math:`2*deg - 1` or less over the interval :math:`[0, \\inf]`\n    with the weight function :math:`f(x) = \\exp(-x)`.\n\n    Parameters\n    ----------\n    deg : int\n        Number of sample points and weights. It must be >= 1.\n\n    Returns\n    -------\n    x : ndarray\n        1-D ndarray containing the sample points.\n    y : ndarray\n        1-D ndarray containing the weights.\n\n    Notes\n    -----\n\n    .. versionadded:: 1.7.0\n\n    The results have only been tested up to degree 100 higher degrees may\n    be problematic. The weights are determined by using the fact that\n\n    .. math:: w_k = c / (L'_n(x_k) * L_{n-1}(x_k))\n\n    where :math:`c` is a constant independent of :math:`k` and :math:`x_k`\n    is the k'th root of :math:`L_n`, and then scaling the results to get\n    the right value when integrating 1.\n\n    "
    ideg = pu._as_int(deg, 'deg')
    if ideg <= 0:
        raise ValueError('deg must be a positive integer')
    c = np.array([0] * deg + [1])
    m = lagcompanion(c)
    x = la.eigvalsh(m)
    dy = lagval(x, c)
    df = lagval(x, lagder(c))
    x -= dy / df
    fm = lagval(x, c[1:])
    fm /= np.abs(fm).max()
    df /= np.abs(df).max()
    w = 1 / (fm * df)
    w /= w.sum()
    return (x, w)

def lagweight(x):
    if False:
        i = 10
        return i + 15
    'Weight function of the Laguerre polynomials.\n\n    The weight function is :math:`exp(-x)` and the interval of integration\n    is :math:`[0, \\inf]`. The Laguerre polynomials are orthogonal, but not\n    normalized, with respect to this weight function.\n\n    Parameters\n    ----------\n    x : array_like\n       Values at which the weight function will be computed.\n\n    Returns\n    -------\n    w : ndarray\n       The weight function at `x`.\n\n    Notes\n    -----\n\n    .. versionadded:: 1.7.0\n\n    '
    w = np.exp(-x)
    return w

class Laguerre(ABCPolyBase):
    """A Laguerre series class.

    The Laguerre class provides the standard Python numerical methods
    '+', '-', '*', '//', '%', 'divmod', '**', and '()' as well as the
    attributes and methods listed below.

    Parameters
    ----------
    coef : array_like
        Laguerre coefficients in order of increasing degree, i.e,
        ``(1, 2, 3)`` gives ``1*L_0(x) + 2*L_1(X) + 3*L_2(x)``.
    domain : (2,) array_like, optional
        Domain to use. The interval ``[domain[0], domain[1]]`` is mapped
        to the interval ``[window[0], window[1]]`` by shifting and scaling.
        The default value is [0, 1].
    window : (2,) array_like, optional
        Window, see `domain` for its use. The default value is [0, 1].

        .. versionadded:: 1.6.0
    symbol : str, optional
        Symbol used to represent the independent variable in string
        representations of the polynomial expression, e.g. for printing.
        The symbol must be a valid Python identifier. Default value is 'x'.

        .. versionadded:: 1.24

    """
    _add = staticmethod(lagadd)
    _sub = staticmethod(lagsub)
    _mul = staticmethod(lagmul)
    _div = staticmethod(lagdiv)
    _pow = staticmethod(lagpow)
    _val = staticmethod(lagval)
    _int = staticmethod(lagint)
    _der = staticmethod(lagder)
    _fit = staticmethod(lagfit)
    _line = staticmethod(lagline)
    _roots = staticmethod(lagroots)
    _fromroots = staticmethod(lagfromroots)
    domain = np.array(lagdomain)
    window = np.array(lagdomain)
    basis_name = 'L'