import numpy as np
from ._ufuncs import _ellip_harm
from ._ellip_harm_2 import _ellipsoid, _ellipsoid_norm

def ellip_harm(h2, k2, n, p, s, signm=1, signn=1):
    if False:
        for i in range(10):
            print('nop')
    '\n    Ellipsoidal harmonic functions E^p_n(l)\n\n    These are also known as Lame functions of the first kind, and are\n    solutions to the Lame equation:\n\n    .. math:: (s^2 - h^2)(s^2 - k^2)E\'\'(s) + s(2s^2 - h^2 - k^2)E\'(s) + (a - q s^2)E(s) = 0\n\n    where :math:`q = (n+1)n` and :math:`a` is the eigenvalue (not\n    returned) corresponding to the solutions.\n\n    Parameters\n    ----------\n    h2 : float\n        ``h**2``\n    k2 : float\n        ``k**2``; should be larger than ``h**2``\n    n : int\n        Degree\n    s : float\n        Coordinate\n    p : int\n        Order, can range between [1,2n+1]\n    signm : {1, -1}, optional\n        Sign of prefactor of functions. Can be +/-1. See Notes.\n    signn : {1, -1}, optional\n        Sign of prefactor of functions. Can be +/-1. See Notes.\n\n    Returns\n    -------\n    E : float\n        the harmonic :math:`E^p_n(s)`\n\n    See Also\n    --------\n    ellip_harm_2, ellip_normal\n\n    Notes\n    -----\n    The geometric interpretation of the ellipsoidal functions is\n    explained in [2]_, [3]_, [4]_. The `signm` and `signn` arguments control the\n    sign of prefactors for functions according to their type::\n\n        K : +1\n        L : signm\n        M : signn\n        N : signm*signn\n\n    .. versionadded:: 0.15.0\n\n    References\n    ----------\n    .. [1] Digital Library of Mathematical Functions 29.12\n       https://dlmf.nist.gov/29.12\n    .. [2] Bardhan and Knepley, "Computational science and\n       re-discovery: open-source implementations of\n       ellipsoidal harmonics for problems in potential theory",\n       Comput. Sci. Disc. 5, 014006 (2012)\n       :doi:`10.1088/1749-4699/5/1/014006`.\n    .. [3] David J.and Dechambre P, "Computation of Ellipsoidal\n       Gravity Field Harmonics for small solar system bodies"\n       pp. 30-36, 2000\n    .. [4] George Dassios, "Ellipsoidal Harmonics: Theory and Applications"\n       pp. 418, 2012\n\n    Examples\n    --------\n    >>> from scipy.special import ellip_harm\n    >>> w = ellip_harm(5,8,1,1,2.5)\n    >>> w\n    2.5\n\n    Check that the functions indeed are solutions to the Lame equation:\n\n    >>> import numpy as np\n    >>> from scipy.interpolate import UnivariateSpline\n    >>> def eigenvalue(f, df, ddf):\n    ...     r = ((s**2 - h**2)*(s**2 - k**2)*ddf + s*(2*s**2 - h**2 - k**2)*df - n*(n+1)*s**2*f)/f\n    ...     return -r.mean(), r.std()\n    >>> s = np.linspace(0.1, 10, 200)\n    >>> k, h, n, p = 8.0, 2.2, 3, 2\n    >>> E = ellip_harm(h**2, k**2, n, p, s)\n    >>> E_spl = UnivariateSpline(s, E)\n    >>> a, a_err = eigenvalue(E_spl(s), E_spl(s,1), E_spl(s,2))\n    >>> a, a_err\n    (583.44366156701483, 6.4580890640310646e-11)\n\n    '
    return _ellip_harm(h2, k2, n, p, s, signm, signn)
_ellip_harm_2_vec = np.vectorize(_ellipsoid, otypes='d')

def ellip_harm_2(h2, k2, n, p, s):
    if False:
        while True:
            i = 10
    "\n    Ellipsoidal harmonic functions F^p_n(l)\n\n    These are also known as Lame functions of the second kind, and are\n    solutions to the Lame equation:\n\n    .. math:: (s^2 - h^2)(s^2 - k^2)F''(s) + s(2s^2 - h^2 - k^2)F'(s) + (a - q s^2)F(s) = 0\n\n    where :math:`q = (n+1)n` and :math:`a` is the eigenvalue (not\n    returned) corresponding to the solutions.\n\n    Parameters\n    ----------\n    h2 : float\n        ``h**2``\n    k2 : float\n        ``k**2``; should be larger than ``h**2``\n    n : int\n        Degree.\n    p : int\n        Order, can range between [1,2n+1].\n    s : float\n        Coordinate\n\n    Returns\n    -------\n    F : float\n        The harmonic :math:`F^p_n(s)`\n\n    See Also\n    --------\n    ellip_harm, ellip_normal\n\n    Notes\n    -----\n    Lame functions of the second kind are related to the functions of the first kind:\n\n    .. math::\n\n       F^p_n(s)=(2n + 1)E^p_n(s)\\int_{0}^{1/s}\\frac{du}{(E^p_n(1/u))^2\\sqrt{(1-u^2k^2)(1-u^2h^2)}}\n\n    .. versionadded:: 0.15.0\n\n    Examples\n    --------\n    >>> from scipy.special import ellip_harm_2\n    >>> w = ellip_harm_2(5,8,2,1,10)\n    >>> w\n    0.00108056853382\n\n    "
    with np.errstate(all='ignore'):
        return _ellip_harm_2_vec(h2, k2, n, p, s)

def _ellip_normal_vec(h2, k2, n, p):
    if False:
        print('Hello World!')
    return _ellipsoid_norm(h2, k2, n, p)
_ellip_normal_vec = np.vectorize(_ellip_normal_vec, otypes='d')

def ellip_normal(h2, k2, n, p):
    if False:
        print('Hello World!')
    '\n    Ellipsoidal harmonic normalization constants gamma^p_n\n\n    The normalization constant is defined as\n\n    .. math::\n\n       \\gamma^p_n=8\\int_{0}^{h}dx\\int_{h}^{k}dy\\frac{(y^2-x^2)(E^p_n(y)E^p_n(x))^2}{\\sqrt((k^2-y^2)(y^2-h^2)(h^2-x^2)(k^2-x^2)}\n\n    Parameters\n    ----------\n    h2 : float\n        ``h**2``\n    k2 : float\n        ``k**2``; should be larger than ``h**2``\n    n : int\n        Degree.\n    p : int\n        Order, can range between [1,2n+1].\n\n    Returns\n    -------\n    gamma : float\n        The normalization constant :math:`\\gamma^p_n`\n\n    See Also\n    --------\n    ellip_harm, ellip_harm_2\n\n    Notes\n    -----\n    .. versionadded:: 0.15.0\n\n    Examples\n    --------\n    >>> from scipy.special import ellip_normal\n    >>> w = ellip_normal(5,8,3,7)\n    >>> w\n    1723.38796997\n\n    '
    with np.errstate(all='ignore'):
        return _ellip_normal_vec(h2, k2, n, p)