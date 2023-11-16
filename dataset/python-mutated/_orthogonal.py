"""
A collection of functions to find the weights and abscissas for
Gaussian Quadrature.

These calculations are done by finding the eigenvalues of a
tridiagonal matrix whose entries are dependent on the coefficients
in the recursion formula for the orthogonal polynomials with the
corresponding weighting function over the interval.

Many recursion relations for orthogonal polynomials are given:

.. math::

    a1n f_{n+1} (x) = (a2n + a3n x ) f_n (x) - a4n f_{n-1} (x)

The recursion relation of interest is

.. math::

    P_{n+1} (x) = (x - A_n) P_n (x) - B_n P_{n-1} (x)

where :math:`P` has a different normalization than :math:`f`.

The coefficients can be found as:

.. math::

    A_n = -a2n / a3n
    \\qquad
    B_n = ( a4n / a3n \\sqrt{h_n-1 / h_n})^2

where

.. math::

    h_n = \\int_a^b w(x) f_n(x)^2

assume:

.. math::

    P_0 (x) = 1
    \\qquad
    P_{-1} (x) == 0

For the mathematical background, see [golub.welsch-1969-mathcomp]_ and
[abramowitz.stegun-1965]_.

References
----------
.. [golub.welsch-1969-mathcomp]
   Golub, Gene H, and John H Welsch. 1969. Calculation of Gauss
   Quadrature Rules. *Mathematics of Computation* 23, 221-230+s1--s10.

.. [abramowitz.stegun-1965]
   Abramowitz, Milton, and Irene A Stegun. (1965) *Handbook of
   Mathematical Functions: with Formulas, Graphs, and Mathematical
   Tables*. Gaithersburg, MD: National Bureau of Standards.
   http://www.math.sfu.ca/~cbm/aands/

.. [townsend.trogdon.olver-2014]
   Townsend, A. and Trogdon, T. and Olver, S. (2014)
   *Fast computation of Gauss quadrature nodes and
   weights on the whole real line*. :arXiv:`1410.5286`.

.. [townsend.trogdon.olver-2015]
   Townsend, A. and Trogdon, T. and Olver, S. (2015)
   *Fast computation of Gauss quadrature nodes and
   weights on the whole real line*.
   IMA Journal of Numerical Analysis
   :doi:`10.1093/imanum/drv002`.
"""
import numpy as np
from numpy import exp, inf, pi, sqrt, floor, sin, cos, around, hstack, arccos, arange
from scipy import linalg
from scipy.special import airy
from . import _specfun
from . import _ufuncs
_gam = _ufuncs.gamma
_polyfuns = ['legendre', 'chebyt', 'chebyu', 'chebyc', 'chebys', 'jacobi', 'laguerre', 'genlaguerre', 'hermite', 'hermitenorm', 'gegenbauer', 'sh_legendre', 'sh_chebyt', 'sh_chebyu', 'sh_jacobi']
_rootfuns_map = {'roots_legendre': 'p_roots', 'roots_chebyt': 't_roots', 'roots_chebyu': 'u_roots', 'roots_chebyc': 'c_roots', 'roots_chebys': 's_roots', 'roots_jacobi': 'j_roots', 'roots_laguerre': 'l_roots', 'roots_genlaguerre': 'la_roots', 'roots_hermite': 'h_roots', 'roots_hermitenorm': 'he_roots', 'roots_gegenbauer': 'cg_roots', 'roots_sh_legendre': 'ps_roots', 'roots_sh_chebyt': 'ts_roots', 'roots_sh_chebyu': 'us_roots', 'roots_sh_jacobi': 'js_roots'}
__all__ = _polyfuns + list(_rootfuns_map.keys())

class orthopoly1d(np.poly1d):

    def __init__(self, roots, weights=None, hn=1.0, kn=1.0, wfunc=None, limits=None, monic=False, eval_func=None):
        if False:
            i = 10
            return i + 15
        equiv_weights = [weights[k] / wfunc(roots[k]) for k in range(len(roots))]
        mu = sqrt(hn)
        if monic:
            evf = eval_func
            if evf:
                knn = kn

                def eval_func(x):
                    if False:
                        return 10
                    return evf(x) / knn
            mu = mu / abs(kn)
            kn = 1.0
        poly = np.poly1d(roots, r=True)
        np.poly1d.__init__(self, poly.coeffs * float(kn))
        self.weights = np.array(list(zip(roots, weights, equiv_weights)))
        self.weight_func = wfunc
        self.limits = limits
        self.normcoef = mu
        self._eval_func = eval_func

    def __call__(self, v):
        if False:
            print('Hello World!')
        if self._eval_func and (not isinstance(v, np.poly1d)):
            return self._eval_func(v)
        else:
            return np.poly1d.__call__(self, v)

    def _scale(self, p):
        if False:
            i = 10
            return i + 15
        if p == 1.0:
            return
        self._coeffs *= p
        evf = self._eval_func
        if evf:
            self._eval_func = lambda x: evf(x) * p
        self.normcoef *= p

def _gen_roots_and_weights(n, mu0, an_func, bn_func, f, df, symmetrize, mu):
    if False:
        while True:
            i = 10
    '[x,w] = gen_roots_and_weights(n,an_func,sqrt_bn_func,mu)\n\n    Returns the roots (x) of an nth order orthogonal polynomial,\n    and weights (w) to use in appropriate Gaussian quadrature with that\n    orthogonal polynomial.\n\n    The polynomials have the recurrence relation\n          P_n+1(x) = (x - A_n) P_n(x) - B_n P_n-1(x)\n\n    an_func(n)          should return A_n\n    sqrt_bn_func(n)     should return sqrt(B_n)\n    mu ( = h_0 )        is the integral of the weight over the orthogonal\n                        interval\n    '
    k = np.arange(n, dtype='d')
    c = np.zeros((2, n))
    c[0, 1:] = bn_func(k[1:])
    c[1, :] = an_func(k)
    x = linalg.eigvals_banded(c, overwrite_a_band=True)
    y = f(n, x)
    dy = df(n, x)
    x -= y / dy
    fm = f(n - 1, x)
    log_fm = np.log(np.abs(fm))
    log_dy = np.log(np.abs(dy))
    fm /= np.exp((log_fm.max() + log_fm.min()) / 2.0)
    dy /= np.exp((log_dy.max() + log_dy.min()) / 2.0)
    w = 1.0 / (fm * dy)
    if symmetrize:
        w = (w + w[::-1]) / 2
        x = (x - x[::-1]) / 2
    w *= mu0 / w.sum()
    if mu:
        return (x, w, mu0)
    else:
        return (x, w)

def roots_jacobi(n, alpha, beta, mu=False):
    if False:
        for i in range(10):
            print('nop')
    'Gauss-Jacobi quadrature.\n\n    Compute the sample points and weights for Gauss-Jacobi\n    quadrature. The sample points are the roots of the nth degree\n    Jacobi polynomial, :math:`P^{\\alpha, \\beta}_n(x)`. These sample\n    points and weights correctly integrate polynomials of degree\n    :math:`2n - 1` or less over the interval :math:`[-1, 1]` with\n    weight function :math:`w(x) = (1 - x)^{\\alpha} (1 +\n    x)^{\\beta}`. See 22.2.1 in [AS]_ for details.\n\n    Parameters\n    ----------\n    n : int\n        quadrature order\n    alpha : float\n        alpha must be > -1\n    beta : float\n        beta must be > -1\n    mu : bool, optional\n        If True, return the sum of the weights, optional.\n\n    Returns\n    -------\n    x : ndarray\n        Sample points\n    w : ndarray\n        Weights\n    mu : float\n        Sum of the weights\n\n    See Also\n    --------\n    scipy.integrate.quadrature\n    scipy.integrate.fixed_quad\n\n    References\n    ----------\n    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.\n        Handbook of Mathematical Functions with Formulas,\n        Graphs, and Mathematical Tables. New York: Dover, 1972.\n\n    '
    m = int(n)
    if n < 1 or n != m:
        raise ValueError('n must be a positive integer.')
    if alpha <= -1 or beta <= -1:
        raise ValueError('alpha and beta must be greater than -1.')
    if alpha == 0.0 and beta == 0.0:
        return roots_legendre(m, mu)
    if alpha == beta:
        return roots_gegenbauer(m, alpha + 0.5, mu)
    if alpha + beta <= 1000:
        mu0 = 2.0 ** (alpha + beta + 1) * _ufuncs.beta(alpha + 1, beta + 1)
    else:
        mu0 = np.exp((alpha + beta + 1) * np.log(2.0) + _ufuncs.betaln(alpha + 1, beta + 1))
    a = alpha
    b = beta
    if a + b == 0.0:

        def an_func(k):
            if False:
                for i in range(10):
                    print('nop')
            return np.where(k == 0, (b - a) / (2 + a + b), 0.0)
    else:

        def an_func(k):
            if False:
                return 10
            return np.where(k == 0, (b - a) / (2 + a + b), (b * b - a * a) / ((2.0 * k + a + b) * (2.0 * k + a + b + 2)))

    def bn_func(k):
        if False:
            for i in range(10):
                print('nop')
        return 2.0 / (2.0 * k + a + b) * np.sqrt((k + a) * (k + b) / (2 * k + a + b + 1)) * np.where(k == 1, 1.0, np.sqrt(k * (k + a + b) / (2.0 * k + a + b - 1)))

    def f(n, x):
        if False:
            return 10
        return _ufuncs.eval_jacobi(n, a, b, x)

    def df(n, x):
        if False:
            print('Hello World!')
        return 0.5 * (n + a + b + 1) * _ufuncs.eval_jacobi(n - 1, a + 1, b + 1, x)
    return _gen_roots_and_weights(m, mu0, an_func, bn_func, f, df, False, mu)

def jacobi(n, alpha, beta, monic=False):
    if False:
        i = 10
        return i + 15
    "Jacobi polynomial.\n\n    Defined to be the solution of\n\n    .. math::\n        (1 - x^2)\\frac{d^2}{dx^2}P_n^{(\\alpha, \\beta)}\n          + (\\beta - \\alpha - (\\alpha + \\beta + 2)x)\n            \\frac{d}{dx}P_n^{(\\alpha, \\beta)}\n          + n(n + \\alpha + \\beta + 1)P_n^{(\\alpha, \\beta)} = 0\n\n    for :math:`\\alpha, \\beta > -1`; :math:`P_n^{(\\alpha, \\beta)}` is a\n    polynomial of degree :math:`n`.\n\n    Parameters\n    ----------\n    n : int\n        Degree of the polynomial.\n    alpha : float\n        Parameter, must be greater than -1.\n    beta : float\n        Parameter, must be greater than -1.\n    monic : bool, optional\n        If `True`, scale the leading coefficient to be 1. Default is\n        `False`.\n\n    Returns\n    -------\n    P : orthopoly1d\n        Jacobi polynomial.\n\n    Notes\n    -----\n    For fixed :math:`\\alpha, \\beta`, the polynomials\n    :math:`P_n^{(\\alpha, \\beta)}` are orthogonal over :math:`[-1, 1]`\n    with weight function :math:`(1 - x)^\\alpha(1 + x)^\\beta`.\n\n    References\n    ----------\n    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.\n        Handbook of Mathematical Functions with Formulas,\n        Graphs, and Mathematical Tables. New York: Dover, 1972.\n\n    Examples\n    --------\n    The Jacobi polynomials satisfy the recurrence relation:\n\n    .. math::\n        P_n^{(\\alpha, \\beta-1)}(x) - P_n^{(\\alpha-1, \\beta)}(x)\n          = P_{n-1}^{(\\alpha, \\beta)}(x)\n\n    This can be verified, for example, for :math:`\\alpha = \\beta = 2`\n    and :math:`n = 1` over the interval :math:`[-1, 1]`:\n\n    >>> import numpy as np\n    >>> from scipy.special import jacobi\n    >>> x = np.arange(-1.0, 1.0, 0.01)\n    >>> np.allclose(jacobi(0, 2, 2)(x),\n    ...             jacobi(1, 2, 1)(x) - jacobi(1, 1, 2)(x))\n    True\n\n    Plot of the Jacobi polynomial :math:`P_5^{(\\alpha, -0.5)}` for\n    different values of :math:`\\alpha`:\n\n    >>> import matplotlib.pyplot as plt\n    >>> x = np.arange(-1.0, 1.0, 0.01)\n    >>> fig, ax = plt.subplots()\n    >>> ax.set_ylim(-2.0, 2.0)\n    >>> ax.set_title(r'Jacobi polynomials $P_5^{(\\alpha, -0.5)}$')\n    >>> for alpha in np.arange(0, 4, 1):\n    ...     ax.plot(x, jacobi(5, alpha, -0.5)(x), label=rf'$\\alpha={alpha}$')\n    >>> plt.legend(loc='best')\n    >>> plt.show()\n\n    "
    if n < 0:
        raise ValueError('n must be nonnegative.')

    def wfunc(x):
        if False:
            print('Hello World!')
        return (1 - x) ** alpha * (1 + x) ** beta
    if n == 0:
        return orthopoly1d([], [], 1.0, 1.0, wfunc, (-1, 1), monic, eval_func=np.ones_like)
    (x, w, mu) = roots_jacobi(n, alpha, beta, mu=True)
    ab1 = alpha + beta + 1.0
    hn = 2 ** ab1 / (2 * n + ab1) * _gam(n + alpha + 1)
    hn *= _gam(n + beta + 1.0) / _gam(n + 1) / _gam(n + ab1)
    kn = _gam(2 * n + ab1) / 2.0 ** n / _gam(n + 1) / _gam(n + ab1)
    p = orthopoly1d(x, w, hn, kn, wfunc, (-1, 1), monic, lambda x: _ufuncs.eval_jacobi(n, alpha, beta, x))
    return p

def roots_sh_jacobi(n, p1, q1, mu=False):
    if False:
        return 10
    'Gauss-Jacobi (shifted) quadrature.\n\n    Compute the sample points and weights for Gauss-Jacobi (shifted)\n    quadrature. The sample points are the roots of the nth degree\n    shifted Jacobi polynomial, :math:`G^{p,q}_n(x)`. These sample\n    points and weights correctly integrate polynomials of degree\n    :math:`2n - 1` or less over the interval :math:`[0, 1]` with\n    weight function :math:`w(x) = (1 - x)^{p-q} x^{q-1}`. See 22.2.2\n    in [AS]_ for details.\n\n    Parameters\n    ----------\n    n : int\n        quadrature order\n    p1 : float\n        (p1 - q1) must be > -1\n    q1 : float\n        q1 must be > 0\n    mu : bool, optional\n        If True, return the sum of the weights, optional.\n\n    Returns\n    -------\n    x : ndarray\n        Sample points\n    w : ndarray\n        Weights\n    mu : float\n        Sum of the weights\n\n    See Also\n    --------\n    scipy.integrate.quadrature\n    scipy.integrate.fixed_quad\n\n    References\n    ----------\n    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.\n        Handbook of Mathematical Functions with Formulas,\n        Graphs, and Mathematical Tables. New York: Dover, 1972.\n\n    '
    if p1 - q1 <= -1 or q1 <= 0:
        raise ValueError('(p - q) must be greater than -1, and q must be greater than 0.')
    (x, w, m) = roots_jacobi(n, p1 - q1, q1 - 1, True)
    x = (x + 1) / 2
    scale = 2.0 ** p1
    w /= scale
    m /= scale
    if mu:
        return (x, w, m)
    else:
        return (x, w)

def sh_jacobi(n, p, q, monic=False):
    if False:
        i = 10
        return i + 15
    'Shifted Jacobi polynomial.\n\n    Defined by\n\n    .. math::\n\n        G_n^{(p, q)}(x)\n          = \\binom{2n + p - 1}{n}^{-1}P_n^{(p - q, q - 1)}(2x - 1),\n\n    where :math:`P_n^{(\\cdot, \\cdot)}` is the nth Jacobi polynomial.\n\n    Parameters\n    ----------\n    n : int\n        Degree of the polynomial.\n    p : float\n        Parameter, must have :math:`p > q - 1`.\n    q : float\n        Parameter, must be greater than 0.\n    monic : bool, optional\n        If `True`, scale the leading coefficient to be 1. Default is\n        `False`.\n\n    Returns\n    -------\n    G : orthopoly1d\n        Shifted Jacobi polynomial.\n\n    Notes\n    -----\n    For fixed :math:`p, q`, the polynomials :math:`G_n^{(p, q)}` are\n    orthogonal over :math:`[0, 1]` with weight function :math:`(1 -\n    x)^{p - q}x^{q - 1}`.\n\n    '
    if n < 0:
        raise ValueError('n must be nonnegative.')

    def wfunc(x):
        if False:
            while True:
                i = 10
        return (1.0 - x) ** (p - q) * x ** (q - 1.0)
    if n == 0:
        return orthopoly1d([], [], 1.0, 1.0, wfunc, (-1, 1), monic, eval_func=np.ones_like)
    n1 = n
    (x, w) = roots_sh_jacobi(n1, p, q)
    hn = _gam(n + 1) * _gam(n + q) * _gam(n + p) * _gam(n + p - q + 1)
    hn /= (2 * n + p) * _gam(2 * n + p) ** 2
    kn = 1.0
    pp = orthopoly1d(x, w, hn, kn, wfunc=wfunc, limits=(0, 1), monic=monic, eval_func=lambda x: _ufuncs.eval_sh_jacobi(n, p, q, x))
    return pp

def roots_genlaguerre(n, alpha, mu=False):
    if False:
        i = 10
        return i + 15
    'Gauss-generalized Laguerre quadrature.\n\n    Compute the sample points and weights for Gauss-generalized\n    Laguerre quadrature. The sample points are the roots of the nth\n    degree generalized Laguerre polynomial, :math:`L^{\\alpha}_n(x)`.\n    These sample points and weights correctly integrate polynomials of\n    degree :math:`2n - 1` or less over the interval :math:`[0,\n    \\infty]` with weight function :math:`w(x) = x^{\\alpha}\n    e^{-x}`. See 22.3.9 in [AS]_ for details.\n\n    Parameters\n    ----------\n    n : int\n        quadrature order\n    alpha : float\n        alpha must be > -1\n    mu : bool, optional\n        If True, return the sum of the weights, optional.\n\n    Returns\n    -------\n    x : ndarray\n        Sample points\n    w : ndarray\n        Weights\n    mu : float\n        Sum of the weights\n\n    See Also\n    --------\n    scipy.integrate.quadrature\n    scipy.integrate.fixed_quad\n\n    References\n    ----------\n    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.\n        Handbook of Mathematical Functions with Formulas,\n        Graphs, and Mathematical Tables. New York: Dover, 1972.\n\n    '
    m = int(n)
    if n < 1 or n != m:
        raise ValueError('n must be a positive integer.')
    if alpha < -1:
        raise ValueError('alpha must be greater than -1.')
    mu0 = _ufuncs.gamma(alpha + 1)
    if m == 1:
        x = np.array([alpha + 1.0], 'd')
        w = np.array([mu0], 'd')
        if mu:
            return (x, w, mu0)
        else:
            return (x, w)

    def an_func(k):
        if False:
            while True:
                i = 10
        return 2 * k + alpha + 1

    def bn_func(k):
        if False:
            for i in range(10):
                print('nop')
        return -np.sqrt(k * (k + alpha))

    def f(n, x):
        if False:
            while True:
                i = 10
        return _ufuncs.eval_genlaguerre(n, alpha, x)

    def df(n, x):
        if False:
            i = 10
            return i + 15
        return (n * _ufuncs.eval_genlaguerre(n, alpha, x) - (n + alpha) * _ufuncs.eval_genlaguerre(n - 1, alpha, x)) / x
    return _gen_roots_and_weights(m, mu0, an_func, bn_func, f, df, False, mu)

def genlaguerre(n, alpha, monic=False):
    if False:
        return 10
    "Generalized (associated) Laguerre polynomial.\n\n    Defined to be the solution of\n\n    .. math::\n        x\\frac{d^2}{dx^2}L_n^{(\\alpha)}\n          + (\\alpha + 1 - x)\\frac{d}{dx}L_n^{(\\alpha)}\n          + nL_n^{(\\alpha)} = 0,\n\n    where :math:`\\alpha > -1`; :math:`L_n^{(\\alpha)}` is a polynomial\n    of degree :math:`n`.\n\n    Parameters\n    ----------\n    n : int\n        Degree of the polynomial.\n    alpha : float\n        Parameter, must be greater than -1.\n    monic : bool, optional\n        If `True`, scale the leading coefficient to be 1. Default is\n        `False`.\n\n    Returns\n    -------\n    L : orthopoly1d\n        Generalized Laguerre polynomial.\n\n    See Also\n    --------\n    laguerre : Laguerre polynomial.\n    hyp1f1 : confluent hypergeometric function\n\n    Notes\n    -----\n    For fixed :math:`\\alpha`, the polynomials :math:`L_n^{(\\alpha)}`\n    are orthogonal over :math:`[0, \\infty)` with weight function\n    :math:`e^{-x}x^\\alpha`.\n\n    The Laguerre polynomials are the special case where :math:`\\alpha\n    = 0`.\n\n    References\n    ----------\n    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.\n        Handbook of Mathematical Functions with Formulas,\n        Graphs, and Mathematical Tables. New York: Dover, 1972.\n\n    Examples\n    --------\n    The generalized Laguerre polynomials are closely related to the confluent\n    hypergeometric function :math:`{}_1F_1`:\n\n        .. math::\n            L_n^{(\\alpha)} = \\binom{n + \\alpha}{n} {}_1F_1(-n, \\alpha +1, x)\n\n    This can be verified, for example,  for :math:`n = \\alpha = 3` over the\n    interval :math:`[-1, 1]`:\n\n    >>> import numpy as np\n    >>> from scipy.special import binom\n    >>> from scipy.special import genlaguerre\n    >>> from scipy.special import hyp1f1\n    >>> x = np.arange(-1.0, 1.0, 0.01)\n    >>> np.allclose(genlaguerre(3, 3)(x), binom(6, 3) * hyp1f1(-3, 4, x))\n    True\n\n    This is the plot of the generalized Laguerre polynomials\n    :math:`L_3^{(\\alpha)}` for some values of :math:`\\alpha`:\n\n    >>> import matplotlib.pyplot as plt\n    >>> x = np.arange(-4.0, 12.0, 0.01)\n    >>> fig, ax = plt.subplots()\n    >>> ax.set_ylim(-5.0, 10.0)\n    >>> ax.set_title(r'Generalized Laguerre polynomials $L_3^{\\alpha}$')\n    >>> for alpha in np.arange(0, 5):\n    ...     ax.plot(x, genlaguerre(3, alpha)(x), label=rf'$L_3^{(alpha)}$')\n    >>> plt.legend(loc='best')\n    >>> plt.show()\n\n    "
    if alpha <= -1:
        raise ValueError('alpha must be > -1')
    if n < 0:
        raise ValueError('n must be nonnegative.')
    if n == 0:
        n1 = n + 1
    else:
        n1 = n
    (x, w) = roots_genlaguerre(n1, alpha)

    def wfunc(x):
        if False:
            for i in range(10):
                print('nop')
        return exp(-x) * x ** alpha
    if n == 0:
        (x, w) = ([], [])
    hn = _gam(n + alpha + 1) / _gam(n + 1)
    kn = (-1) ** n / _gam(n + 1)
    p = orthopoly1d(x, w, hn, kn, wfunc, (0, inf), monic, lambda x: _ufuncs.eval_genlaguerre(n, alpha, x))
    return p

def roots_laguerre(n, mu=False):
    if False:
        return 10
    'Gauss-Laguerre quadrature.\n\n    Compute the sample points and weights for Gauss-Laguerre\n    quadrature. The sample points are the roots of the nth degree\n    Laguerre polynomial, :math:`L_n(x)`. These sample points and\n    weights correctly integrate polynomials of degree :math:`2n - 1`\n    or less over the interval :math:`[0, \\infty]` with weight function\n    :math:`w(x) = e^{-x}`. See 22.2.13 in [AS]_ for details.\n\n    Parameters\n    ----------\n    n : int\n        quadrature order\n    mu : bool, optional\n        If True, return the sum of the weights, optional.\n\n    Returns\n    -------\n    x : ndarray\n        Sample points\n    w : ndarray\n        Weights\n    mu : float\n        Sum of the weights\n\n    See Also\n    --------\n    scipy.integrate.quadrature\n    scipy.integrate.fixed_quad\n    numpy.polynomial.laguerre.laggauss\n\n    References\n    ----------\n    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.\n        Handbook of Mathematical Functions with Formulas,\n        Graphs, and Mathematical Tables. New York: Dover, 1972.\n\n    '
    return roots_genlaguerre(n, 0.0, mu=mu)

def laguerre(n, monic=False):
    if False:
        i = 10
        return i + 15
    "Laguerre polynomial.\n\n    Defined to be the solution of\n\n    .. math::\n        x\\frac{d^2}{dx^2}L_n + (1 - x)\\frac{d}{dx}L_n + nL_n = 0;\n\n    :math:`L_n` is a polynomial of degree :math:`n`.\n\n    Parameters\n    ----------\n    n : int\n        Degree of the polynomial.\n    monic : bool, optional\n        If `True`, scale the leading coefficient to be 1. Default is\n        `False`.\n\n    Returns\n    -------\n    L : orthopoly1d\n        Laguerre Polynomial.\n\n    See Also\n    --------\n    genlaguerre : Generalized (associated) Laguerre polynomial.\n\n    Notes\n    -----\n    The polynomials :math:`L_n` are orthogonal over :math:`[0,\n    \\infty)` with weight function :math:`e^{-x}`.\n\n    References\n    ----------\n    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.\n        Handbook of Mathematical Functions with Formulas,\n        Graphs, and Mathematical Tables. New York: Dover, 1972.\n\n    Examples\n    --------\n    The Laguerre polynomials :math:`L_n` are the special case\n    :math:`\\alpha = 0` of the generalized Laguerre polynomials\n    :math:`L_n^{(\\alpha)}`.\n    Let's verify it on the interval :math:`[-1, 1]`:\n\n    >>> import numpy as np\n    >>> from scipy.special import genlaguerre\n    >>> from scipy.special import laguerre\n    >>> x = np.arange(-1.0, 1.0, 0.01)\n    >>> np.allclose(genlaguerre(3, 0)(x), laguerre(3)(x))\n    True\n\n    The polynomials :math:`L_n` also satisfy the recurrence relation:\n\n    .. math::\n        (n + 1)L_{n+1}(x) = (2n +1 -x)L_n(x) - nL_{n-1}(x)\n\n    This can be easily checked on :math:`[0, 1]` for :math:`n = 3`:\n\n    >>> x = np.arange(0.0, 1.0, 0.01)\n    >>> np.allclose(4 * laguerre(4)(x),\n    ...             (7 - x) * laguerre(3)(x) - 3 * laguerre(2)(x))\n    True\n\n    This is the plot of the first few Laguerre polynomials :math:`L_n`:\n\n    >>> import matplotlib.pyplot as plt\n    >>> x = np.arange(-1.0, 5.0, 0.01)\n    >>> fig, ax = plt.subplots()\n    >>> ax.set_ylim(-5.0, 5.0)\n    >>> ax.set_title(r'Laguerre polynomials $L_n$')\n    >>> for n in np.arange(0, 5):\n    ...     ax.plot(x, laguerre(n)(x), label=rf'$L_{n}$')\n    >>> plt.legend(loc='best')\n    >>> plt.show()\n\n    "
    if n < 0:
        raise ValueError('n must be nonnegative.')
    if n == 0:
        n1 = n + 1
    else:
        n1 = n
    (x, w) = roots_laguerre(n1)
    if n == 0:
        (x, w) = ([], [])
    hn = 1.0
    kn = (-1) ** n / _gam(n + 1)
    p = orthopoly1d(x, w, hn, kn, lambda x: exp(-x), (0, inf), monic, lambda x: _ufuncs.eval_laguerre(n, x))
    return p

def roots_hermite(n, mu=False):
    if False:
        i = 10
        return i + 15
    "Gauss-Hermite (physicist's) quadrature.\n\n    Compute the sample points and weights for Gauss-Hermite\n    quadrature. The sample points are the roots of the nth degree\n    Hermite polynomial, :math:`H_n(x)`. These sample points and\n    weights correctly integrate polynomials of degree :math:`2n - 1`\n    or less over the interval :math:`[-\\infty, \\infty]` with weight\n    function :math:`w(x) = e^{-x^2}`. See 22.2.14 in [AS]_ for\n    details.\n\n    Parameters\n    ----------\n    n : int\n        quadrature order\n    mu : bool, optional\n        If True, return the sum of the weights, optional.\n\n    Returns\n    -------\n    x : ndarray\n        Sample points\n    w : ndarray\n        Weights\n    mu : float\n        Sum of the weights\n\n    See Also\n    --------\n    scipy.integrate.quadrature\n    scipy.integrate.fixed_quad\n    numpy.polynomial.hermite.hermgauss\n    roots_hermitenorm\n\n    Notes\n    -----\n    For small n up to 150 a modified version of the Golub-Welsch\n    algorithm is used. Nodes are computed from the eigenvalue\n    problem and improved by one step of a Newton iteration.\n    The weights are computed from the well-known analytical formula.\n\n    For n larger than 150 an optimal asymptotic algorithm is applied\n    which computes nodes and weights in a numerically stable manner.\n    The algorithm has linear runtime making computation for very\n    large n (several thousand or more) feasible.\n\n    References\n    ----------\n    .. [townsend.trogdon.olver-2014]\n        Townsend, A. and Trogdon, T. and Olver, S. (2014)\n        *Fast computation of Gauss quadrature nodes and\n        weights on the whole real line*. :arXiv:`1410.5286`.\n    .. [townsend.trogdon.olver-2015]\n        Townsend, A. and Trogdon, T. and Olver, S. (2015)\n        *Fast computation of Gauss quadrature nodes and\n        weights on the whole real line*.\n        IMA Journal of Numerical Analysis\n        :doi:`10.1093/imanum/drv002`.\n    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.\n        Handbook of Mathematical Functions with Formulas,\n        Graphs, and Mathematical Tables. New York: Dover, 1972.\n\n    "
    m = int(n)
    if n < 1 or n != m:
        raise ValueError('n must be a positive integer.')
    mu0 = np.sqrt(np.pi)
    if n <= 150:

        def an_func(k):
            if False:
                return 10
            return 0.0 * k

        def bn_func(k):
            if False:
                return 10
            return np.sqrt(k / 2.0)
        f = _ufuncs.eval_hermite

        def df(n, x):
            if False:
                return 10
            return 2.0 * n * _ufuncs.eval_hermite(n - 1, x)
        return _gen_roots_and_weights(m, mu0, an_func, bn_func, f, df, True, mu)
    else:
        (nodes, weights) = _roots_hermite_asy(m)
        if mu:
            return (nodes, weights, mu0)
        else:
            return (nodes, weights)

def _compute_tauk(n, k, maxit=5):
    if False:
        for i in range(10):
            print('nop')
    'Helper function for Tricomi initial guesses\n\n    For details, see formula 3.1 in lemma 3.1 in the\n    original paper.\n\n    Parameters\n    ----------\n    n : int\n        Quadrature order\n    k : ndarray of type int\n        Index of roots :math:`\tau_k` to compute\n    maxit : int\n        Number of Newton maxit performed, the default\n        value of 5 is sufficient.\n\n    Returns\n    -------\n    tauk : ndarray\n        Roots of equation 3.1\n\n    See Also\n    --------\n    initial_nodes_a\n    roots_hermite_asy\n    '
    a = n % 2 - 0.5
    c = (4.0 * floor(n / 2.0) - 4.0 * k + 3.0) * pi / (4.0 * floor(n / 2.0) + 2.0 * a + 2.0)

    def f(x):
        if False:
            i = 10
            return i + 15
        return x - sin(x) - c

    def df(x):
        if False:
            i = 10
            return i + 15
        return 1.0 - cos(x)
    xi = 0.5 * pi
    for i in range(maxit):
        xi = xi - f(xi) / df(xi)
    return xi

def _initial_nodes_a(n, k):
    if False:
        while True:
            i = 10
    'Tricomi initial guesses\n\n    Computes an initial approximation to the square of the `k`-th\n    (positive) root :math:`x_k` of the Hermite polynomial :math:`H_n`\n    of order :math:`n`. The formula is the one from lemma 3.1 in the\n    original paper. The guesses are accurate except in the region\n    near :math:`\\sqrt{2n + 1}`.\n\n    Parameters\n    ----------\n    n : int\n        Quadrature order\n    k : ndarray of type int\n        Index of roots to compute\n\n    Returns\n    -------\n    xksq : ndarray\n        Square of the approximate roots\n\n    See Also\n    --------\n    initial_nodes\n    roots_hermite_asy\n    '
    tauk = _compute_tauk(n, k)
    sigk = cos(0.5 * tauk) ** 2
    a = n % 2 - 0.5
    nu = 4.0 * floor(n / 2.0) + 2.0 * a + 2.0
    xksq = nu * sigk - 1.0 / (3.0 * nu) * (5.0 / (4.0 * (1.0 - sigk) ** 2) - 1.0 / (1.0 - sigk) - 0.25)
    return xksq

def _initial_nodes_b(n, k):
    if False:
        while True:
            i = 10
    'Gatteschi initial guesses\n\n    Computes an initial approximation to the square of the kth\n    (positive) root :math:`x_k` of the Hermite polynomial :math:`H_n`\n    of order :math:`n`. The formula is the one from lemma 3.2 in the\n    original paper. The guesses are accurate in the region just\n    below :math:`\\sqrt{2n + 1}`.\n\n    Parameters\n    ----------\n    n : int\n        Quadrature order\n    k : ndarray of type int\n        Index of roots to compute\n\n    Returns\n    -------\n    xksq : ndarray\n        Square of the approximate root\n\n    See Also\n    --------\n    initial_nodes\n    roots_hermite_asy\n    '
    a = n % 2 - 0.5
    nu = 4.0 * floor(n / 2.0) + 2.0 * a + 2.0
    ak = _specfun.airyzo(k.max(), 1)[0][::-1]
    xksq = nu + 2.0 ** (2.0 / 3.0) * ak * nu ** (1.0 / 3.0) + 1.0 / 5.0 * 2.0 ** (4.0 / 3.0) * ak ** 2 * nu ** (-1.0 / 3.0) + (9.0 / 140.0 - 12.0 / 175.0 * ak ** 3) * nu ** (-1.0) + (16.0 / 1575.0 * ak + 92.0 / 7875.0 * ak ** 4) * 2.0 ** (2.0 / 3.0) * nu ** (-5.0 / 3.0) - (15152.0 / 3031875.0 * ak ** 5 + 1088.0 / 121275.0 * ak ** 2) * 2.0 ** (1.0 / 3.0) * nu ** (-7.0 / 3.0)
    return xksq

def _initial_nodes(n):
    if False:
        print('Hello World!')
    'Initial guesses for the Hermite roots\n\n    Computes an initial approximation to the non-negative\n    roots :math:`x_k` of the Hermite polynomial :math:`H_n`\n    of order :math:`n`. The Tricomi and Gatteschi initial\n    guesses are used in the region where they are accurate.\n\n    Parameters\n    ----------\n    n : int\n        Quadrature order\n\n    Returns\n    -------\n    xk : ndarray\n        Approximate roots\n\n    See Also\n    --------\n    roots_hermite_asy\n    '
    fit = 0.49082003 * n - 4.37859653
    turnover = around(fit).astype(int)
    ia = arange(1, int(floor(n * 0.5) + 1))
    ib = ia[::-1]
    xasq = _initial_nodes_a(n, ia[:turnover + 1])
    xbsq = _initial_nodes_b(n, ib[turnover + 1:])
    iv = sqrt(hstack([xasq, xbsq]))
    if n % 2 == 1:
        iv = hstack([0.0, iv])
    return iv

def _pbcf(n, theta):
    if False:
        while True:
            i = 10
    'Asymptotic series expansion of parabolic cylinder function\n\n    The implementation is based on sections 3.2 and 3.3 from the\n    original paper. Compared to the published version this code\n    adds one more term to the asymptotic series. The detailed\n    formulas can be found at [parabolic-asymptotics]_. The evaluation\n    is done in a transformed variable :math:`\\theta := \\arccos(t)`\n    where :math:`t := x / \\mu` and :math:`\\mu := \\sqrt{2n + 1}`.\n\n    Parameters\n    ----------\n    n : int\n        Quadrature order\n    theta : ndarray\n        Transformed position variable\n\n    Returns\n    -------\n    U : ndarray\n        Value of the parabolic cylinder function :math:`U(a, \\theta)`.\n    Ud : ndarray\n        Value of the derivative :math:`U^{\\prime}(a, \\theta)` of\n        the parabolic cylinder function.\n\n    See Also\n    --------\n    roots_hermite_asy\n\n    References\n    ----------\n    .. [parabolic-asymptotics]\n       https://dlmf.nist.gov/12.10#vii\n    '
    st = sin(theta)
    ct = cos(theta)
    mu = 2.0 * n + 1.0
    eta = 0.5 * theta - 0.5 * st * ct
    zeta = -(3.0 * eta / 2.0) ** (2.0 / 3.0)
    phi = (-zeta / st ** 2) ** 0.25
    a0 = 1.0
    a1 = 0.10416666666666667
    a2 = 0.08355034722222222
    a3 = 0.12822657455632716
    a4 = 0.29184902646414046
    a5 = 0.8816272674437576
    b0 = 1.0
    b1 = -0.14583333333333334
    b2 = -0.09874131944444445
    b3 = -0.14331205391589505
    b4 = -0.31722720267841353
    b5 = -0.9424291479571203
    ctp = ct ** arange(16).reshape((-1, 1))
    u0 = 1.0
    u1 = (1.0 * ctp[3, :] - 6.0 * ct) / 24.0
    u2 = (-9.0 * ctp[4, :] + 249.0 * ctp[2, :] + 145.0) / 1152.0
    u3 = (-4042.0 * ctp[9, :] + 18189.0 * ctp[7, :] - 28287.0 * ctp[5, :] - 151995.0 * ctp[3, :] - 259290.0 * ct) / 414720.0
    u4 = (72756.0 * ctp[10, :] - 321339.0 * ctp[8, :] - 154982.0 * ctp[6, :] + 50938215.0 * ctp[4, :] + 122602962.0 * ctp[2, :] + 12773113.0) / 39813120.0
    u5 = (82393456.0 * ctp[15, :] - 617950920.0 * ctp[13, :] + 1994971575.0 * ctp[11, :] - 3630137104.0 * ctp[9, :] + 4433574213.0 * ctp[7, :] - 37370295816.0 * ctp[5, :] - 119582875013.0 * ctp[3, :] - 34009066266.0 * ct) / 6688604160.0
    v0 = 1.0
    v1 = (1.0 * ctp[3, :] + 6.0 * ct) / 24.0
    v2 = (15.0 * ctp[4, :] - 327.0 * ctp[2, :] - 143.0) / 1152.0
    v3 = (-4042.0 * ctp[9, :] + 18189.0 * ctp[7, :] - 36387.0 * ctp[5, :] + 238425.0 * ctp[3, :] + 259290.0 * ct) / 414720.0
    v4 = (-121260.0 * ctp[10, :] + 551733.0 * ctp[8, :] - 151958.0 * ctp[6, :] - 57484425.0 * ctp[4, :] - 132752238.0 * ctp[2, :] - 12118727) / 39813120.0
    v5 = (82393456.0 * ctp[15, :] - 617950920.0 * ctp[13, :] + 2025529095.0 * ctp[11, :] - 3750839308.0 * ctp[9, :] + 3832454253.0 * ctp[7, :] + 35213253348.0 * ctp[5, :] + 130919230435.0 * ctp[3, :] + 34009066266 * ct) / 6688604160.0
    (Ai, Aip, Bi, Bip) = airy(mu ** (4.0 / 6.0) * zeta)
    P = 2.0 * sqrt(pi) * mu ** (1.0 / 6.0) * phi
    phip = phi ** arange(6, 31, 6).reshape((-1, 1))
    A0 = b0 * u0
    A1 = (b2 * u0 + phip[0, :] * b1 * u1 + phip[1, :] * b0 * u2) / zeta ** 3
    A2 = (b4 * u0 + phip[0, :] * b3 * u1 + phip[1, :] * b2 * u2 + phip[2, :] * b1 * u3 + phip[3, :] * b0 * u4) / zeta ** 6
    B0 = -(a1 * u0 + phip[0, :] * a0 * u1) / zeta ** 2
    B1 = -(a3 * u0 + phip[0, :] * a2 * u1 + phip[1, :] * a1 * u2 + phip[2, :] * a0 * u3) / zeta ** 5
    B2 = -(a5 * u0 + phip[0, :] * a4 * u1 + phip[1, :] * a3 * u2 + phip[2, :] * a2 * u3 + phip[3, :] * a1 * u4 + phip[4, :] * a0 * u5) / zeta ** 8
    U = P * (Ai * (A0 + A1 / mu ** 2.0 + A2 / mu ** 4.0) + Aip * (B0 + B1 / mu ** 2.0 + B2 / mu ** 4.0) / mu ** (8.0 / 6.0))
    Pd = sqrt(2.0 * pi) * mu ** (2.0 / 6.0) / phi
    C0 = -(b1 * v0 + phip[0, :] * b0 * v1) / zeta
    C1 = -(b3 * v0 + phip[0, :] * b2 * v1 + phip[1, :] * b1 * v2 + phip[2, :] * b0 * v3) / zeta ** 4
    C2 = -(b5 * v0 + phip[0, :] * b4 * v1 + phip[1, :] * b3 * v2 + phip[2, :] * b2 * v3 + phip[3, :] * b1 * v4 + phip[4, :] * b0 * v5) / zeta ** 7
    D0 = a0 * v0
    D1 = (a2 * v0 + phip[0, :] * a1 * v1 + phip[1, :] * a0 * v2) / zeta ** 3
    D2 = (a4 * v0 + phip[0, :] * a3 * v1 + phip[1, :] * a2 * v2 + phip[2, :] * a1 * v3 + phip[3, :] * a0 * v4) / zeta ** 6
    Ud = Pd * (Ai * (C0 + C1 / mu ** 2.0 + C2 / mu ** 4.0) / mu ** (4.0 / 6.0) + Aip * (D0 + D1 / mu ** 2.0 + D2 / mu ** 4.0))
    return (U, Ud)

def _newton(n, x_initial, maxit=5):
    if False:
        for i in range(10):
            print('nop')
    'Newton iteration for polishing the asymptotic approximation\n    to the zeros of the Hermite polynomials.\n\n    Parameters\n    ----------\n    n : int\n        Quadrature order\n    x_initial : ndarray\n        Initial guesses for the roots\n    maxit : int\n        Maximal number of Newton iterations.\n        The default 5 is sufficient, usually\n        only one or two steps are needed.\n\n    Returns\n    -------\n    nodes : ndarray\n        Quadrature nodes\n    weights : ndarray\n        Quadrature weights\n\n    See Also\n    --------\n    roots_hermite_asy\n    '
    mu = sqrt(2.0 * n + 1.0)
    t = x_initial / mu
    theta = arccos(t)
    for i in range(maxit):
        (u, ud) = _pbcf(n, theta)
        dtheta = u / (sqrt(2.0) * mu * sin(theta) * ud)
        theta = theta + dtheta
        if max(abs(dtheta)) < 1e-14:
            break
    x = mu * cos(theta)
    if n % 2 == 1:
        x[0] = 0.0
    w = exp(-x ** 2) / (2.0 * ud ** 2)
    return (x, w)

def _roots_hermite_asy(n):
    if False:
        while True:
            i = 10
    "Gauss-Hermite (physicist's) quadrature for large n.\n\n    Computes the sample points and weights for Gauss-Hermite quadrature.\n    The sample points are the roots of the nth degree Hermite polynomial,\n    :math:`H_n(x)`. These sample points and weights correctly integrate\n    polynomials of degree :math:`2n - 1` or less over the interval\n    :math:`[-\\infty, \\infty]` with weight function :math:`f(x) = e^{-x^2}`.\n\n    This method relies on asymptotic expansions which work best for n > 150.\n    The algorithm has linear runtime making computation for very large n\n    feasible.\n\n    Parameters\n    ----------\n    n : int\n        quadrature order\n\n    Returns\n    -------\n    nodes : ndarray\n        Quadrature nodes\n    weights : ndarray\n        Quadrature weights\n\n    See Also\n    --------\n    roots_hermite\n\n    References\n    ----------\n    .. [townsend.trogdon.olver-2014]\n       Townsend, A. and Trogdon, T. and Olver, S. (2014)\n       *Fast computation of Gauss quadrature nodes and\n       weights on the whole real line*. :arXiv:`1410.5286`.\n\n    .. [townsend.trogdon.olver-2015]\n       Townsend, A. and Trogdon, T. and Olver, S. (2015)\n       *Fast computation of Gauss quadrature nodes and\n       weights on the whole real line*.\n       IMA Journal of Numerical Analysis\n       :doi:`10.1093/imanum/drv002`.\n    "
    iv = _initial_nodes(n)
    (nodes, weights) = _newton(n, iv)
    if n % 2 == 0:
        nodes = hstack([-nodes[::-1], nodes])
        weights = hstack([weights[::-1], weights])
    else:
        nodes = hstack([-nodes[-1:0:-1], nodes])
        weights = hstack([weights[-1:0:-1], weights])
    weights *= sqrt(pi) / sum(weights)
    return (nodes, weights)

def hermite(n, monic=False):
    if False:
        print('Hello World!')
    'Physicist\'s Hermite polynomial.\n\n    Defined by\n\n    .. math::\n\n        H_n(x) = (-1)^ne^{x^2}\\frac{d^n}{dx^n}e^{-x^2};\n\n    :math:`H_n` is a polynomial of degree :math:`n`.\n\n    Parameters\n    ----------\n    n : int\n        Degree of the polynomial.\n    monic : bool, optional\n        If `True`, scale the leading coefficient to be 1. Default is\n        `False`.\n\n    Returns\n    -------\n    H : orthopoly1d\n        Hermite polynomial.\n\n    Notes\n    -----\n    The polynomials :math:`H_n` are orthogonal over :math:`(-\\infty,\n    \\infty)` with weight function :math:`e^{-x^2}`.\n\n    Examples\n    --------\n    >>> from scipy import special\n    >>> import matplotlib.pyplot as plt\n    >>> import numpy as np\n\n    >>> p_monic = special.hermite(3, monic=True)\n    >>> p_monic\n    poly1d([ 1. ,  0. , -1.5,  0. ])\n    >>> p_monic(1)\n    -0.49999999999999983\n    >>> x = np.linspace(-3, 3, 400)\n    >>> y = p_monic(x)\n    >>> plt.plot(x, y)\n    >>> plt.title("Monic Hermite polynomial of degree 3")\n    >>> plt.xlabel("x")\n    >>> plt.ylabel("H_3(x)")\n    >>> plt.show()\n\n    '
    if n < 0:
        raise ValueError('n must be nonnegative.')
    if n == 0:
        n1 = n + 1
    else:
        n1 = n
    (x, w) = roots_hermite(n1)

    def wfunc(x):
        if False:
            while True:
                i = 10
        return exp(-x * x)
    if n == 0:
        (x, w) = ([], [])
    hn = 2 ** n * _gam(n + 1) * sqrt(pi)
    kn = 2 ** n
    p = orthopoly1d(x, w, hn, kn, wfunc, (-inf, inf), monic, lambda x: _ufuncs.eval_hermite(n, x))
    return p

def roots_hermitenorm(n, mu=False):
    if False:
        print('Hello World!')
    "Gauss-Hermite (statistician's) quadrature.\n\n    Compute the sample points and weights for Gauss-Hermite\n    quadrature. The sample points are the roots of the nth degree\n    Hermite polynomial, :math:`He_n(x)`. These sample points and\n    weights correctly integrate polynomials of degree :math:`2n - 1`\n    or less over the interval :math:`[-\\infty, \\infty]` with weight\n    function :math:`w(x) = e^{-x^2/2}`. See 22.2.15 in [AS]_ for more\n    details.\n\n    Parameters\n    ----------\n    n : int\n        quadrature order\n    mu : bool, optional\n        If True, return the sum of the weights, optional.\n\n    Returns\n    -------\n    x : ndarray\n        Sample points\n    w : ndarray\n        Weights\n    mu : float\n        Sum of the weights\n\n    See Also\n    --------\n    scipy.integrate.quadrature\n    scipy.integrate.fixed_quad\n    numpy.polynomial.hermite_e.hermegauss\n\n    Notes\n    -----\n    For small n up to 150 a modified version of the Golub-Welsch\n    algorithm is used. Nodes are computed from the eigenvalue\n    problem and improved by one step of a Newton iteration.\n    The weights are computed from the well-known analytical formula.\n\n    For n larger than 150 an optimal asymptotic algorithm is used\n    which computes nodes and weights in a numerical stable manner.\n    The algorithm has linear runtime making computation for very\n    large n (several thousand or more) feasible.\n\n    References\n    ----------\n    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.\n        Handbook of Mathematical Functions with Formulas,\n        Graphs, and Mathematical Tables. New York: Dover, 1972.\n\n    "
    m = int(n)
    if n < 1 or n != m:
        raise ValueError('n must be a positive integer.')
    mu0 = np.sqrt(2.0 * np.pi)
    if n <= 150:

        def an_func(k):
            if False:
                i = 10
                return i + 15
            return 0.0 * k

        def bn_func(k):
            if False:
                while True:
                    i = 10
            return np.sqrt(k)
        f = _ufuncs.eval_hermitenorm

        def df(n, x):
            if False:
                return 10
            return n * _ufuncs.eval_hermitenorm(n - 1, x)
        return _gen_roots_and_weights(m, mu0, an_func, bn_func, f, df, True, mu)
    else:
        (nodes, weights) = _roots_hermite_asy(m)
        nodes *= sqrt(2)
        weights *= sqrt(2)
        if mu:
            return (nodes, weights, mu0)
        else:
            return (nodes, weights)

def hermitenorm(n, monic=False):
    if False:
        return 10
    "Normalized (probabilist's) Hermite polynomial.\n\n    Defined by\n\n    .. math::\n\n        He_n(x) = (-1)^ne^{x^2/2}\\frac{d^n}{dx^n}e^{-x^2/2};\n\n    :math:`He_n` is a polynomial of degree :math:`n`.\n\n    Parameters\n    ----------\n    n : int\n        Degree of the polynomial.\n    monic : bool, optional\n        If `True`, scale the leading coefficient to be 1. Default is\n        `False`.\n\n    Returns\n    -------\n    He : orthopoly1d\n        Hermite polynomial.\n\n    Notes\n    -----\n\n    The polynomials :math:`He_n` are orthogonal over :math:`(-\\infty,\n    \\infty)` with weight function :math:`e^{-x^2/2}`.\n\n    "
    if n < 0:
        raise ValueError('n must be nonnegative.')
    if n == 0:
        n1 = n + 1
    else:
        n1 = n
    (x, w) = roots_hermitenorm(n1)

    def wfunc(x):
        if False:
            for i in range(10):
                print('nop')
        return exp(-x * x / 2.0)
    if n == 0:
        (x, w) = ([], [])
    hn = sqrt(2 * pi) * _gam(n + 1)
    kn = 1.0
    p = orthopoly1d(x, w, hn, kn, wfunc=wfunc, limits=(-inf, inf), monic=monic, eval_func=lambda x: _ufuncs.eval_hermitenorm(n, x))
    return p

def roots_gegenbauer(n, alpha, mu=False):
    if False:
        while True:
            i = 10
    'Gauss-Gegenbauer quadrature.\n\n    Compute the sample points and weights for Gauss-Gegenbauer\n    quadrature. The sample points are the roots of the nth degree\n    Gegenbauer polynomial, :math:`C^{\\alpha}_n(x)`. These sample\n    points and weights correctly integrate polynomials of degree\n    :math:`2n - 1` or less over the interval :math:`[-1, 1]` with\n    weight function :math:`w(x) = (1 - x^2)^{\\alpha - 1/2}`. See\n    22.2.3 in [AS]_ for more details.\n\n    Parameters\n    ----------\n    n : int\n        quadrature order\n    alpha : float\n        alpha must be > -0.5\n    mu : bool, optional\n        If True, return the sum of the weights, optional.\n\n    Returns\n    -------\n    x : ndarray\n        Sample points\n    w : ndarray\n        Weights\n    mu : float\n        Sum of the weights\n\n    See Also\n    --------\n    scipy.integrate.quadrature\n    scipy.integrate.fixed_quad\n\n    References\n    ----------\n    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.\n        Handbook of Mathematical Functions with Formulas,\n        Graphs, and Mathematical Tables. New York: Dover, 1972.\n\n    '
    m = int(n)
    if n < 1 or n != m:
        raise ValueError('n must be a positive integer.')
    if alpha < -0.5:
        raise ValueError('alpha must be greater than -0.5.')
    elif alpha == 0.0:
        return roots_chebyt(n, mu)
    if alpha <= 170:
        mu0 = np.sqrt(np.pi) * _ufuncs.gamma(alpha + 0.5) / _ufuncs.gamma(alpha + 1)
    else:
        inv_alpha = 1.0 / alpha
        coeffs = np.array([0.000207186, -0.00152206, -0.000640869, 0.00488281, 0.0078125, -0.125, 1.0])
        mu0 = coeffs[0]
        for term in range(1, len(coeffs)):
            mu0 = mu0 * inv_alpha + coeffs[term]
        mu0 = mu0 * np.sqrt(np.pi / alpha)

    def an_func(k):
        if False:
            i = 10
            return i + 15
        return 0.0 * k

    def bn_func(k):
        if False:
            print('Hello World!')
        return np.sqrt(k * (k + 2 * alpha - 1) / (4 * (k + alpha) * (k + alpha - 1)))

    def f(n, x):
        if False:
            return 10
        return _ufuncs.eval_gegenbauer(n, alpha, x)

    def df(n, x):
        if False:
            i = 10
            return i + 15
        return (-n * x * _ufuncs.eval_gegenbauer(n, alpha, x) + (n + 2 * alpha - 1) * _ufuncs.eval_gegenbauer(n - 1, alpha, x)) / (1 - x ** 2)
    return _gen_roots_and_weights(m, mu0, an_func, bn_func, f, df, True, mu)

def gegenbauer(n, alpha, monic=False):
    if False:
        while True:
            i = 10
    'Gegenbauer (ultraspherical) polynomial.\n\n    Defined to be the solution of\n\n    .. math::\n        (1 - x^2)\\frac{d^2}{dx^2}C_n^{(\\alpha)}\n          - (2\\alpha + 1)x\\frac{d}{dx}C_n^{(\\alpha)}\n          + n(n + 2\\alpha)C_n^{(\\alpha)} = 0\n\n    for :math:`\\alpha > -1/2`; :math:`C_n^{(\\alpha)}` is a polynomial\n    of degree :math:`n`.\n\n    Parameters\n    ----------\n    n : int\n        Degree of the polynomial.\n    alpha : float\n        Parameter, must be greater than -0.5.\n    monic : bool, optional\n        If `True`, scale the leading coefficient to be 1. Default is\n        `False`.\n\n    Returns\n    -------\n    C : orthopoly1d\n        Gegenbauer polynomial.\n\n    Notes\n    -----\n    The polynomials :math:`C_n^{(\\alpha)}` are orthogonal over\n    :math:`[-1,1]` with weight function :math:`(1 - x^2)^{(\\alpha -\n    1/2)}`.\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from scipy import special\n    >>> import matplotlib.pyplot as plt\n\n    We can initialize a variable ``p`` as a Gegenbauer polynomial using the\n    `gegenbauer` function and evaluate at a point ``x = 1``.\n\n    >>> p = special.gegenbauer(3, 0.5, monic=False)\n    >>> p\n    poly1d([ 2.5,  0. , -1.5,  0. ])\n    >>> p(1)\n    1.0\n\n    To evaluate ``p`` at various points ``x`` in the interval ``(-3, 3)``,\n    simply pass an array ``x`` to ``p`` as follows:\n\n    >>> x = np.linspace(-3, 3, 400)\n    >>> y = p(x)\n\n    We can then visualize ``x, y`` using `matplotlib.pyplot`.\n\n    >>> fig, ax = plt.subplots()\n    >>> ax.plot(x, y)\n    >>> ax.set_title("Gegenbauer (ultraspherical) polynomial of degree 3")\n    >>> ax.set_xlabel("x")\n    >>> ax.set_ylabel("G_3(x)")\n    >>> plt.show()\n\n    '
    base = jacobi(n, alpha - 0.5, alpha - 0.5, monic=monic)
    if monic:
        return base
    factor = _gam(2 * alpha + n) * _gam(alpha + 0.5) / _gam(2 * alpha) / _gam(alpha + 0.5 + n)
    base._scale(factor)
    base.__dict__['_eval_func'] = lambda x: _ufuncs.eval_gegenbauer(float(n), alpha, x)
    return base

def roots_chebyt(n, mu=False):
    if False:
        while True:
            i = 10
    'Gauss-Chebyshev (first kind) quadrature.\n\n    Computes the sample points and weights for Gauss-Chebyshev\n    quadrature. The sample points are the roots of the nth degree\n    Chebyshev polynomial of the first kind, :math:`T_n(x)`. These\n    sample points and weights correctly integrate polynomials of\n    degree :math:`2n - 1` or less over the interval :math:`[-1, 1]`\n    with weight function :math:`w(x) = 1/\\sqrt{1 - x^2}`. See 22.2.4\n    in [AS]_ for more details.\n\n    Parameters\n    ----------\n    n : int\n        quadrature order\n    mu : bool, optional\n        If True, return the sum of the weights, optional.\n\n    Returns\n    -------\n    x : ndarray\n        Sample points\n    w : ndarray\n        Weights\n    mu : float\n        Sum of the weights\n\n    See Also\n    --------\n    scipy.integrate.quadrature\n    scipy.integrate.fixed_quad\n    numpy.polynomial.chebyshev.chebgauss\n\n    References\n    ----------\n    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.\n        Handbook of Mathematical Functions with Formulas,\n        Graphs, and Mathematical Tables. New York: Dover, 1972.\n\n    '
    m = int(n)
    if n < 1 or n != m:
        raise ValueError('n must be a positive integer.')
    x = _ufuncs._sinpi(np.arange(-m + 1, m, 2) / (2 * m))
    w = np.full_like(x, pi / m)
    if mu:
        return (x, w, pi)
    else:
        return (x, w)

def chebyt(n, monic=False):
    if False:
        return 10
    "Chebyshev polynomial of the first kind.\n\n    Defined to be the solution of\n\n    .. math::\n        (1 - x^2)\\frac{d^2}{dx^2}T_n - x\\frac{d}{dx}T_n + n^2T_n = 0;\n\n    :math:`T_n` is a polynomial of degree :math:`n`.\n\n    Parameters\n    ----------\n    n : int\n        Degree of the polynomial.\n    monic : bool, optional\n        If `True`, scale the leading coefficient to be 1. Default is\n        `False`.\n\n    Returns\n    -------\n    T : orthopoly1d\n        Chebyshev polynomial of the first kind.\n\n    See Also\n    --------\n    chebyu : Chebyshev polynomial of the second kind.\n\n    Notes\n    -----\n    The polynomials :math:`T_n` are orthogonal over :math:`[-1, 1]`\n    with weight function :math:`(1 - x^2)^{-1/2}`.\n\n    References\n    ----------\n    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.\n        Handbook of Mathematical Functions with Formulas,\n        Graphs, and Mathematical Tables. New York: Dover, 1972.\n\n    Examples\n    --------\n    Chebyshev polynomials of the first kind of order :math:`n` can\n    be obtained as the determinant of specific :math:`n \\times n`\n    matrices. As an example we can check how the points obtained from\n    the determinant of the following :math:`3 \\times 3` matrix\n    lay exactly on :math:`T_3`:\n\n    >>> import numpy as np\n    >>> import matplotlib.pyplot as plt\n    >>> from scipy.linalg import det\n    >>> from scipy.special import chebyt\n    >>> x = np.arange(-1.0, 1.0, 0.01)\n    >>> fig, ax = plt.subplots()\n    >>> ax.set_ylim(-2.0, 2.0)\n    >>> ax.set_title(r'Chebyshev polynomial $T_3$')\n    >>> ax.plot(x, chebyt(3)(x), label=rf'$T_3$')\n    >>> for p in np.arange(-1.0, 1.0, 0.1):\n    ...     ax.plot(p,\n    ...             det(np.array([[p, 1, 0], [1, 2*p, 1], [0, 1, 2*p]])),\n    ...             'rx')\n    >>> plt.legend(loc='best')\n    >>> plt.show()\n\n    They are also related to the Jacobi Polynomials\n    :math:`P_n^{(-0.5, -0.5)}` through the relation:\n\n    .. math::\n        P_n^{(-0.5, -0.5)}(x) = \\frac{1}{4^n} \\binom{2n}{n} T_n(x)\n\n    Let's verify it for :math:`n = 3`:\n\n    >>> from scipy.special import binom\n    >>> from scipy.special import jacobi\n    >>> x = np.arange(-1.0, 1.0, 0.01)\n    >>> np.allclose(jacobi(3, -0.5, -0.5)(x),\n    ...             1/64 * binom(6, 3) * chebyt(3)(x))\n    True\n\n    We can plot the Chebyshev polynomials :math:`T_n` for some values\n    of :math:`n`:\n\n    >>> x = np.arange(-1.5, 1.5, 0.01)\n    >>> fig, ax = plt.subplots()\n    >>> ax.set_ylim(-4.0, 4.0)\n    >>> ax.set_title(r'Chebyshev polynomials $T_n$')\n    >>> for n in np.arange(2,5):\n    ...     ax.plot(x, chebyt(n)(x), label=rf'$T_n={n}$')\n    >>> plt.legend(loc='best')\n    >>> plt.show()\n\n    "
    if n < 0:
        raise ValueError('n must be nonnegative.')

    def wfunc(x):
        if False:
            while True:
                i = 10
        return 1.0 / sqrt(1 - x * x)
    if n == 0:
        return orthopoly1d([], [], pi, 1.0, wfunc, (-1, 1), monic, lambda x: _ufuncs.eval_chebyt(n, x))
    n1 = n
    (x, w, mu) = roots_chebyt(n1, mu=True)
    hn = pi / 2
    kn = 2 ** (n - 1)
    p = orthopoly1d(x, w, hn, kn, wfunc, (-1, 1), monic, lambda x: _ufuncs.eval_chebyt(n, x))
    return p

def roots_chebyu(n, mu=False):
    if False:
        while True:
            i = 10
    'Gauss-Chebyshev (second kind) quadrature.\n\n    Computes the sample points and weights for Gauss-Chebyshev\n    quadrature. The sample points are the roots of the nth degree\n    Chebyshev polynomial of the second kind, :math:`U_n(x)`. These\n    sample points and weights correctly integrate polynomials of\n    degree :math:`2n - 1` or less over the interval :math:`[-1, 1]`\n    with weight function :math:`w(x) = \\sqrt{1 - x^2}`. See 22.2.5 in\n    [AS]_ for details.\n\n    Parameters\n    ----------\n    n : int\n        quadrature order\n    mu : bool, optional\n        If True, return the sum of the weights, optional.\n\n    Returns\n    -------\n    x : ndarray\n        Sample points\n    w : ndarray\n        Weights\n    mu : float\n        Sum of the weights\n\n    See Also\n    --------\n    scipy.integrate.quadrature\n    scipy.integrate.fixed_quad\n\n    References\n    ----------\n    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.\n        Handbook of Mathematical Functions with Formulas,\n        Graphs, and Mathematical Tables. New York: Dover, 1972.\n\n    '
    m = int(n)
    if n < 1 or n != m:
        raise ValueError('n must be a positive integer.')
    t = np.arange(m, 0, -1) * pi / (m + 1)
    x = np.cos(t)
    w = pi * np.sin(t) ** 2 / (m + 1)
    if mu:
        return (x, w, pi / 2)
    else:
        return (x, w)

def chebyu(n, monic=False):
    if False:
        print('Hello World!')
    "Chebyshev polynomial of the second kind.\n\n    Defined to be the solution of\n\n    .. math::\n        (1 - x^2)\\frac{d^2}{dx^2}U_n - 3x\\frac{d}{dx}U_n\n          + n(n + 2)U_n = 0;\n\n    :math:`U_n` is a polynomial of degree :math:`n`.\n\n    Parameters\n    ----------\n    n : int\n        Degree of the polynomial.\n    monic : bool, optional\n        If `True`, scale the leading coefficient to be 1. Default is\n        `False`.\n\n    Returns\n    -------\n    U : orthopoly1d\n        Chebyshev polynomial of the second kind.\n\n    See Also\n    --------\n    chebyt : Chebyshev polynomial of the first kind.\n\n    Notes\n    -----\n    The polynomials :math:`U_n` are orthogonal over :math:`[-1, 1]`\n    with weight function :math:`(1 - x^2)^{1/2}`.\n\n    References\n    ----------\n    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.\n        Handbook of Mathematical Functions with Formulas,\n        Graphs, and Mathematical Tables. New York: Dover, 1972.\n\n    Examples\n    --------\n    Chebyshev polynomials of the second kind of order :math:`n` can\n    be obtained as the determinant of specific :math:`n \\times n`\n    matrices. As an example we can check how the points obtained from\n    the determinant of the following :math:`3 \\times 3` matrix\n    lay exactly on :math:`U_3`:\n\n    >>> import numpy as np\n    >>> import matplotlib.pyplot as plt\n    >>> from scipy.linalg import det\n    >>> from scipy.special import chebyu\n    >>> x = np.arange(-1.0, 1.0, 0.01)\n    >>> fig, ax = plt.subplots()\n    >>> ax.set_ylim(-2.0, 2.0)\n    >>> ax.set_title(r'Chebyshev polynomial $U_3$')\n    >>> ax.plot(x, chebyu(3)(x), label=rf'$U_3$')\n    >>> for p in np.arange(-1.0, 1.0, 0.1):\n    ...     ax.plot(p,\n    ...             det(np.array([[2*p, 1, 0], [1, 2*p, 1], [0, 1, 2*p]])),\n    ...             'rx')\n    >>> plt.legend(loc='best')\n    >>> plt.show()\n\n    They satisfy the recurrence relation:\n\n    .. math::\n        U_{2n-1}(x) = 2 T_n(x)U_{n-1}(x)\n\n    where the :math:`T_n` are the Chebyshev polynomial of the first kind.\n    Let's verify it for :math:`n = 2`:\n\n    >>> from scipy.special import chebyt\n    >>> x = np.arange(-1.0, 1.0, 0.01)\n    >>> np.allclose(chebyu(3)(x), 2 * chebyt(2)(x) * chebyu(1)(x))\n    True\n\n    We can plot the Chebyshev polynomials :math:`U_n` for some values\n    of :math:`n`:\n\n    >>> x = np.arange(-1.0, 1.0, 0.01)\n    >>> fig, ax = plt.subplots()\n    >>> ax.set_ylim(-1.5, 1.5)\n    >>> ax.set_title(r'Chebyshev polynomials $U_n$')\n    >>> for n in np.arange(1,5):\n    ...     ax.plot(x, chebyu(n)(x), label=rf'$U_n={n}$')\n    >>> plt.legend(loc='best')\n    >>> plt.show()\n\n    "
    base = jacobi(n, 0.5, 0.5, monic=monic)
    if monic:
        return base
    factor = sqrt(pi) / 2.0 * _gam(n + 2) / _gam(n + 1.5)
    base._scale(factor)
    return base

def roots_chebyc(n, mu=False):
    if False:
        print('Hello World!')
    'Gauss-Chebyshev (first kind) quadrature.\n\n    Compute the sample points and weights for Gauss-Chebyshev\n    quadrature. The sample points are the roots of the nth degree\n    Chebyshev polynomial of the first kind, :math:`C_n(x)`. These\n    sample points and weights correctly integrate polynomials of\n    degree :math:`2n - 1` or less over the interval :math:`[-2, 2]`\n    with weight function :math:`w(x) = 1 / \\sqrt{1 - (x/2)^2}`. See\n    22.2.6 in [AS]_ for more details.\n\n    Parameters\n    ----------\n    n : int\n        quadrature order\n    mu : bool, optional\n        If True, return the sum of the weights, optional.\n\n    Returns\n    -------\n    x : ndarray\n        Sample points\n    w : ndarray\n        Weights\n    mu : float\n        Sum of the weights\n\n    See Also\n    --------\n    scipy.integrate.quadrature\n    scipy.integrate.fixed_quad\n\n    References\n    ----------\n    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.\n        Handbook of Mathematical Functions with Formulas,\n        Graphs, and Mathematical Tables. New York: Dover, 1972.\n\n    '
    (x, w, m) = roots_chebyt(n, True)
    x *= 2
    w *= 2
    m *= 2
    if mu:
        return (x, w, m)
    else:
        return (x, w)

def chebyc(n, monic=False):
    if False:
        i = 10
        return i + 15
    'Chebyshev polynomial of the first kind on :math:`[-2, 2]`.\n\n    Defined as :math:`C_n(x) = 2T_n(x/2)`, where :math:`T_n` is the\n    nth Chebychev polynomial of the first kind.\n\n    Parameters\n    ----------\n    n : int\n        Degree of the polynomial.\n    monic : bool, optional\n        If `True`, scale the leading coefficient to be 1. Default is\n        `False`.\n\n    Returns\n    -------\n    C : orthopoly1d\n        Chebyshev polynomial of the first kind on :math:`[-2, 2]`.\n\n    See Also\n    --------\n    chebyt : Chebyshev polynomial of the first kind.\n\n    Notes\n    -----\n    The polynomials :math:`C_n(x)` are orthogonal over :math:`[-2, 2]`\n    with weight function :math:`1/\\sqrt{1 - (x/2)^2}`.\n\n    References\n    ----------\n    .. [1] Abramowitz and Stegun, "Handbook of Mathematical Functions"\n           Section 22. National Bureau of Standards, 1972.\n\n    '
    if n < 0:
        raise ValueError('n must be nonnegative.')
    if n == 0:
        n1 = n + 1
    else:
        n1 = n
    (x, w) = roots_chebyc(n1)
    if n == 0:
        (x, w) = ([], [])
    hn = 4 * pi * ((n == 0) + 1)
    kn = 1.0
    p = orthopoly1d(x, w, hn, kn, wfunc=lambda x: 1.0 / sqrt(1 - x * x / 4.0), limits=(-2, 2), monic=monic)
    if not monic:
        p._scale(2.0 / p(2))
        p.__dict__['_eval_func'] = lambda x: _ufuncs.eval_chebyc(n, x)
    return p

def roots_chebys(n, mu=False):
    if False:
        for i in range(10):
            print('nop')
    'Gauss-Chebyshev (second kind) quadrature.\n\n    Compute the sample points and weights for Gauss-Chebyshev\n    quadrature. The sample points are the roots of the nth degree\n    Chebyshev polynomial of the second kind, :math:`S_n(x)`. These\n    sample points and weights correctly integrate polynomials of\n    degree :math:`2n - 1` or less over the interval :math:`[-2, 2]`\n    with weight function :math:`w(x) = \\sqrt{1 - (x/2)^2}`. See 22.2.7\n    in [AS]_ for more details.\n\n    Parameters\n    ----------\n    n : int\n        quadrature order\n    mu : bool, optional\n        If True, return the sum of the weights, optional.\n\n    Returns\n    -------\n    x : ndarray\n        Sample points\n    w : ndarray\n        Weights\n    mu : float\n        Sum of the weights\n\n    See Also\n    --------\n    scipy.integrate.quadrature\n    scipy.integrate.fixed_quad\n\n    References\n    ----------\n    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.\n        Handbook of Mathematical Functions with Formulas,\n        Graphs, and Mathematical Tables. New York: Dover, 1972.\n\n    '
    (x, w, m) = roots_chebyu(n, True)
    x *= 2
    w *= 2
    m *= 2
    if mu:
        return (x, w, m)
    else:
        return (x, w)

def chebys(n, monic=False):
    if False:
        while True:
            i = 10
    'Chebyshev polynomial of the second kind on :math:`[-2, 2]`.\n\n    Defined as :math:`S_n(x) = U_n(x/2)` where :math:`U_n` is the\n    nth Chebychev polynomial of the second kind.\n\n    Parameters\n    ----------\n    n : int\n        Degree of the polynomial.\n    monic : bool, optional\n        If `True`, scale the leading coefficient to be 1. Default is\n        `False`.\n\n    Returns\n    -------\n    S : orthopoly1d\n        Chebyshev polynomial of the second kind on :math:`[-2, 2]`.\n\n    See Also\n    --------\n    chebyu : Chebyshev polynomial of the second kind\n\n    Notes\n    -----\n    The polynomials :math:`S_n(x)` are orthogonal over :math:`[-2, 2]`\n    with weight function :math:`\\sqrt{1 - (x/2)}^2`.\n\n    References\n    ----------\n    .. [1] Abramowitz and Stegun, "Handbook of Mathematical Functions"\n           Section 22. National Bureau of Standards, 1972.\n\n    '
    if n < 0:
        raise ValueError('n must be nonnegative.')
    if n == 0:
        n1 = n + 1
    else:
        n1 = n
    (x, w) = roots_chebys(n1)
    if n == 0:
        (x, w) = ([], [])
    hn = pi
    kn = 1.0
    p = orthopoly1d(x, w, hn, kn, wfunc=lambda x: sqrt(1 - x * x / 4.0), limits=(-2, 2), monic=monic)
    if not monic:
        factor = (n + 1.0) / p(2)
        p._scale(factor)
        p.__dict__['_eval_func'] = lambda x: _ufuncs.eval_chebys(n, x)
    return p

def roots_sh_chebyt(n, mu=False):
    if False:
        while True:
            i = 10
    'Gauss-Chebyshev (first kind, shifted) quadrature.\n\n    Compute the sample points and weights for Gauss-Chebyshev\n    quadrature. The sample points are the roots of the nth degree\n    shifted Chebyshev polynomial of the first kind, :math:`T_n(x)`.\n    These sample points and weights correctly integrate polynomials of\n    degree :math:`2n - 1` or less over the interval :math:`[0, 1]`\n    with weight function :math:`w(x) = 1/\\sqrt{x - x^2}`. See 22.2.8\n    in [AS]_ for more details.\n\n    Parameters\n    ----------\n    n : int\n        quadrature order\n    mu : bool, optional\n        If True, return the sum of the weights, optional.\n\n    Returns\n    -------\n    x : ndarray\n        Sample points\n    w : ndarray\n        Weights\n    mu : float\n        Sum of the weights\n\n    See Also\n    --------\n    scipy.integrate.quadrature\n    scipy.integrate.fixed_quad\n\n    References\n    ----------\n    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.\n        Handbook of Mathematical Functions with Formulas,\n        Graphs, and Mathematical Tables. New York: Dover, 1972.\n\n    '
    xw = roots_chebyt(n, mu)
    return ((xw[0] + 1) / 2,) + xw[1:]

def sh_chebyt(n, monic=False):
    if False:
        while True:
            i = 10
    'Shifted Chebyshev polynomial of the first kind.\n\n    Defined as :math:`T^*_n(x) = T_n(2x - 1)` for :math:`T_n` the nth\n    Chebyshev polynomial of the first kind.\n\n    Parameters\n    ----------\n    n : int\n        Degree of the polynomial.\n    monic : bool, optional\n        If `True`, scale the leading coefficient to be 1. Default is\n        `False`.\n\n    Returns\n    -------\n    T : orthopoly1d\n        Shifted Chebyshev polynomial of the first kind.\n\n    Notes\n    -----\n    The polynomials :math:`T^*_n` are orthogonal over :math:`[0, 1]`\n    with weight function :math:`(x - x^2)^{-1/2}`.\n\n    '
    base = sh_jacobi(n, 0.0, 0.5, monic=monic)
    if monic:
        return base
    if n > 0:
        factor = 4 ** n / 2.0
    else:
        factor = 1.0
    base._scale(factor)
    return base

def roots_sh_chebyu(n, mu=False):
    if False:
        for i in range(10):
            print('nop')
    'Gauss-Chebyshev (second kind, shifted) quadrature.\n\n    Computes the sample points and weights for Gauss-Chebyshev\n    quadrature. The sample points are the roots of the nth degree\n    shifted Chebyshev polynomial of the second kind, :math:`U_n(x)`.\n    These sample points and weights correctly integrate polynomials of\n    degree :math:`2n - 1` or less over the interval :math:`[0, 1]`\n    with weight function :math:`w(x) = \\sqrt{x - x^2}`. See 22.2.9 in\n    [AS]_ for more details.\n\n    Parameters\n    ----------\n    n : int\n        quadrature order\n    mu : bool, optional\n        If True, return the sum of the weights, optional.\n\n    Returns\n    -------\n    x : ndarray\n        Sample points\n    w : ndarray\n        Weights\n    mu : float\n        Sum of the weights\n\n    See Also\n    --------\n    scipy.integrate.quadrature\n    scipy.integrate.fixed_quad\n\n    References\n    ----------\n    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.\n        Handbook of Mathematical Functions with Formulas,\n        Graphs, and Mathematical Tables. New York: Dover, 1972.\n\n    '
    (x, w, m) = roots_chebyu(n, True)
    x = (x + 1) / 2
    m_us = _ufuncs.beta(1.5, 1.5)
    w *= m_us / m
    if mu:
        return (x, w, m_us)
    else:
        return (x, w)

def sh_chebyu(n, monic=False):
    if False:
        while True:
            i = 10
    'Shifted Chebyshev polynomial of the second kind.\n\n    Defined as :math:`U^*_n(x) = U_n(2x - 1)` for :math:`U_n` the nth\n    Chebyshev polynomial of the second kind.\n\n    Parameters\n    ----------\n    n : int\n        Degree of the polynomial.\n    monic : bool, optional\n        If `True`, scale the leading coefficient to be 1. Default is\n        `False`.\n\n    Returns\n    -------\n    U : orthopoly1d\n        Shifted Chebyshev polynomial of the second kind.\n\n    Notes\n    -----\n    The polynomials :math:`U^*_n` are orthogonal over :math:`[0, 1]`\n    with weight function :math:`(x - x^2)^{1/2}`.\n\n    '
    base = sh_jacobi(n, 2.0, 1.5, monic=monic)
    if monic:
        return base
    factor = 4 ** n
    base._scale(factor)
    return base

def roots_legendre(n, mu=False):
    if False:
        return 10
    "Gauss-Legendre quadrature.\n\n    Compute the sample points and weights for Gauss-Legendre\n    quadrature [GL]_. The sample points are the roots of the nth degree\n    Legendre polynomial :math:`P_n(x)`. These sample points and\n    weights correctly integrate polynomials of degree :math:`2n - 1`\n    or less over the interval :math:`[-1, 1]` with weight function\n    :math:`w(x) = 1`. See 2.2.10 in [AS]_ for more details.\n\n    Parameters\n    ----------\n    n : int\n        quadrature order\n    mu : bool, optional\n        If True, return the sum of the weights, optional.\n\n    Returns\n    -------\n    x : ndarray\n        Sample points\n    w : ndarray\n        Weights\n    mu : float\n        Sum of the weights\n\n    See Also\n    --------\n    scipy.integrate.quadrature\n    scipy.integrate.fixed_quad\n    numpy.polynomial.legendre.leggauss\n\n    References\n    ----------\n    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.\n        Handbook of Mathematical Functions with Formulas,\n        Graphs, and Mathematical Tables. New York: Dover, 1972.\n    .. [GL] Gauss-Legendre quadrature, Wikipedia,\n        https://en.wikipedia.org/wiki/Gauss%E2%80%93Legendre_quadrature\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from scipy.special import roots_legendre, eval_legendre\n    >>> roots, weights = roots_legendre(9)\n\n    ``roots`` holds the roots, and ``weights`` holds the weights for\n    Gauss-Legendre quadrature.\n\n    >>> roots\n    array([-0.96816024, -0.83603111, -0.61337143, -0.32425342,  0.        ,\n            0.32425342,  0.61337143,  0.83603111,  0.96816024])\n    >>> weights\n    array([0.08127439, 0.18064816, 0.2606107 , 0.31234708, 0.33023936,\n           0.31234708, 0.2606107 , 0.18064816, 0.08127439])\n\n    Verify that we have the roots by evaluating the degree 9 Legendre\n    polynomial at ``roots``.  All the values are approximately zero:\n\n    >>> eval_legendre(9, roots)\n    array([-8.88178420e-16, -2.22044605e-16,  1.11022302e-16,  1.11022302e-16,\n            0.00000000e+00, -5.55111512e-17, -1.94289029e-16,  1.38777878e-16,\n           -8.32667268e-17])\n\n    Here we'll show how the above values can be used to estimate the\n    integral from 1 to 2 of f(t) = t + 1/t with Gauss-Legendre\n    quadrature [GL]_.  First define the function and the integration\n    limits.\n\n    >>> def f(t):\n    ...    return t + 1/t\n    ...\n    >>> a = 1\n    >>> b = 2\n\n    We'll use ``integral(f(t), t=a, t=b)`` to denote the definite integral\n    of f from t=a to t=b.  The sample points in ``roots`` are from the\n    interval [-1, 1], so we'll rewrite the integral with the simple change\n    of variable::\n\n        x = 2/(b - a) * t - (a + b)/(b - a)\n\n    with inverse::\n\n        t = (b - a)/2 * x + (a + 2)/2\n\n    Then::\n\n        integral(f(t), a, b) =\n            (b - a)/2 * integral(f((b-a)/2*x + (a+b)/2), x=-1, x=1)\n\n    We can approximate the latter integral with the values returned\n    by `roots_legendre`.\n\n    Map the roots computed above from [-1, 1] to [a, b].\n\n    >>> t = (b - a)/2 * roots + (a + b)/2\n\n    Approximate the integral as the weighted sum of the function values.\n\n    >>> (b - a)/2 * f(t).dot(weights)\n    2.1931471805599276\n\n    Compare that to the exact result, which is 3/2 + log(2):\n\n    >>> 1.5 + np.log(2)\n    2.1931471805599454\n\n    "
    m = int(n)
    if n < 1 or n != m:
        raise ValueError('n must be a positive integer.')
    mu0 = 2.0

    def an_func(k):
        if False:
            print('Hello World!')
        return 0.0 * k

    def bn_func(k):
        if False:
            return 10
        return k * np.sqrt(1.0 / (4 * k * k - 1))
    f = _ufuncs.eval_legendre

    def df(n, x):
        if False:
            for i in range(10):
                print('nop')
        return (-n * x * _ufuncs.eval_legendre(n, x) + n * _ufuncs.eval_legendre(n - 1, x)) / (1 - x ** 2)
    return _gen_roots_and_weights(m, mu0, an_func, bn_func, f, df, True, mu)

def legendre(n, monic=False):
    if False:
        return 10
    'Legendre polynomial.\n\n    Defined to be the solution of\n\n    .. math::\n        \\frac{d}{dx}\\left[(1 - x^2)\\frac{d}{dx}P_n(x)\\right]\n          + n(n + 1)P_n(x) = 0;\n\n    :math:`P_n(x)` is a polynomial of degree :math:`n`.\n\n    Parameters\n    ----------\n    n : int\n        Degree of the polynomial.\n    monic : bool, optional\n        If `True`, scale the leading coefficient to be 1. Default is\n        `False`.\n\n    Returns\n    -------\n    P : orthopoly1d\n        Legendre polynomial.\n\n    Notes\n    -----\n    The polynomials :math:`P_n` are orthogonal over :math:`[-1, 1]`\n    with weight function 1.\n\n    Examples\n    --------\n    Generate the 3rd-order Legendre polynomial 1/2*(5x^3 + 0x^2 - 3x + 0):\n\n    >>> from scipy.special import legendre\n    >>> legendre(3)\n    poly1d([ 2.5,  0. , -1.5,  0. ])\n\n    '
    if n < 0:
        raise ValueError('n must be nonnegative.')
    if n == 0:
        n1 = n + 1
    else:
        n1 = n
    (x, w) = roots_legendre(n1)
    if n == 0:
        (x, w) = ([], [])
    hn = 2.0 / (2 * n + 1)
    kn = _gam(2 * n + 1) / _gam(n + 1) ** 2 / 2.0 ** n
    p = orthopoly1d(x, w, hn, kn, wfunc=lambda x: 1.0, limits=(-1, 1), monic=monic, eval_func=lambda x: _ufuncs.eval_legendre(n, x))
    return p

def roots_sh_legendre(n, mu=False):
    if False:
        i = 10
        return i + 15
    'Gauss-Legendre (shifted) quadrature.\n\n    Compute the sample points and weights for Gauss-Legendre\n    quadrature. The sample points are the roots of the nth degree\n    shifted Legendre polynomial :math:`P^*_n(x)`. These sample points\n    and weights correctly integrate polynomials of degree :math:`2n -\n    1` or less over the interval :math:`[0, 1]` with weight function\n    :math:`w(x) = 1.0`. See 2.2.11 in [AS]_ for details.\n\n    Parameters\n    ----------\n    n : int\n        quadrature order\n    mu : bool, optional\n        If True, return the sum of the weights, optional.\n\n    Returns\n    -------\n    x : ndarray\n        Sample points\n    w : ndarray\n        Weights\n    mu : float\n        Sum of the weights\n\n    See Also\n    --------\n    scipy.integrate.quadrature\n    scipy.integrate.fixed_quad\n\n    References\n    ----------\n    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.\n        Handbook of Mathematical Functions with Formulas,\n        Graphs, and Mathematical Tables. New York: Dover, 1972.\n\n    '
    (x, w) = roots_legendre(n)
    x = (x + 1) / 2
    w /= 2
    if mu:
        return (x, w, 1.0)
    else:
        return (x, w)

def sh_legendre(n, monic=False):
    if False:
        return 10
    'Shifted Legendre polynomial.\n\n    Defined as :math:`P^*_n(x) = P_n(2x - 1)` for :math:`P_n` the nth\n    Legendre polynomial.\n\n    Parameters\n    ----------\n    n : int\n        Degree of the polynomial.\n    monic : bool, optional\n        If `True`, scale the leading coefficient to be 1. Default is\n        `False`.\n\n    Returns\n    -------\n    P : orthopoly1d\n        Shifted Legendre polynomial.\n\n    Notes\n    -----\n    The polynomials :math:`P^*_n` are orthogonal over :math:`[0, 1]`\n    with weight function 1.\n\n    '
    if n < 0:
        raise ValueError('n must be nonnegative.')

    def wfunc(x):
        if False:
            while True:
                i = 10
        return 0.0 * x + 1.0
    if n == 0:
        return orthopoly1d([], [], 1.0, 1.0, wfunc, (0, 1), monic, lambda x: _ufuncs.eval_sh_legendre(n, x))
    (x, w) = roots_sh_legendre(n)
    hn = 1.0 / (2 * n + 1.0)
    kn = _gam(2 * n + 1) / _gam(n + 1) ** 2
    p = orthopoly1d(x, w, hn, kn, wfunc, limits=(0, 1), monic=monic, eval_func=lambda x: _ufuncs.eval_sh_legendre(n, x))
    return p
_modattrs = globals()
for (newfun, oldfun) in _rootfuns_map.items():
    _modattrs[oldfun] = _modattrs[newfun]
    __all__.append(oldfun)