"""Efficient functions for generating orthogonal polynomials."""
from sympy.core.symbol import Dummy
from sympy.polys.densearith import dup_mul, dup_mul_ground, dup_lshift, dup_sub, dup_add
from sympy.polys.domains import ZZ, QQ
from sympy.polys.polytools import named_poly
from sympy.utilities import public

def dup_jacobi(n, a, b, K):
    if False:
        for i in range(10):
            print('nop')
    'Low-level implementation of Jacobi polynomials.'
    if n < 1:
        return [K.one]
    (m2, m1) = ([K.one], [(a + b) / K(2) + K.one, (a - b) / K(2)])
    for i in range(2, n + 1):
        den = K(i) * (a + b + i) * (a + b + K(2) * i - K(2))
        f0 = (a + b + K(2) * i - K.one) * (a * a - b * b) / (K(2) * den)
        f1 = (a + b + K(2) * i - K.one) * (a + b + K(2) * i - K(2)) * (a + b + K(2) * i) / (K(2) * den)
        f2 = (a + i - K.one) * (b + i - K.one) * (a + b + K(2) * i) / den
        p0 = dup_mul_ground(m1, f0, K)
        p1 = dup_mul_ground(dup_lshift(m1, 1, K), f1, K)
        p2 = dup_mul_ground(m2, f2, K)
        (m2, m1) = (m1, dup_sub(dup_add(p0, p1, K), p2, K))
    return m1

@public
def jacobi_poly(n, a, b, x=None, polys=False):
    if False:
        while True:
            i = 10
    'Generates the Jacobi polynomial `P_n^{(a,b)}(x)`.\n\n    Parameters\n    ==========\n\n    n : int\n        Degree of the polynomial.\n    a\n        Lower limit of minimal domain for the list of coefficients.\n    b\n        Upper limit of minimal domain for the list of coefficients.\n    x : optional\n    polys : bool, optional\n        If True, return a Poly, otherwise (default) return an expression.\n    '
    return named_poly(n, dup_jacobi, None, 'Jacobi polynomial', (x, a, b), polys)

def dup_gegenbauer(n, a, K):
    if False:
        for i in range(10):
            print('nop')
    'Low-level implementation of Gegenbauer polynomials.'
    if n < 1:
        return [K.one]
    (m2, m1) = ([K.one], [K(2) * a, K.zero])
    for i in range(2, n + 1):
        p1 = dup_mul_ground(dup_lshift(m1, 1, K), K(2) * (a - K.one) / K(i) + K(2), K)
        p2 = dup_mul_ground(m2, K(2) * (a - K.one) / K(i) + K.one, K)
        (m2, m1) = (m1, dup_sub(p1, p2, K))
    return m1

def gegenbauer_poly(n, a, x=None, polys=False):
    if False:
        print('Hello World!')
    'Generates the Gegenbauer polynomial `C_n^{(a)}(x)`.\n\n    Parameters\n    ==========\n\n    n : int\n        Degree of the polynomial.\n    x : optional\n    a\n        Decides minimal domain for the list of coefficients.\n    polys : bool, optional\n        If True, return a Poly, otherwise (default) return an expression.\n    '
    return named_poly(n, dup_gegenbauer, None, 'Gegenbauer polynomial', (x, a), polys)

def dup_chebyshevt(n, K):
    if False:
        print('Hello World!')
    'Low-level implementation of Chebyshev polynomials of the first kind.'
    if n < 1:
        return [K.one]
    (m2, m1) = ([K.one], [K.one, K.zero])
    for i in range(2, n + 1):
        (m2, m1) = (m1, dup_sub(dup_mul_ground(dup_lshift(m1, 1, K), K(2), K), m2, K))
    return m1

def dup_chebyshevu(n, K):
    if False:
        print('Hello World!')
    'Low-level implementation of Chebyshev polynomials of the second kind.'
    if n < 1:
        return [K.one]
    (m2, m1) = ([K.one], [K(2), K.zero])
    for i in range(2, n + 1):
        (m2, m1) = (m1, dup_sub(dup_mul_ground(dup_lshift(m1, 1, K), K(2), K), m2, K))
    return m1

@public
def chebyshevt_poly(n, x=None, polys=False):
    if False:
        while True:
            i = 10
    'Generates the Chebyshev polynomial of the first kind `T_n(x)`.\n\n    Parameters\n    ==========\n\n    n : int\n        Degree of the polynomial.\n    x : optional\n    polys : bool, optional\n        If True, return a Poly, otherwise (default) return an expression.\n    '
    return named_poly(n, dup_chebyshevt, ZZ, 'Chebyshev polynomial of the first kind', (x,), polys)

@public
def chebyshevu_poly(n, x=None, polys=False):
    if False:
        i = 10
        return i + 15
    'Generates the Chebyshev polynomial of the second kind `U_n(x)`.\n\n    Parameters\n    ==========\n\n    n : int\n        Degree of the polynomial.\n    x : optional\n    polys : bool, optional\n        If True, return a Poly, otherwise (default) return an expression.\n    '
    return named_poly(n, dup_chebyshevu, ZZ, 'Chebyshev polynomial of the second kind', (x,), polys)

def dup_hermite(n, K):
    if False:
        while True:
            i = 10
    'Low-level implementation of Hermite polynomials.'
    if n < 1:
        return [K.one]
    (m2, m1) = ([K.one], [K(2), K.zero])
    for i in range(2, n + 1):
        a = dup_lshift(m1, 1, K)
        b = dup_mul_ground(m2, K(i - 1), K)
        (m2, m1) = (m1, dup_mul_ground(dup_sub(a, b, K), K(2), K))
    return m1

def dup_hermite_prob(n, K):
    if False:
        return 10
    "Low-level implementation of probabilist's Hermite polynomials."
    if n < 1:
        return [K.one]
    (m2, m1) = ([K.one], [K.one, K.zero])
    for i in range(2, n + 1):
        a = dup_lshift(m1, 1, K)
        b = dup_mul_ground(m2, K(i - 1), K)
        (m2, m1) = (m1, dup_sub(a, b, K))
    return m1

@public
def hermite_poly(n, x=None, polys=False):
    if False:
        return 10
    'Generates the Hermite polynomial `H_n(x)`.\n\n    Parameters\n    ==========\n\n    n : int\n        Degree of the polynomial.\n    x : optional\n    polys : bool, optional\n        If True, return a Poly, otherwise (default) return an expression.\n    '
    return named_poly(n, dup_hermite, ZZ, 'Hermite polynomial', (x,), polys)

@public
def hermite_prob_poly(n, x=None, polys=False):
    if False:
        i = 10
        return i + 15
    "Generates the probabilist's Hermite polynomial `He_n(x)`.\n\n    Parameters\n    ==========\n\n    n : int\n        Degree of the polynomial.\n    x : optional\n    polys : bool, optional\n        If True, return a Poly, otherwise (default) return an expression.\n    "
    return named_poly(n, dup_hermite_prob, ZZ, "probabilist's Hermite polynomial", (x,), polys)

def dup_legendre(n, K):
    if False:
        i = 10
        return i + 15
    'Low-level implementation of Legendre polynomials.'
    if n < 1:
        return [K.one]
    (m2, m1) = ([K.one], [K.one, K.zero])
    for i in range(2, n + 1):
        a = dup_mul_ground(dup_lshift(m1, 1, K), K(2 * i - 1, i), K)
        b = dup_mul_ground(m2, K(i - 1, i), K)
        (m2, m1) = (m1, dup_sub(a, b, K))
    return m1

@public
def legendre_poly(n, x=None, polys=False):
    if False:
        while True:
            i = 10
    'Generates the Legendre polynomial `P_n(x)`.\n\n    Parameters\n    ==========\n\n    n : int\n        Degree of the polynomial.\n    x : optional\n    polys : bool, optional\n        If True, return a Poly, otherwise (default) return an expression.\n    '
    return named_poly(n, dup_legendre, QQ, 'Legendre polynomial', (x,), polys)

def dup_laguerre(n, alpha, K):
    if False:
        i = 10
        return i + 15
    'Low-level implementation of Laguerre polynomials.'
    (m2, m1) = ([K.zero], [K.one])
    for i in range(1, n + 1):
        a = dup_mul(m1, [-K.one / K(i), (alpha - K.one) / K(i) + K(2)], K)
        b = dup_mul_ground(m2, (alpha - K.one) / K(i) + K.one, K)
        (m2, m1) = (m1, dup_sub(a, b, K))
    return m1

@public
def laguerre_poly(n, x=None, alpha=0, polys=False):
    if False:
        for i in range(10):
            print('nop')
    'Generates the Laguerre polynomial `L_n^{(\\alpha)}(x)`.\n\n    Parameters\n    ==========\n\n    n : int\n        Degree of the polynomial.\n    x : optional\n    alpha : optional\n        Decides minimal domain for the list of coefficients.\n    polys : bool, optional\n        If True, return a Poly, otherwise (default) return an expression.\n    '
    return named_poly(n, dup_laguerre, None, 'Laguerre polynomial', (x, alpha), polys)

def dup_spherical_bessel_fn(n, K):
    if False:
        while True:
            i = 10
    'Low-level implementation of fn(n, x).'
    if n < 1:
        return [K.one, K.zero]
    (m2, m1) = ([K.one], [K.one, K.zero])
    for i in range(2, n + 1):
        (m2, m1) = (m1, dup_sub(dup_mul_ground(dup_lshift(m1, 1, K), K(2 * i - 1), K), m2, K))
    return dup_lshift(m1, 1, K)

def dup_spherical_bessel_fn_minus(n, K):
    if False:
        return 10
    'Low-level implementation of fn(-n, x).'
    (m2, m1) = ([K.one, K.zero], [K.zero])
    for i in range(2, n + 1):
        (m2, m1) = (m1, dup_sub(dup_mul_ground(dup_lshift(m1, 1, K), K(3 - 2 * i), K), m2, K))
    return m1

def spherical_bessel_fn(n, x=None, polys=False):
    if False:
        i = 10
        return i + 15
    '\n    Coefficients for the spherical Bessel functions.\n\n    These are only needed in the jn() function.\n\n    The coefficients are calculated from:\n\n    fn(0, z) = 1/z\n    fn(1, z) = 1/z**2\n    fn(n-1, z) + fn(n+1, z) == (2*n+1)/z * fn(n, z)\n\n    Parameters\n    ==========\n\n    n : int\n        Degree of the polynomial.\n    x : optional\n    polys : bool, optional\n        If True, return a Poly, otherwise (default) return an expression.\n\n    Examples\n    ========\n\n    >>> from sympy.polys.orthopolys import spherical_bessel_fn as fn\n    >>> from sympy import Symbol\n    >>> z = Symbol("z")\n    >>> fn(1, z)\n    z**(-2)\n    >>> fn(2, z)\n    -1/z + 3/z**3\n    >>> fn(3, z)\n    -6/z**2 + 15/z**4\n    >>> fn(4, z)\n    1/z - 45/z**3 + 105/z**5\n\n    '
    if x is None:
        x = Dummy('x')
    f = dup_spherical_bessel_fn_minus if n < 0 else dup_spherical_bessel_fn
    return named_poly(abs(n), f, ZZ, '', (QQ(1) / x,), polys)