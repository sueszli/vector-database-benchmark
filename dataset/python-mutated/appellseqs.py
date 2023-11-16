"""
Efficient functions for generating Appell sequences.

An Appell sequence is a zero-indexed sequence of polynomials `p_i(x)`
satisfying `p_{i+1}'(x)=(i+1)p_i(x)` for all `i`. This definition leads
to the following iterative algorithm:

.. math :: p_0(x) = c_0,\\ p_i(x) = i \\int_0^x p_{i-1}(t)\\,dt + c_i

The constant coefficients `c_i` are usually determined from the
just-evaluated integral and `i`.

Appell sequences satisfy the following identity from umbral calculus:

.. math :: p_n(x+y) = \\sum_{k=0}^n \\binom{n}{k} p_k(x) y^{n-k}

References
==========

.. [1] https://en.wikipedia.org/wiki/Appell_sequence
.. [2] Peter Luschny, "An introduction to the Bernoulli function",
       https://arxiv.org/abs/2009.06743
"""
from sympy.polys.densearith import dup_mul_ground, dup_sub_ground, dup_quo_ground
from sympy.polys.densetools import dup_eval, dup_integrate
from sympy.polys.domains import ZZ, QQ
from sympy.polys.polytools import named_poly
from sympy.utilities import public

def dup_bernoulli(n, K):
    if False:
        for i in range(10):
            print('nop')
    'Low-level implementation of Bernoulli polynomials.'
    if n < 1:
        return [K.one]
    p = [K.one, K(-1, 2)]
    for i in range(2, n + 1):
        p = dup_integrate(dup_mul_ground(p, K(i), K), 1, K)
        if i % 2 == 0:
            p = dup_sub_ground(p, dup_eval(p, K(1, 2), K) * K(1 << i - 1, (1 << i) - 1), K)
    return p

@public
def bernoulli_poly(n, x=None, polys=False):
    if False:
        return 10
    "Generates the Bernoulli polynomial `\\operatorname{B}_n(x)`.\n\n    `\\operatorname{B}_n(x)` is the unique polynomial satisfying\n\n    .. math :: \\int_{x}^{x+1} \\operatorname{B}_n(t) \\,dt = x^n.\n\n    Based on this, we have for nonnegative integer `s` and integer\n    `a` and `b`\n\n    .. math :: \\sum_{k=a}^{b} k^s = \\frac{\\operatorname{B}_{s+1}(b+1) -\n            \\operatorname{B}_{s+1}(a)}{s+1}\n\n    which is related to Jakob Bernoulli's original motivation for introducing\n    the Bernoulli numbers, the values of these polynomials at `x = 1`.\n\n    Examples\n    ========\n\n    >>> from sympy import summation\n    >>> from sympy.abc import x\n    >>> from sympy.polys import bernoulli_poly\n    >>> bernoulli_poly(5, x)\n    x**5 - 5*x**4/2 + 5*x**3/3 - x/6\n\n    >>> def psum(p, a, b):\n    ...     return (bernoulli_poly(p+1,b+1) - bernoulli_poly(p+1,a)) / (p+1)\n    >>> psum(4, -6, 27)\n    3144337\n    >>> summation(x**4, (x, -6, 27))\n    3144337\n\n    >>> psum(1, 1, x).factor()\n    x*(x + 1)/2\n    >>> psum(2, 1, x).factor()\n    x*(x + 1)*(2*x + 1)/6\n    >>> psum(3, 1, x).factor()\n    x**2*(x + 1)**2/4\n\n    Parameters\n    ==========\n\n    n : int\n        Degree of the polynomial.\n    x : optional\n    polys : bool, optional\n        If True, return a Poly, otherwise (default) return an expression.\n\n    See Also\n    ========\n\n    sympy.functions.combinatorial.numbers.bernoulli\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Bernoulli_polynomials\n    "
    return named_poly(n, dup_bernoulli, QQ, 'Bernoulli polynomial', (x,), polys)

def dup_bernoulli_c(n, K):
    if False:
        for i in range(10):
            print('nop')
    'Low-level implementation of central Bernoulli polynomials.'
    p = [K.one]
    for i in range(1, n + 1):
        p = dup_integrate(dup_mul_ground(p, K(i), K), 1, K)
        if i % 2 == 0:
            p = dup_sub_ground(p, dup_eval(p, K.one, K) * K((1 << i - 1) - 1, (1 << i) - 1), K)
    return p

@public
def bernoulli_c_poly(n, x=None, polys=False):
    if False:
        while True:
            i = 10
    'Generates the central Bernoulli polynomial `\\operatorname{B}_n^c(x)`.\n\n    These are scaled and shifted versions of the plain Bernoulli polynomials,\n    done in such a way that `\\operatorname{B}_n^c(x)` is an even or odd function\n    for even or odd `n` respectively:\n\n    .. math :: \\operatorname{B}_n^c(x) = 2^n \\operatorname{B}_n\n            \\left(\\frac{x+1}{2}\\right)\n\n    Parameters\n    ==========\n\n    n : int\n        Degree of the polynomial.\n    x : optional\n    polys : bool, optional\n        If True, return a Poly, otherwise (default) return an expression.\n    '
    return named_poly(n, dup_bernoulli_c, QQ, 'central Bernoulli polynomial', (x,), polys)

def dup_genocchi(n, K):
    if False:
        while True:
            i = 10
    'Low-level implementation of Genocchi polynomials.'
    if n < 1:
        return [K.zero]
    p = [-K.one]
    for i in range(2, n + 1):
        p = dup_integrate(dup_mul_ground(p, K(i), K), 1, K)
        if i % 2 == 0:
            p = dup_sub_ground(p, dup_eval(p, K.one, K) // K(2), K)
    return p

@public
def genocchi_poly(n, x=None, polys=False):
    if False:
        for i in range(10):
            print('nop')
    'Generates the Genocchi polynomial `\\operatorname{G}_n(x)`.\n\n    `\\operatorname{G}_n(x)` is twice the difference between the plain and\n    central Bernoulli polynomials, so has degree `n-1`:\n\n    .. math :: \\operatorname{G}_n(x) = 2 (\\operatorname{B}_n(x) -\n            \\operatorname{B}_n^c(x))\n\n    The factor of 2 in the definition endows `\\operatorname{G}_n(x)` with\n    integer coefficients.\n\n    Parameters\n    ==========\n\n    n : int\n        Degree of the polynomial plus one.\n    x : optional\n    polys : bool, optional\n        If True, return a Poly, otherwise (default) return an expression.\n\n    See Also\n    ========\n\n    sympy.functions.combinatorial.numbers.genocchi\n    '
    return named_poly(n, dup_genocchi, ZZ, 'Genocchi polynomial', (x,), polys)

def dup_euler(n, K):
    if False:
        while True:
            i = 10
    'Low-level implementation of Euler polynomials.'
    return dup_quo_ground(dup_genocchi(n + 1, ZZ), K(-n - 1), K)

@public
def euler_poly(n, x=None, polys=False):
    if False:
        while True:
            i = 10
    'Generates the Euler polynomial `\\operatorname{E}_n(x)`.\n\n    These are scaled and reindexed versions of the Genocchi polynomials:\n\n    .. math :: \\operatorname{E}_n(x) = -\\frac{\\operatorname{G}_{n+1}(x)}{n+1}\n\n    Parameters\n    ==========\n\n    n : int\n        Degree of the polynomial.\n    x : optional\n    polys : bool, optional\n        If True, return a Poly, otherwise (default) return an expression.\n\n    See Also\n    ========\n\n    sympy.functions.combinatorial.numbers.euler\n    '
    return named_poly(n, dup_euler, QQ, 'Euler polynomial', (x,), polys)

def dup_andre(n, K):
    if False:
        i = 10
        return i + 15
    'Low-level implementation of Andre polynomials.'
    p = [K.one]
    for i in range(1, n + 1):
        p = dup_integrate(dup_mul_ground(p, K(i), K), 1, K)
        if i % 2 == 0:
            p = dup_sub_ground(p, dup_eval(p, K.one, K), K)
    return p

@public
def andre_poly(n, x=None, polys=False):
    if False:
        print('Hello World!')
    'Generates the Andre polynomial `\\mathcal{A}_n(x)`.\n\n    This is the Appell sequence where the constant coefficients form the sequence\n    of Euler numbers ``euler(n)``. As such they have integer coefficients\n    and parities matching the parity of `n`.\n\n    Luschny calls these the *Swiss-knife polynomials* because their values\n    at 0 and 1 can be simply transformed into both the Bernoulli and Euler\n    numbers. Here they are called the Andre polynomials because\n    `|\\mathcal{A}_n(n\\bmod 2)|` for `n \\ge 0` generates what Luschny calls\n    the *Andre numbers*, A000111 in the OEIS.\n\n    Examples\n    ========\n\n    >>> from sympy import bernoulli, euler, genocchi\n    >>> from sympy.abc import x\n    >>> from sympy.polys import andre_poly\n    >>> andre_poly(9, x)\n    x**9 - 36*x**7 + 630*x**5 - 5124*x**3 + 12465*x\n\n    >>> [andre_poly(n, 0) for n in range(11)]\n    [1, 0, -1, 0, 5, 0, -61, 0, 1385, 0, -50521]\n    >>> [euler(n) for n in range(11)]\n    [1, 0, -1, 0, 5, 0, -61, 0, 1385, 0, -50521]\n    >>> [andre_poly(n-1, 1) * n / (4**n - 2**n) for n in range(1, 11)]\n    [1/2, 1/6, 0, -1/30, 0, 1/42, 0, -1/30, 0, 5/66]\n    >>> [bernoulli(n) for n in range(1, 11)]\n    [1/2, 1/6, 0, -1/30, 0, 1/42, 0, -1/30, 0, 5/66]\n    >>> [-andre_poly(n-1, -1) * n / (-2)**(n-1) for n in range(1, 11)]\n    [-1, -1, 0, 1, 0, -3, 0, 17, 0, -155]\n    >>> [genocchi(n) for n in range(1, 11)]\n    [-1, -1, 0, 1, 0, -3, 0, 17, 0, -155]\n\n    >>> [abs(andre_poly(n, n%2)) for n in range(11)]\n    [1, 1, 1, 2, 5, 16, 61, 272, 1385, 7936, 50521]\n\n    Parameters\n    ==========\n\n    n : int\n        Degree of the polynomial.\n    x : optional\n    polys : bool, optional\n        If True, return a Poly, otherwise (default) return an expression.\n\n    See Also\n    ========\n\n    sympy.functions.combinatorial.numbers.andre\n\n    References\n    ==========\n\n    .. [1] Peter Luschny, "An introduction to the Bernoulli function",\n           https://arxiv.org/abs/2009.06743\n    '
    return named_poly(n, dup_andre, ZZ, 'Andre polynomial', (x,), polys)