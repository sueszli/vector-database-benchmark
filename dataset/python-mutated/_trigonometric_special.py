"""A module for special angle forumlas for trigonometric functions

TODO
====

This module should be developed in the future to contain direct squrae root
representation of

.. math
    F(\\frac{n}{m} \\pi)

for every

- $m \\in \\{ 3, 5, 17, 257, 65537 \\}$
- $n \\in \\mathbb{N}$, $0 \\le n < m$
- $F \\in \\{\\sin, \\cos, \\tan, \\csc, \\sec, \\cot\\}$

Without multi-step rewrites
(e.g. $\\tan \\to \\cos/\\sin \\to \\cos/\\sqrt \\to \\ sqrt$)
or using chebyshev identities
(e.g. $\\cos \\to \\cos + \\cos^2 + \\cdots \\to \\sqrt{} + \\sqrt{}^2 + \\cdots $),
which are trivial to implement in sympy,
and had used to give overly complicated expressions.

The reference can be found below, if anyone may need help implementing them.

References
==========

.. [*] Gottlieb, Christian. (1999). The Simple and straightforward construction
   of the regular 257-gon. The Mathematical Intelligencer. 21. 31-37.
   10.1007/BF03024829.
.. [*] https://resources.wolframcloud.com/FunctionRepository/resources/Cos2PiOverFermatPrime
"""
from __future__ import annotations
from typing import Callable
from functools import reduce
from sympy.core.expr import Expr
from sympy.core.singleton import S
from sympy.core.intfunc import igcdex
from sympy.core.numbers import Integer
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.core.cache import cacheit

def migcdex(*x: int) -> tuple[tuple[int, ...], int]:
    if False:
        i = 10
        return i + 15
    'Compute extended gcd for multiple integers.\n\n    Explanation\n    ===========\n\n    Given the integers $x_1, \\cdots, x_n$ and\n    an extended gcd for multiple arguments are defined as a solution\n    $(y_1, \\cdots, y_n), g$ for the diophantine equation\n    $x_1 y_1 + \\cdots + x_n y_n = g$ such that\n    $g = \\gcd(x_1, \\cdots, x_n)$.\n\n    Examples\n    ========\n\n    >>> from sympy.functions.elementary._trigonometric_special import migcdex\n    >>> migcdex()\n    ((), 0)\n    >>> migcdex(4)\n    ((1,), 4)\n    >>> migcdex(4, 6)\n    ((-1, 1), 2)\n    >>> migcdex(6, 10, 15)\n    ((1, 1, -1), 1)\n    '
    if not x:
        return ((), 0)
    if len(x) == 1:
        return ((1,), x[0])
    if len(x) == 2:
        (u, v, h) = igcdex(x[0], x[1])
        return ((u, v), h)
    (y, g) = migcdex(*x[1:])
    (u, v, h) = igcdex(x[0], g)
    return ((u, *(v * i for i in y)), h)

def ipartfrac(*denoms: int) -> tuple[int, ...]:
    if False:
        return 10
    'Compute the the partial fraction decomposition.\n\n    Explanation\n    ===========\n\n    Given a rational number $\\frac{1}{q_1 \\cdots q_n}$ where all\n    $q_1, \\cdots, q_n$ are pairwise coprime,\n\n    A partial fraction decomposition is defined as\n\n    .. math::\n        \\frac{1}{q_1 \\cdots q_n} = \\frac{p_1}{q_1} + \\cdots + \\frac{p_n}{q_n}\n\n    And it can be derived from solving the following diophantine equation for\n    the $p_1, \\cdots, p_n$\n\n    .. math::\n        1 = p_1 \\prod_{i \\ne 1}q_i + \\cdots + p_n \\prod_{i \\ne n}q_i\n\n    Where $q_1, \\cdots, q_n$ being pairwise coprime implies\n    $\\gcd(\\prod_{i \\ne 1}q_i, \\cdots, \\prod_{i \\ne n}q_i) = 1$,\n    which guarantees the existance of the solution.\n\n    It is sufficient to compute partial fraction decomposition only\n    for numerator $1$ because partial fraction decomposition for any\n    $\\frac{n}{q_1 \\cdots q_n}$ can be easily computed by multiplying\n    the result by $n$ afterwards.\n\n    Parameters\n    ==========\n\n    denoms : int\n        The pairwise coprime integer denominators $q_i$ which defines the\n        rational number $\\frac{1}{q_1 \\cdots q_n}$\n\n    Returns\n    =======\n\n    tuple[int, ...]\n        The list of numerators which semantically corresponds to $p_i$ of the\n        partial fraction decomposition\n        $\\frac{1}{q_1 \\cdots q_n} = \\frac{p_1}{q_1} + \\cdots + \\frac{p_n}{q_n}$\n\n    Examples\n    ========\n\n    >>> from sympy import Rational, Mul\n    >>> from sympy.functions.elementary._trigonometric_special import ipartfrac\n\n    >>> denoms = 2, 3, 5\n    >>> numers = ipartfrac(2, 3, 5)\n    >>> numers\n    (1, 7, -14)\n\n    >>> Rational(1, Mul(*denoms))\n    1/30\n    >>> out = 0\n    >>> for n, d in zip(numers, denoms):\n    ...    out += Rational(n, d)\n    >>> out\n    1/30\n    '
    if not denoms:
        return ()

    def mul(x: int, y: int) -> int:
        if False:
            return 10
        return x * y
    denom = reduce(mul, denoms)
    a = [denom // x for x in denoms]
    (h, _) = migcdex(*a)
    return h

def fermat_coords(n: int) -> list[int] | None:
    if False:
        while True:
            i = 10
    'If n can be factored in terms of Fermat primes with\n    multiplicity of each being 1, return those primes, else\n    None\n    '
    primes = []
    for p in [3, 5, 17, 257, 65537]:
        (quotient, remainder) = divmod(n, p)
        if remainder == 0:
            n = quotient
            primes.append(p)
            if n == 1:
                return primes
    return None

@cacheit
def cos_3() -> Expr:
    if False:
        print('Hello World!')
    'Computes $\\cos \\frac{\\pi}{3}$ in square roots'
    return S.Half

@cacheit
def cos_5() -> Expr:
    if False:
        for i in range(10):
            print('nop')
    'Computes $\\cos \\frac{\\pi}{5}$ in square roots'
    return (sqrt(5) + 1) / 4

@cacheit
def cos_17() -> Expr:
    if False:
        i = 10
        return i + 15
    'Computes $\\cos \\frac{\\pi}{17}$ in square roots'
    return sqrt((15 + sqrt(17)) / 32 + sqrt(2) * (sqrt(17 - sqrt(17)) + sqrt(sqrt(2) * (-8 * sqrt(17 + sqrt(17)) - (1 - sqrt(17)) * sqrt(17 - sqrt(17))) + 6 * sqrt(17) + 34)) / 32)

@cacheit
def cos_257() -> Expr:
    if False:
        while True:
            i = 10
    'Computes $\\cos \\frac{\\pi}{257}$ in square roots\n\n    References\n    ==========\n\n    .. [*] https://math.stackexchange.com/questions/516142/how-does-cos2-pi-257-look-like-in-real-radicals\n    .. [*] https://r-knott.surrey.ac.uk/Fibonacci/simpleTrig.html\n    '

    def f1(a: Expr, b: Expr) -> tuple[Expr, Expr]:
        if False:
            while True:
                i = 10
        return ((a + sqrt(a ** 2 + b)) / 2, (a - sqrt(a ** 2 + b)) / 2)

    def f2(a: Expr, b: Expr) -> Expr:
        if False:
            print('Hello World!')
        return (a - sqrt(a ** 2 + b)) / 2
    (t1, t2) = f1(S.NegativeOne, Integer(256))
    (z1, z3) = f1(t1, Integer(64))
    (z2, z4) = f1(t2, Integer(64))
    (y1, y5) = f1(z1, 4 * (5 + t1 + 2 * z1))
    (y6, y2) = f1(z2, 4 * (5 + t2 + 2 * z2))
    (y3, y7) = f1(z3, 4 * (5 + t1 + 2 * z3))
    (y8, y4) = f1(z4, 4 * (5 + t2 + 2 * z4))
    (x1, x9) = f1(y1, -4 * (t1 + y1 + y3 + 2 * y6))
    (x2, x10) = f1(y2, -4 * (t2 + y2 + y4 + 2 * y7))
    (x3, x11) = f1(y3, -4 * (t1 + y3 + y5 + 2 * y8))
    (x4, x12) = f1(y4, -4 * (t2 + y4 + y6 + 2 * y1))
    (x5, x13) = f1(y5, -4 * (t1 + y5 + y7 + 2 * y2))
    (x6, x14) = f1(y6, -4 * (t2 + y6 + y8 + 2 * y3))
    (x15, x7) = f1(y7, -4 * (t1 + y7 + y1 + 2 * y4))
    (x8, x16) = f1(y8, -4 * (t2 + y8 + y2 + 2 * y5))
    v1 = f2(x1, -4 * (x1 + x2 + x3 + x6))
    v2 = f2(x2, -4 * (x2 + x3 + x4 + x7))
    v3 = f2(x8, -4 * (x8 + x9 + x10 + x13))
    v4 = f2(x9, -4 * (x9 + x10 + x11 + x14))
    v5 = f2(x10, -4 * (x10 + x11 + x12 + x15))
    v6 = f2(x16, -4 * (x16 + x1 + x2 + x5))
    u1 = -f2(-v1, -4 * (v2 + v3))
    u2 = -f2(-v4, -4 * (v5 + v6))
    w1 = -2 * f2(-u1, -4 * u2)
    return sqrt(sqrt(2) * sqrt(w1 + 4) / 8 + S.Half)

def cos_table() -> dict[int, Callable[[], Expr]]:
    if False:
        for i in range(10):
            print('nop')
    'Lazily evaluated table for $\\cos \\frac{\\pi}{n}$ in square roots for\n    $n \\in \\{3, 5, 17, 257, 65537\\}$.\n\n    Notes\n    =====\n\n    65537 is the only other known Fermat prime and it is nearly impossible to\n    build in the current SymPy due to performance issues.\n\n    References\n    ==========\n\n    https://r-knott.surrey.ac.uk/Fibonacci/simpleTrig.html\n    '
    return {3: cos_3, 5: cos_5, 17: cos_17, 257: cos_257}