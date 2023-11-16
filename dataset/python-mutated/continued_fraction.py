from __future__ import annotations
import itertools
from sympy.core.exprtools import factor_terms
from sympy.core.numbers import Integer, Rational
from sympy.core.singleton import S
from sympy.core.symbol import Dummy
from sympy.core.sympify import _sympify
from sympy.utilities.misc import as_int

def continued_fraction(a) -> list:
    if False:
        i = 10
        return i + 15
    'Return the continued fraction representation of a Rational or\n    quadratic irrational.\n\n    Examples\n    ========\n\n    >>> from sympy.ntheory.continued_fraction import continued_fraction\n    >>> from sympy import sqrt\n    >>> continued_fraction((1 + 2*sqrt(3))/5)\n    [0, 1, [8, 3, 34, 3]]\n\n    See Also\n    ========\n    continued_fraction_periodic, continued_fraction_reduce, continued_fraction_convergents\n    '
    e = _sympify(a)
    if all((i.is_Rational for i in e.atoms())):
        if e.is_Integer:
            return continued_fraction_periodic(e, 1, 0)
        elif e.is_Rational:
            return continued_fraction_periodic(e.p, e.q, 0)
        elif e.is_Pow and e.exp is S.Half and e.base.is_Integer:
            return continued_fraction_periodic(0, 1, e.base)
        elif e.is_Mul and len(e.args) == 2 and (e.args[0].is_Rational and e.args[1].is_Pow and e.args[1].base.is_Integer and (e.args[1].exp is S.Half)):
            (a, b) = e.args
            return continued_fraction_periodic(0, a.q, b.base, a.p)
        else:
            (p, d) = e.expand().as_numer_denom()
            if d.is_Integer:
                if p.is_Rational:
                    return continued_fraction_periodic(p, d)
                if p.is_Add and len(p.args) == 2:
                    (a, bc) = p.args
                else:
                    a = S.Zero
                    bc = p
                if a.is_Integer:
                    b = S.NaN
                    if bc.is_Mul and len(bc.args) == 2:
                        (b, c) = bc.args
                    elif bc.is_Pow:
                        b = Integer(1)
                        c = bc
                    if b.is_Integer and (c.is_Pow and c.exp is S.Half and c.base.is_Integer):
                        c = c.base
                        return continued_fraction_periodic(a, d, c, b)
    raise ValueError('expecting a rational or quadratic irrational, not %s' % e)

def continued_fraction_periodic(p, q, d=0, s=1) -> list:
    if False:
        for i in range(10):
            print('nop')
    "\n    Find the periodic continued fraction expansion of a quadratic irrational.\n\n    Compute the continued fraction expansion of a rational or a\n    quadratic irrational number, i.e. `\\frac{p + s\\sqrt{d}}{q}`, where\n    `p`, `q \\ne 0` and `d \\ge 0` are integers.\n\n    Returns the continued fraction representation (canonical form) as\n    a list of integers, optionally ending (for quadratic irrationals)\n    with list of integers representing the repeating digits.\n\n    Parameters\n    ==========\n\n    p : int\n        the rational part of the number's numerator\n    q : int\n        the denominator of the number\n    d : int, optional\n        the irrational part (discriminator) of the number's numerator\n    s : int, optional\n        the coefficient of the irrational part\n\n    Examples\n    ========\n\n    >>> from sympy.ntheory.continued_fraction import continued_fraction_periodic\n    >>> continued_fraction_periodic(3, 2, 7)\n    [2, [1, 4, 1, 1]]\n\n    Golden ratio has the simplest continued fraction expansion:\n\n    >>> continued_fraction_periodic(1, 2, 5)\n    [[1]]\n\n    If the discriminator is zero or a perfect square then the number will be a\n    rational number:\n\n    >>> continued_fraction_periodic(4, 3, 0)\n    [1, 3]\n    >>> continued_fraction_periodic(4, 3, 49)\n    [3, 1, 2]\n\n    See Also\n    ========\n\n    continued_fraction_iterator, continued_fraction_reduce\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Periodic_continued_fraction\n    .. [2] K. Rosen. Elementary Number theory and its applications.\n           Addison-Wesley, 3 Sub edition, pages 379-381, January 1992.\n\n    "
    from sympy.functions import sqrt, floor
    (p, q, d, s) = list(map(as_int, [p, q, d, s]))
    if d < 0:
        raise ValueError('expected non-negative for `d` but got %s' % d)
    if q == 0:
        raise ValueError('The denominator cannot be 0.')
    if not s:
        d = 0
    sd = sqrt(d)
    if sd.is_Integer:
        return list(continued_fraction_iterator(Rational(p + s * sd, q)))
    if q < 0:
        (p, q, s) = (-p, -q, -s)
    n = (p + s * sd) / q
    if n < 0:
        w = floor(-n)
        f = -n - w
        one_f = continued_fraction(1 - f)
        one_f[0] -= w + 1
        return one_f
    d *= s ** 2
    sd *= s
    if (d - p ** 2) % q:
        d *= q ** 2
        sd *= q
        p *= q
        q *= q
    terms: list[int] = []
    pq = {}
    while (p, q) not in pq:
        pq[p, q] = len(terms)
        terms.append((p + sd) // q)
        p = terms[-1] * q - p
        q = (d - p ** 2) // q
    i = pq[p, q]
    return terms[:i] + [terms[i:]]

def continued_fraction_reduce(cf):
    if False:
        return 10
    '\n    Reduce a continued fraction to a rational or quadratic irrational.\n\n    Compute the rational or quadratic irrational number from its\n    terminating or periodic continued fraction expansion.  The\n    continued fraction expansion (cf) should be supplied as a\n    terminating iterator supplying the terms of the expansion.  For\n    terminating continued fractions, this is equivalent to\n    ``list(continued_fraction_convergents(cf))[-1]``, only a little more\n    efficient.  If the expansion has a repeating part, a list of the\n    repeating terms should be returned as the last element from the\n    iterator.  This is the format returned by\n    continued_fraction_periodic.\n\n    For quadratic irrationals, returns the largest solution found,\n    which is generally the one sought, if the fraction is in canonical\n    form (all terms positive except possibly the first).\n\n    Examples\n    ========\n\n    >>> from sympy.ntheory.continued_fraction import continued_fraction_reduce\n    >>> continued_fraction_reduce([1, 2, 3, 4, 5])\n    225/157\n    >>> continued_fraction_reduce([-2, 1, 9, 7, 1, 2])\n    -256/233\n    >>> continued_fraction_reduce([2, 1, 2, 1, 1, 4, 1, 1, 6, 1, 1, 8]).n(10)\n    2.718281835\n    >>> continued_fraction_reduce([1, 4, 2, [3, 1]])\n    (sqrt(21) + 287)/238\n    >>> continued_fraction_reduce([[1]])\n    (1 + sqrt(5))/2\n    >>> from sympy.ntheory.continued_fraction import continued_fraction_periodic\n    >>> continued_fraction_reduce(continued_fraction_periodic(8, 5, 13))\n    (sqrt(13) + 8)/5\n\n    See Also\n    ========\n\n    continued_fraction_periodic\n\n    '
    from sympy.solvers import solve
    period = []
    x = Dummy('x')

    def untillist(cf):
        if False:
            while True:
                i = 10
        for nxt in cf:
            if isinstance(nxt, list):
                period.extend(nxt)
                yield x
                break
            yield nxt
    a = S.Zero
    for a in continued_fraction_convergents(untillist(cf)):
        pass
    if period:
        y = Dummy('y')
        solns = solve(continued_fraction_reduce(period + [y]) - y, y)
        solns.sort()
        pure = solns[-1]
        rv = a.subs(x, pure).radsimp()
    else:
        rv = a
    if rv.is_Add:
        rv = factor_terms(rv)
        if rv.is_Mul and rv.args[0] == -1:
            rv = rv.func(*rv.args)
    return rv

def continued_fraction_iterator(x):
    if False:
        print('Hello World!')
    '\n    Return continued fraction expansion of x as iterator.\n\n    Examples\n    ========\n\n    >>> from sympy import Rational, pi\n    >>> from sympy.ntheory.continued_fraction import continued_fraction_iterator\n\n    >>> list(continued_fraction_iterator(Rational(3, 8)))\n    [0, 2, 1, 2]\n    >>> list(continued_fraction_iterator(Rational(-3, 8)))\n    [-1, 1, 1, 1, 2]\n\n    >>> for i, v in enumerate(continued_fraction_iterator(pi)):\n    ...     if i > 7:\n    ...         break\n    ...     print(v)\n    3\n    7\n    15\n    1\n    292\n    1\n    1\n    1\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Continued_fraction\n\n    '
    from sympy.functions import floor
    while True:
        i = floor(x)
        yield i
        x -= i
        if not x:
            break
        x = 1 / x

def continued_fraction_convergents(cf):
    if False:
        return 10
    "\n    Return an iterator over the convergents of a continued fraction (cf).\n\n    The parameter should be in either of the following to forms:\n    - A list of partial quotients, possibly with the last element being a list\n    of repeating partial quotients, such as might be returned by\n    continued_fraction and continued_fraction_periodic.\n    - An iterable returning successive partial quotients of the continued\n    fraction, such as might be returned by continued_fraction_iterator.\n\n    In computing the convergents, the continued fraction need not be strictly\n    in canonical form (all integers, all but the first positive).\n    Rational and negative elements may be present in the expansion.\n\n    Examples\n    ========\n\n    >>> from sympy.core import pi\n    >>> from sympy import S\n    >>> from sympy.ntheory.continued_fraction import             continued_fraction_convergents, continued_fraction_iterator\n\n    >>> list(continued_fraction_convergents([0, 2, 1, 2]))\n    [0, 1/2, 1/3, 3/8]\n\n    >>> list(continued_fraction_convergents([1, S('1/2'), -7, S('1/4')]))\n    [1, 3, 19/5, 7]\n\n    >>> it = continued_fraction_convergents(continued_fraction_iterator(pi))\n    >>> for n in range(7):\n    ...     print(next(it))\n    3\n    22/7\n    333/106\n    355/113\n    103993/33102\n    104348/33215\n    208341/66317\n\n    >>> it = continued_fraction_convergents([1, [1, 2]])  # sqrt(3)\n    >>> for n in range(7):\n    ...     print(next(it))\n    1\n    2\n    5/3\n    7/4\n    19/11\n    26/15\n    71/41\n\n    See Also\n    ========\n\n    continued_fraction_iterator, continued_fraction, continued_fraction_periodic\n\n    "
    if isinstance(cf, list) and isinstance(cf[-1], list):
        cf = itertools.chain(cf[:-1], itertools.cycle(cf[-1]))
    (p_2, q_2) = (S.Zero, S.One)
    (p_1, q_1) = (S.One, S.Zero)
    for a in cf:
        (p, q) = (a * p_1 + p_2, a * q_1 + q_2)
        (p_2, q_2) = (p_1, q_1)
        (p_1, q_1) = (p, q)
        yield (p / q)