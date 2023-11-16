"""Power series evaluation and manipulation using sparse Polynomials

Implementing a new function
---------------------------

There are a few things to be kept in mind when adding a new function here::

    - The implementation should work on all possible input domains/rings.
      Special cases include the ``EX`` ring and a constant term in the series
      to be expanded. There can be two types of constant terms in the series:

        + A constant value or symbol.
        + A term of a multivariate series not involving the generator, with
          respect to which the series is to expanded.

      Strictly speaking, a generator of a ring should not be considered a
      constant. However, for series expansion both the cases need similar
      treatment (as the user does not care about inner details), i.e, use an
      addition formula to separate the constant part and the variable part (see
      rs_sin for reference).

    - All the algorithms used here are primarily designed to work for Taylor
      series (number of iterations in the algo equals the required order).
      Hence, it becomes tricky to get the series of the right order if a
      Puiseux series is input. Use rs_puiseux? in your function if your
      algorithm is not designed to handle fractional powers.

Extending rs_series
-------------------

To make a function work with rs_series you need to do two things::

    - Many sure it works with a constant term (as explained above).
    - If the series contains constant terms, you might need to extend its ring.
      You do so by adding the new terms to the rings as generators.
      ``PolyRing.compose`` and ``PolyRing.add_gens`` are two functions that do
      so and need to be called every time you expand a series containing a
      constant term.

Look at rs_sin and rs_series for further reference.

"""
from sympy.polys.domains import QQ, EX
from sympy.polys.rings import PolyElement, ring, sring
from sympy.polys.polyerrors import DomainError
from sympy.polys.monomials import monomial_min, monomial_mul, monomial_div, monomial_ldiv
from mpmath.libmp.libintmath import ifac
from sympy.core import PoleError, Function, Expr
from sympy.core.numbers import Rational
from sympy.core.intfunc import igcd
from sympy.functions import sin, cos, tan, atan, exp, atanh, tanh, log, ceiling
from sympy.utilities.misc import as_int
from mpmath.libmp.libintmath import giant_steps
import math

def _invert_monoms(p1):
    if False:
        while True:
            i = 10
    "\n    Compute ``x**n * p1(1/x)`` for a univariate polynomial ``p1`` in ``x``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys.domains import ZZ\n    >>> from sympy.polys.rings import ring\n    >>> from sympy.polys.ring_series import _invert_monoms\n    >>> R, x = ring('x', ZZ)\n    >>> p = x**2 + 2*x + 3\n    >>> _invert_monoms(p)\n    3*x**2 + 2*x + 1\n\n    See Also\n    ========\n\n    sympy.polys.densebasic.dup_reverse\n    "
    terms = list(p1.items())
    terms.sort()
    deg = p1.degree()
    R = p1.ring
    p = R.zero
    cv = p1.listcoeffs()
    mv = p1.listmonoms()
    for (mvi, cvi) in zip(mv, cv):
        p[deg - mvi[0],] = cvi
    return p

def _giant_steps(target):
    if False:
        return 10
    "Return a list of precision steps for the Newton's method"
    res = giant_steps(2, target)
    if res[0] != 2:
        res = [2] + res
    return res

def rs_trunc(p1, x, prec):
    if False:
        print('Hello World!')
    "\n    Truncate the series in the ``x`` variable with precision ``prec``,\n    that is, modulo ``O(x**prec)``\n\n    Examples\n    ========\n\n    >>> from sympy.polys.domains import QQ\n    >>> from sympy.polys.rings import ring\n    >>> from sympy.polys.ring_series import rs_trunc\n    >>> R, x = ring('x', QQ)\n    >>> p = x**10 + x**5 + x + 1\n    >>> rs_trunc(p, x, 12)\n    x**10 + x**5 + x + 1\n    >>> rs_trunc(p, x, 10)\n    x**5 + x + 1\n    "
    R = p1.ring
    p = R.zero
    i = R.gens.index(x)
    for exp1 in p1:
        if exp1[i] >= prec:
            continue
        p[exp1] = p1[exp1]
    return p

def rs_is_puiseux(p, x):
    if False:
        i = 10
        return i + 15
    "\n    Test if ``p`` is Puiseux series in ``x``.\n\n    Raise an exception if it has a negative power in ``x``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys.domains import QQ\n    >>> from sympy.polys.rings import ring\n    >>> from sympy.polys.ring_series import rs_is_puiseux\n    >>> R, x = ring('x', QQ)\n    >>> p = x**QQ(2,5) + x**QQ(2,3) + x\n    >>> rs_is_puiseux(p, x)\n    True\n    "
    index = p.ring.gens.index(x)
    for k in p:
        if k[index] != int(k[index]):
            return True
        if k[index] < 0:
            raise ValueError('The series is not regular in %s' % x)
    return False

def rs_puiseux(f, p, x, prec):
    if False:
        for i in range(10):
            print('nop')
    "\n    Return the puiseux series for `f(p, x, prec)`.\n\n    To be used when function ``f`` is implemented only for regular series.\n\n    Examples\n    ========\n\n    >>> from sympy.polys.domains import QQ\n    >>> from sympy.polys.rings import ring\n    >>> from sympy.polys.ring_series import rs_puiseux, rs_exp\n    >>> R, x = ring('x', QQ)\n    >>> p = x**QQ(2,5) + x**QQ(2,3) + x\n    >>> rs_puiseux(rs_exp,p, x, 1)\n    1/2*x**(4/5) + x**(2/3) + x**(2/5) + 1\n    "
    index = p.ring.gens.index(x)
    n = 1
    for k in p:
        power = k[index]
        if isinstance(power, Rational):
            (num, den) = power.as_numer_denom()
            n = int(n * den // igcd(n, den))
        elif power != int(power):
            den = power.denominator
            n = int(n * den // igcd(n, den))
    if n != 1:
        p1 = pow_xin(p, index, n)
        r = f(p1, x, prec * n)
        n1 = QQ(1, n)
        if isinstance(r, tuple):
            r = tuple([pow_xin(rx, index, n1) for rx in r])
        else:
            r = pow_xin(r, index, n1)
    else:
        r = f(p, x, prec)
    return r

def rs_puiseux2(f, p, q, x, prec):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return the puiseux series for `f(p, q, x, prec)`.\n\n    To be used when function ``f`` is implemented only for regular series.\n    '
    index = p.ring.gens.index(x)
    n = 1
    for k in p:
        power = k[index]
        if isinstance(power, Rational):
            (num, den) = power.as_numer_denom()
            n = n * den // igcd(n, den)
        elif power != int(power):
            den = power.denominator
            n = n * den // igcd(n, den)
    if n != 1:
        p1 = pow_xin(p, index, n)
        r = f(p1, q, x, prec * n)
        n1 = QQ(1, n)
        r = pow_xin(r, index, n1)
    else:
        r = f(p, q, x, prec)
    return r

def rs_mul(p1, p2, x, prec):
    if False:
        i = 10
        return i + 15
    "\n    Return the product of the given two series, modulo ``O(x**prec)``.\n\n    ``x`` is the series variable or its position in the generators.\n\n    Examples\n    ========\n\n    >>> from sympy.polys.domains import QQ\n    >>> from sympy.polys.rings import ring\n    >>> from sympy.polys.ring_series import rs_mul\n    >>> R, x = ring('x', QQ)\n    >>> p1 = x**2 + 2*x + 1\n    >>> p2 = x + 1\n    >>> rs_mul(p1, p2, x, 3)\n    3*x**2 + 3*x + 1\n    "
    R = p1.ring
    p = R.zero
    if R.__class__ != p2.ring.__class__ or R != p2.ring:
        raise ValueError('p1 and p2 must have the same ring')
    iv = R.gens.index(x)
    if not isinstance(p2, PolyElement):
        raise ValueError('p2 must be a polynomial')
    if R == p2.ring:
        get = p.get
        items2 = list(p2.items())
        items2.sort(key=lambda e: e[0][iv])
        if R.ngens == 1:
            for (exp1, v1) in p1.items():
                for (exp2, v2) in items2:
                    exp = exp1[0] + exp2[0]
                    if exp < prec:
                        exp = (exp,)
                        p[exp] = get(exp, 0) + v1 * v2
                    else:
                        break
        else:
            monomial_mul = R.monomial_mul
            for (exp1, v1) in p1.items():
                for (exp2, v2) in items2:
                    if exp1[iv] + exp2[iv] < prec:
                        exp = monomial_mul(exp1, exp2)
                        p[exp] = get(exp, 0) + v1 * v2
                    else:
                        break
    p.strip_zero()
    return p

def rs_square(p1, x, prec):
    if False:
        i = 10
        return i + 15
    "\n    Square the series modulo ``O(x**prec)``\n\n    Examples\n    ========\n\n    >>> from sympy.polys.domains import QQ\n    >>> from sympy.polys.rings import ring\n    >>> from sympy.polys.ring_series import rs_square\n    >>> R, x = ring('x', QQ)\n    >>> p = x**2 + 2*x + 1\n    >>> rs_square(p, x, 3)\n    6*x**2 + 4*x + 1\n    "
    R = p1.ring
    p = R.zero
    iv = R.gens.index(x)
    get = p.get
    items = list(p1.items())
    items.sort(key=lambda e: e[0][iv])
    monomial_mul = R.monomial_mul
    for i in range(len(items)):
        (exp1, v1) = items[i]
        for j in range(i):
            (exp2, v2) = items[j]
            if exp1[iv] + exp2[iv] < prec:
                exp = monomial_mul(exp1, exp2)
                p[exp] = get(exp, 0) + v1 * v2
            else:
                break
    p = p.imul_num(2)
    get = p.get
    for (expv, v) in p1.items():
        if 2 * expv[iv] < prec:
            e2 = monomial_mul(expv, expv)
            p[e2] = get(e2, 0) + v ** 2
    p.strip_zero()
    return p

def rs_pow(p1, n, x, prec):
    if False:
        while True:
            i = 10
    "\n    Return ``p1**n`` modulo ``O(x**prec)``\n\n    Examples\n    ========\n\n    >>> from sympy.polys.domains import QQ\n    >>> from sympy.polys.rings import ring\n    >>> from sympy.polys.ring_series import rs_pow\n    >>> R, x = ring('x', QQ)\n    >>> p = x + 1\n    >>> rs_pow(p, 4, x, 3)\n    6*x**2 + 4*x + 1\n    "
    R = p1.ring
    if isinstance(n, Rational):
        np = int(n.p)
        nq = int(n.q)
        if nq != 1:
            res = rs_nth_root(p1, nq, x, prec)
            if np != 1:
                res = rs_pow(res, np, x, prec)
        else:
            res = rs_pow(p1, np, x, prec)
        return res
    n = as_int(n)
    if n == 0:
        if p1:
            return R(1)
        else:
            raise ValueError('0**0 is undefined')
    if n < 0:
        p1 = rs_pow(p1, -n, x, prec)
        return rs_series_inversion(p1, x, prec)
    if n == 1:
        return rs_trunc(p1, x, prec)
    if n == 2:
        return rs_square(p1, x, prec)
    if n == 3:
        p2 = rs_square(p1, x, prec)
        return rs_mul(p1, p2, x, prec)
    p = R(1)
    while 1:
        if n & 1:
            p = rs_mul(p1, p, x, prec)
            n -= 1
            if not n:
                break
        p1 = rs_square(p1, x, prec)
        n = n // 2
    return p

def rs_subs(p, rules, x, prec):
    if False:
        while True:
            i = 10
    "\n    Substitution with truncation according to the mapping in ``rules``.\n\n    Return a series with precision ``prec`` in the generator ``x``\n\n    Note that substitutions are not done one after the other\n\n    >>> from sympy.polys.domains import QQ\n    >>> from sympy.polys.rings import ring\n    >>> from sympy.polys.ring_series import rs_subs\n    >>> R, x, y = ring('x, y', QQ)\n    >>> p = x**2 + y**2\n    >>> rs_subs(p, {x: x+ y, y: x+ 2*y}, x, 3)\n    2*x**2 + 6*x*y + 5*y**2\n    >>> (x + y)**2 + (x + 2*y)**2\n    2*x**2 + 6*x*y + 5*y**2\n\n    which differs from\n\n    >>> rs_subs(rs_subs(p, {x: x+ y}, x, 3), {y: x+ 2*y}, x, 3)\n    5*x**2 + 12*x*y + 8*y**2\n\n    Parameters\n    ----------\n    p : :class:`~.PolyElement` Input series.\n    rules : ``dict`` with substitution mappings.\n    x : :class:`~.PolyElement` in which the series truncation is to be done.\n    prec : :class:`~.Integer` order of the series after truncation.\n\n    Examples\n    ========\n\n    >>> from sympy.polys.domains import QQ\n    >>> from sympy.polys.rings import ring\n    >>> from sympy.polys.ring_series import rs_subs\n    >>> R, x, y = ring('x, y', QQ)\n    >>> rs_subs(x**2+y**2, {y: (x+y)**2}, x, 3)\n     6*x**2*y**2 + x**2 + 4*x*y**3 + y**4\n    "
    R = p.ring
    ngens = R.ngens
    d = R(0)
    for i in range(ngens):
        d[i, 1] = R.gens[i]
    for var in rules:
        d[R.index(var), 1] = rules[var]
    p1 = R(0)
    p_keys = sorted(p.keys())
    for expv in p_keys:
        p2 = R(1)
        for i in range(ngens):
            power = expv[i]
            if power == 0:
                continue
            if (i, power) not in d:
                (q, r) = divmod(power, 2)
                if r == 0 and (i, q) in d:
                    d[i, power] = rs_square(d[i, q], x, prec)
                elif (i, power - 1) in d:
                    d[i, power] = rs_mul(d[i, power - 1], d[i, 1], x, prec)
                else:
                    d[i, power] = rs_pow(d[i, 1], power, x, prec)
            p2 = rs_mul(p2, d[i, power], x, prec)
        p1 += p2 * p[expv]
    return p1

def _has_constant_term(p, x):
    if False:
        for i in range(10):
            print('nop')
    "\n    Check if ``p`` has a constant term in ``x``\n\n    Examples\n    ========\n\n    >>> from sympy.polys.domains import QQ\n    >>> from sympy.polys.rings import ring\n    >>> from sympy.polys.ring_series import _has_constant_term\n    >>> R, x = ring('x', QQ)\n    >>> p = x**2 + x + 1\n    >>> _has_constant_term(p, x)\n    True\n    "
    R = p.ring
    iv = R.gens.index(x)
    zm = R.zero_monom
    a = [0] * R.ngens
    a[iv] = 1
    miv = tuple(a)
    for expv in p:
        if monomial_min(expv, miv) == zm:
            return True
    return False

def _get_constant_term(p, x):
    if False:
        return 10
    'Return constant term in p with respect to x\n\n    Note that it is not simply `p[R.zero_monom]` as there might be multiple\n    generators in the ring R. We want the `x`-free term which can contain other\n    generators.\n    '
    R = p.ring
    i = R.gens.index(x)
    zm = R.zero_monom
    a = [0] * R.ngens
    a[i] = 1
    miv = tuple(a)
    c = 0
    for expv in p:
        if monomial_min(expv, miv) == zm:
            c += R({expv: p[expv]})
    return c

def _check_series_var(p, x, name):
    if False:
        while True:
            i = 10
    index = p.ring.gens.index(x)
    m = min(p, key=lambda k: k[index])[index]
    if m < 0:
        raise PoleError('Asymptotic expansion of %s around [oo] not implemented.' % name)
    return (index, m)

def _series_inversion1(p, x, prec):
    if False:
        while True:
            i = 10
    "\n    Univariate series inversion ``1/p`` modulo ``O(x**prec)``.\n\n    The Newton method is used.\n\n    Examples\n    ========\n\n    >>> from sympy.polys.domains import QQ\n    >>> from sympy.polys.rings import ring\n    >>> from sympy.polys.ring_series import _series_inversion1\n    >>> R, x = ring('x', QQ)\n    >>> p = x + 1\n    >>> _series_inversion1(p, x, 4)\n    -x**3 + x**2 - x + 1\n    "
    if rs_is_puiseux(p, x):
        return rs_puiseux(_series_inversion1, p, x, prec)
    R = p.ring
    zm = R.zero_monom
    c = p[zm]
    if prec == int(prec):
        prec = int(prec)
    if zm not in p:
        raise ValueError('No constant term in series')
    if _has_constant_term(p - c, x):
        raise ValueError('p cannot contain a constant term depending on parameters')
    one = R(1)
    if R.domain is EX:
        one = 1
    if c != one:
        p1 = R(1) / c
    else:
        p1 = R(1)
    for precx in _giant_steps(prec):
        t = 1 - rs_mul(p1, p, x, precx)
        p1 = p1 + rs_mul(p1, t, x, precx)
    return p1

def rs_series_inversion(p, x, prec):
    if False:
        for i in range(10):
            print('nop')
    "\n    Multivariate series inversion ``1/p`` modulo ``O(x**prec)``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys.domains import QQ\n    >>> from sympy.polys.rings import ring\n    >>> from sympy.polys.ring_series import rs_series_inversion\n    >>> R, x, y = ring('x, y', QQ)\n    >>> rs_series_inversion(1 + x*y**2, x, 4)\n    -x**3*y**6 + x**2*y**4 - x*y**2 + 1\n    >>> rs_series_inversion(1 + x*y**2, y, 4)\n    -x*y**2 + 1\n    >>> rs_series_inversion(x + x**2, x, 4)\n    x**3 - x**2 + x - 1 + x**(-1)\n    "
    R = p.ring
    if p == R.zero:
        raise ZeroDivisionError
    zm = R.zero_monom
    index = R.gens.index(x)
    m = min(p, key=lambda k: k[index])[index]
    if m:
        p = mul_xin(p, index, -m)
        prec = prec + m
    if zm not in p:
        raise NotImplementedError('No constant term in series')
    if _has_constant_term(p - p[zm], x):
        raise NotImplementedError('p - p[0] must not have a constant term in the series variables')
    r = _series_inversion1(p, x, prec)
    if m != 0:
        r = mul_xin(r, index, -m)
    return r

def _coefficient_t(p, t):
    if False:
        return 10
    'Coefficient of `x_i**j` in p, where ``t`` = (i, j)'
    (i, j) = t
    R = p.ring
    expv1 = [0] * R.ngens
    expv1[i] = j
    expv1 = tuple(expv1)
    p1 = R(0)
    for expv in p:
        if expv[i] == j:
            p1[monomial_div(expv, expv1)] = p[expv]
    return p1

def rs_series_reversion(p, x, n, y):
    if False:
        print('Hello World!')
    "\n    Reversion of a series.\n\n    ``p`` is a series with ``O(x**n)`` of the form $p = ax + f(x)$\n    where $a$ is a number different from 0.\n\n    $f(x) = \\sum_{k=2}^{n-1} a_kx_k$\n\n    Parameters\n    ==========\n\n      a_k : Can depend polynomially on other variables, not indicated.\n      x : Variable with name x.\n      y : Variable with name y.\n\n    Returns\n    =======\n\n    Solve $p = y$, that is, given $ax + f(x) - y = 0$,\n    find the solution $x = r(y)$ up to $O(y^n)$.\n\n    Algorithm\n    =========\n\n    If $r_i$ is the solution at order $i$, then:\n    $ar_i + f(r_i) - y = O\\left(y^{i + 1}\\right)$\n\n    and if $r_{i + 1}$ is the solution at order $i + 1$, then:\n    $ar_{i + 1} + f(r_{i + 1}) - y = O\\left(y^{i + 2}\\right)$\n\n    We have, $r_{i + 1} = r_i + e$, such that,\n    $ae + f(r_i) = O\\left(y^{i + 2}\\right)$\n    or $e = -f(r_i)/a$\n\n    So we use the recursion relation:\n    $r_{i + 1} = r_i - f(r_i)/a$\n    with the boundary condition: $r_1 = y$\n\n    Examples\n    ========\n\n    >>> from sympy.polys.domains import QQ\n    >>> from sympy.polys.rings import ring\n    >>> from sympy.polys.ring_series import rs_series_reversion, rs_trunc\n    >>> R, x, y, a, b = ring('x, y, a, b', QQ)\n    >>> p = x - x**2 - 2*b*x**2 + 2*a*b*x**2\n    >>> p1 = rs_series_reversion(p, x, 3, y); p1\n    -2*y**2*a*b + 2*y**2*b + y**2 + y\n    >>> rs_trunc(p.compose(x, p1), y, 3)\n    y\n    "
    if rs_is_puiseux(p, x):
        raise NotImplementedError
    R = p.ring
    nx = R.gens.index(x)
    y = R(y)
    ny = R.gens.index(y)
    if _has_constant_term(p, x):
        raise ValueError('p must not contain a constant term in the series variable')
    a = _coefficient_t(p, (nx, 1))
    zm = R.zero_monom
    assert zm in a and len(a) == 1
    a = a[zm]
    r = y / a
    for i in range(2, n):
        sp = rs_subs(p, {x: r}, y, i + 1)
        sp = _coefficient_t(sp, (ny, i)) * y ** i
        r -= sp / a
    return r

def rs_series_from_list(p, c, x, prec, concur=1):
    if False:
        return 10
    "\n    Return a series `sum c[n]*p**n` modulo `O(x**prec)`.\n\n    It reduces the number of multiplications by summing concurrently.\n\n    `ax = [1, p, p**2, .., p**(J - 1)]`\n    `s = sum(c[i]*ax[i]` for i in `range(r, (r + 1)*J))*p**((K - 1)*J)`\n    with `K >= (n + 1)/J`\n\n    Examples\n    ========\n\n    >>> from sympy.polys.domains import QQ\n    >>> from sympy.polys.rings import ring\n    >>> from sympy.polys.ring_series import rs_series_from_list, rs_trunc\n    >>> R, x = ring('x', QQ)\n    >>> p = x**2 + x + 1\n    >>> c = [1, 2, 3]\n    >>> rs_series_from_list(p, c, x, 4)\n    6*x**3 + 11*x**2 + 8*x + 6\n    >>> rs_trunc(1 + 2*p + 3*p**2, x, 4)\n    6*x**3 + 11*x**2 + 8*x + 6\n    >>> pc = R.from_list(list(reversed(c)))\n    >>> rs_trunc(pc.compose(x, p), x, 4)\n    6*x**3 + 11*x**2 + 8*x + 6\n\n    "
    '\n    See Also\n    ========\n\n    sympy.polys.rings.PolyRing.compose\n\n    '
    R = p.ring
    n = len(c)
    if not concur:
        q = R(1)
        s = c[0] * q
        for i in range(1, n):
            q = rs_mul(q, p, x, prec)
            s += c[i] * q
        return s
    J = int(math.sqrt(n) + 1)
    (K, r) = divmod(n, J)
    if r:
        K += 1
    ax = [R(1)]
    q = R(1)
    if len(p) < 20:
        for i in range(1, J):
            q = rs_mul(q, p, x, prec)
            ax.append(q)
    else:
        for i in range(1, J):
            if i % 2 == 0:
                q = rs_square(ax[i // 2], x, prec)
            else:
                q = rs_mul(q, p, x, prec)
            ax.append(q)
    pj = rs_mul(ax[-1], p, x, prec)
    b = R(1)
    s = R(0)
    for k in range(K - 1):
        r = J * k
        s1 = c[r]
        for j in range(1, J):
            s1 += c[r + j] * ax[j]
        s1 = rs_mul(s1, b, x, prec)
        s += s1
        b = rs_mul(b, pj, x, prec)
        if not b:
            break
    k = K - 1
    r = J * k
    if r < n:
        s1 = c[r] * R(1)
        for j in range(1, J):
            if r + j >= n:
                break
            s1 += c[r + j] * ax[j]
        s1 = rs_mul(s1, b, x, prec)
        s += s1
    return s

def rs_diff(p, x):
    if False:
        while True:
            i = 10
    "\n    Return partial derivative of ``p`` with respect to ``x``.\n\n    Parameters\n    ==========\n\n    x : :class:`~.PolyElement` with respect to which ``p`` is differentiated.\n\n    Examples\n    ========\n\n    >>> from sympy.polys.domains import QQ\n    >>> from sympy.polys.rings import ring\n    >>> from sympy.polys.ring_series import rs_diff\n    >>> R, x, y = ring('x, y', QQ)\n    >>> p = x + x**2*y**3\n    >>> rs_diff(p, x)\n    2*x*y**3 + 1\n    "
    R = p.ring
    n = R.gens.index(x)
    p1 = R.zero
    mn = [0] * R.ngens
    mn[n] = 1
    mn = tuple(mn)
    for expv in p:
        if expv[n]:
            e = monomial_ldiv(expv, mn)
            p1[e] = R.domain_new(p[expv] * expv[n])
    return p1

def rs_integrate(p, x):
    if False:
        i = 10
        return i + 15
    "\n    Integrate ``p`` with respect to ``x``.\n\n    Parameters\n    ==========\n\n    x : :class:`~.PolyElement` with respect to which ``p`` is integrated.\n\n    Examples\n    ========\n\n    >>> from sympy.polys.domains import QQ\n    >>> from sympy.polys.rings import ring\n    >>> from sympy.polys.ring_series import rs_integrate\n    >>> R, x, y = ring('x, y', QQ)\n    >>> p = x + x**2*y**3\n    >>> rs_integrate(p, x)\n    1/3*x**3*y**3 + 1/2*x**2\n    "
    R = p.ring
    p1 = R.zero
    n = R.gens.index(x)
    mn = [0] * R.ngens
    mn[n] = 1
    mn = tuple(mn)
    for expv in p:
        e = monomial_mul(expv, mn)
        p1[e] = R.domain_new(p[expv] / (expv[n] + 1))
    return p1

def rs_fun(p, f, *args):
    if False:
        print('Hello World!')
    "\n    Function of a multivariate series computed by substitution.\n\n    The case with f method name is used to compute `rs\\_tan` and `rs\\_nth\\_root`\n    of a multivariate series:\n\n        `rs\\_fun(p, tan, iv, prec)`\n\n        tan series is first computed for a dummy variable _x,\n        i.e, `rs\\_tan(\\_x, iv, prec)`. Then we substitute _x with p to get the\n        desired series\n\n    Parameters\n    ==========\n\n    p : :class:`~.PolyElement` The multivariate series to be expanded.\n    f : `ring\\_series` function to be applied on `p`.\n    args[-2] : :class:`~.PolyElement` with respect to which, the series is to be expanded.\n    args[-1] : Required order of the expanded series.\n\n    Examples\n    ========\n\n    >>> from sympy.polys.domains import QQ\n    >>> from sympy.polys.rings import ring\n    >>> from sympy.polys.ring_series import rs_fun, _tan1\n    >>> R, x, y = ring('x, y', QQ)\n    >>> p = x + x*y + x**2*y + x**3*y**2\n    >>> rs_fun(p, _tan1, x, 4)\n    1/3*x**3*y**3 + 2*x**3*y**2 + x**3*y + 1/3*x**3 + x**2*y + x*y + x\n    "
    _R = p.ring
    (R1, _x) = ring('_x', _R.domain)
    h = int(args[-1])
    args1 = args[:-2] + (_x, h)
    zm = _R.zero_monom
    if zm in p:
        x1 = _x + p[zm]
        p1 = p - p[zm]
    else:
        x1 = _x
        p1 = p
    if isinstance(f, str):
        q = getattr(x1, f)(*args1)
    else:
        q = f(x1, *args1)
    a = sorted(q.items())
    c = [0] * h
    for x in a:
        c[x[0][0]] = x[1]
    p1 = rs_series_from_list(p1, c, args[-2], args[-1])
    return p1

def mul_xin(p, i, n):
    if False:
        i = 10
        return i + 15
    '\n    Return `p*x_i**n`.\n\n    `x\\_i` is the ith variable in ``p``.\n    '
    R = p.ring
    q = R(0)
    for (k, v) in p.items():
        k1 = list(k)
        k1[i] += n
        q[tuple(k1)] = v
    return q

def pow_xin(p, i, n):
    if False:
        while True:
            i = 10
    "\n    >>> from sympy.polys.domains import QQ\n    >>> from sympy.polys.rings import ring\n    >>> from sympy.polys.ring_series import pow_xin\n    >>> R, x, y = ring('x, y', QQ)\n    >>> p = x**QQ(2,5) + x + x**QQ(2,3)\n    >>> index = p.ring.gens.index(x)\n    >>> pow_xin(p, index, 15)\n    x**15 + x**10 + x**6\n    "
    R = p.ring
    q = R(0)
    for (k, v) in p.items():
        k1 = list(k)
        k1[i] *= n
        q[tuple(k1)] = v
    return q

def _nth_root1(p, n, x, prec):
    if False:
        for i in range(10):
            print('nop')
    '\n    Univariate series expansion of the nth root of ``p``.\n\n    The Newton method is used.\n    '
    if rs_is_puiseux(p, x):
        return rs_puiseux2(_nth_root1, p, n, x, prec)
    R = p.ring
    zm = R.zero_monom
    if zm not in p:
        raise NotImplementedError('No constant term in series')
    n = as_int(n)
    assert p[zm] == 1
    p1 = R(1)
    if p == 1:
        return p
    if n == 0:
        return R(1)
    if n == 1:
        return p
    if n < 0:
        n = -n
        sign = 1
    else:
        sign = 0
    for precx in _giant_steps(prec):
        tmp = rs_pow(p1, n + 1, x, precx)
        tmp = rs_mul(tmp, p, x, precx)
        p1 += p1 / n - tmp / n
    if sign:
        return p1
    else:
        return _series_inversion1(p1, x, prec)

def rs_nth_root(p, n, x, prec):
    if False:
        while True:
            i = 10
    "\n    Multivariate series expansion of the nth root of ``p``.\n\n    Parameters\n    ==========\n\n    p : Expr\n        The polynomial to computer the root of.\n    n : integer\n        The order of the root to be computed.\n    x : :class:`~.PolyElement`\n    prec : integer\n        Order of the expanded series.\n\n    Notes\n    =====\n\n    The result of this function is dependent on the ring over which the\n    polynomial has been defined. If the answer involves a root of a constant,\n    make sure that the polynomial is over a real field. It cannot yet handle\n    roots of symbols.\n\n    Examples\n    ========\n\n    >>> from sympy.polys.domains import QQ, RR\n    >>> from sympy.polys.rings import ring\n    >>> from sympy.polys.ring_series import rs_nth_root\n    >>> R, x, y = ring('x, y', QQ)\n    >>> rs_nth_root(1 + x + x*y, -3, x, 3)\n    2/9*x**2*y**2 + 4/9*x**2*y + 2/9*x**2 - 1/3*x*y - 1/3*x + 1\n    >>> R, x, y = ring('x, y', RR)\n    >>> rs_nth_root(3 + x + x*y, 3, x, 2)\n    0.160249952256379*x*y + 0.160249952256379*x + 1.44224957030741\n    "
    if n == 0:
        if p == 0:
            raise ValueError('0**0 expression')
        else:
            return p.ring(1)
    if n == 1:
        return rs_trunc(p, x, prec)
    R = p.ring
    index = R.gens.index(x)
    m = min(p, key=lambda k: k[index])[index]
    p = mul_xin(p, index, -m)
    prec -= m
    if _has_constant_term(p - 1, x):
        zm = R.zero_monom
        c = p[zm]
        if R.domain is EX:
            c_expr = c.as_expr()
            const = c_expr ** QQ(1, n)
        elif isinstance(c, PolyElement):
            try:
                c_expr = c.as_expr()
                const = R(c_expr ** QQ(1, n))
            except ValueError:
                raise DomainError('The given series cannot be expanded in this domain.')
        else:
            try:
                const = R(c ** Rational(1, n))
            except ValueError:
                raise DomainError('The given series cannot be expanded in this domain.')
        res = rs_nth_root(p / c, n, x, prec) * const
    else:
        res = _nth_root1(p, n, x, prec)
    if m:
        m = QQ(m, n)
        res = mul_xin(res, index, m)
    return res

def rs_log(p, x, prec):
    if False:
        return 10
    "\n    The Logarithm of ``p`` modulo ``O(x**prec)``.\n\n    Notes\n    =====\n\n    Truncation of ``integral dx p**-1*d p/dx`` is used.\n\n    Examples\n    ========\n\n    >>> from sympy.polys.domains import QQ\n    >>> from sympy.polys.rings import ring\n    >>> from sympy.polys.ring_series import rs_log\n    >>> R, x = ring('x', QQ)\n    >>> rs_log(1 + x, x, 8)\n    1/7*x**7 - 1/6*x**6 + 1/5*x**5 - 1/4*x**4 + 1/3*x**3 - 1/2*x**2 + x\n    >>> rs_log(x**QQ(3, 2) + 1, x, 5)\n    1/3*x**(9/2) - 1/2*x**3 + x**(3/2)\n    "
    if rs_is_puiseux(p, x):
        return rs_puiseux(rs_log, p, x, prec)
    R = p.ring
    if p == 1:
        return R.zero
    c = _get_constant_term(p, x)
    if c:
        const = 0
        if c == 1:
            pass
        else:
            c_expr = c.as_expr()
            if R.domain is EX:
                const = log(c_expr)
            elif isinstance(c, PolyElement):
                try:
                    const = R(log(c_expr))
                except ValueError:
                    R = R.add_gens([log(c_expr)])
                    p = p.set_ring(R)
                    x = x.set_ring(R)
                    c = c.set_ring(R)
                    const = R(log(c_expr))
            else:
                try:
                    const = R(log(c))
                except ValueError:
                    raise DomainError('The given series cannot be expanded in this domain.')
        dlog = p.diff(x)
        dlog = rs_mul(dlog, _series_inversion1(p, x, prec), x, prec - 1)
        return rs_integrate(dlog, x) + const
    else:
        raise NotImplementedError

def rs_LambertW(p, x, prec):
    if False:
        for i in range(10):
            print('nop')
    "\n    Calculate the series expansion of the principal branch of the Lambert W\n    function.\n\n    Examples\n    ========\n\n    >>> from sympy.polys.domains import QQ\n    >>> from sympy.polys.rings import ring\n    >>> from sympy.polys.ring_series import rs_LambertW\n    >>> R, x, y = ring('x, y', QQ)\n    >>> rs_LambertW(x + x*y, x, 3)\n    -x**2*y**2 - 2*x**2*y - x**2 + x*y + x\n\n    See Also\n    ========\n\n    LambertW\n    "
    if rs_is_puiseux(p, x):
        return rs_puiseux(rs_LambertW, p, x, prec)
    R = p.ring
    p1 = R(0)
    if _has_constant_term(p, x):
        raise NotImplementedError('Polynomial must not have constant term in the series variables')
    if x in R.gens:
        for precx in _giant_steps(prec):
            e = rs_exp(p1, x, precx)
            p2 = rs_mul(e, p1, x, precx) - p
            p3 = rs_mul(e, p1 + 1, x, precx)
            p3 = rs_series_inversion(p3, x, precx)
            tmp = rs_mul(p2, p3, x, precx)
            p1 -= tmp
        return p1
    else:
        raise NotImplementedError

def _exp1(p, x, prec):
    if False:
        return 10
    'Helper function for `rs\\_exp`. '
    R = p.ring
    p1 = R(1)
    for precx in _giant_steps(prec):
        pt = p - rs_log(p1, x, precx)
        tmp = rs_mul(pt, p1, x, precx)
        p1 += tmp
    return p1

def rs_exp(p, x, prec):
    if False:
        print('Hello World!')
    "\n    Exponentiation of a series modulo ``O(x**prec)``\n\n    Examples\n    ========\n\n    >>> from sympy.polys.domains import QQ\n    >>> from sympy.polys.rings import ring\n    >>> from sympy.polys.ring_series import rs_exp\n    >>> R, x = ring('x', QQ)\n    >>> rs_exp(x**2, x, 7)\n    1/6*x**6 + 1/2*x**4 + x**2 + 1\n    "
    if rs_is_puiseux(p, x):
        return rs_puiseux(rs_exp, p, x, prec)
    R = p.ring
    c = _get_constant_term(p, x)
    if c:
        if R.domain is EX:
            c_expr = c.as_expr()
            const = exp(c_expr)
        elif isinstance(c, PolyElement):
            try:
                c_expr = c.as_expr()
                const = R(exp(c_expr))
            except ValueError:
                R = R.add_gens([exp(c_expr)])
                p = p.set_ring(R)
                x = x.set_ring(R)
                c = c.set_ring(R)
                const = R(exp(c_expr))
        else:
            try:
                const = R(exp(c))
            except ValueError:
                raise DomainError('The given series cannot be expanded in this domain.')
        p1 = p - c
        return const * rs_exp(p1, x, prec)
    if len(p) > 20:
        return _exp1(p, x, prec)
    one = R(1)
    n = 1
    c = []
    for k in range(prec):
        c.append(one / n)
        k += 1
        n *= k
    r = rs_series_from_list(p, c, x, prec)
    return r

def _atan(p, iv, prec):
    if False:
        while True:
            i = 10
    '\n    Expansion using formula.\n\n    Faster on very small and univariate series.\n    '
    R = p.ring
    mo = R(-1)
    c = [-mo]
    p2 = rs_square(p, iv, prec)
    for k in range(1, prec):
        c.append(mo ** k / (2 * k + 1))
    s = rs_series_from_list(p2, c, iv, prec)
    s = rs_mul(s, p, iv, prec)
    return s

def rs_atan(p, x, prec):
    if False:
        return 10
    "\n    The arctangent of a series\n\n    Return the series expansion of the atan of ``p``, about 0.\n\n    Examples\n    ========\n\n    >>> from sympy.polys.domains import QQ\n    >>> from sympy.polys.rings import ring\n    >>> from sympy.polys.ring_series import rs_atan\n    >>> R, x, y = ring('x, y', QQ)\n    >>> rs_atan(x + x*y, x, 4)\n    -1/3*x**3*y**3 - x**3*y**2 - x**3*y - 1/3*x**3 + x*y + x\n\n    See Also\n    ========\n\n    atan\n    "
    if rs_is_puiseux(p, x):
        return rs_puiseux(rs_atan, p, x, prec)
    R = p.ring
    const = 0
    if _has_constant_term(p, x):
        zm = R.zero_monom
        c = p[zm]
        if R.domain is EX:
            c_expr = c.as_expr()
            const = atan(c_expr)
        elif isinstance(c, PolyElement):
            try:
                c_expr = c.as_expr()
                const = R(atan(c_expr))
            except ValueError:
                raise DomainError('The given series cannot be expanded in this domain.')
        else:
            try:
                const = R(atan(c))
            except ValueError:
                raise DomainError('The given series cannot be expanded in this domain.')
    dp = p.diff(x)
    p1 = rs_square(p, x, prec) + R(1)
    p1 = rs_series_inversion(p1, x, prec - 1)
    p1 = rs_mul(dp, p1, x, prec - 1)
    return rs_integrate(p1, x) + const

def rs_asin(p, x, prec):
    if False:
        return 10
    "\n    Arcsine of a series\n\n    Return the series expansion of the asin of ``p``, about 0.\n\n    Examples\n    ========\n\n    >>> from sympy.polys.domains import QQ\n    >>> from sympy.polys.rings import ring\n    >>> from sympy.polys.ring_series import rs_asin\n    >>> R, x, y = ring('x, y', QQ)\n    >>> rs_asin(x, x, 8)\n    5/112*x**7 + 3/40*x**5 + 1/6*x**3 + x\n\n    See Also\n    ========\n\n    asin\n    "
    if rs_is_puiseux(p, x):
        return rs_puiseux(rs_asin, p, x, prec)
    if _has_constant_term(p, x):
        raise NotImplementedError('Polynomial must not have constant term in series variables')
    R = p.ring
    if x in R.gens:
        if len(p) > 20:
            dp = rs_diff(p, x)
            p1 = 1 - rs_square(p, x, prec - 1)
            p1 = rs_nth_root(p1, -2, x, prec - 1)
            p1 = rs_mul(dp, p1, x, prec - 1)
            return rs_integrate(p1, x)
        one = R(1)
        c = [0, one, 0]
        for k in range(3, prec, 2):
            c.append((k - 2) ** 2 * c[-2] / (k * (k - 1)))
            c.append(0)
        return rs_series_from_list(p, c, x, prec)
    else:
        raise NotImplementedError

def _tan1(p, x, prec):
    if False:
        return 10
    "\n    Helper function of :func:`rs_tan`.\n\n    Return the series expansion of tan of a univariate series using Newton's\n    method. It takes advantage of the fact that series expansion of atan is\n    easier than that of tan.\n\n    Consider `f(x) = y - \\arctan(x)`\n    Let r be a root of f(x) found using Newton's method.\n    Then `f(r) = 0`\n    Or `y = \\arctan(x)` where `x = \\tan(y)` as required.\n    "
    R = p.ring
    p1 = R(0)
    for precx in _giant_steps(prec):
        tmp = p - rs_atan(p1, x, precx)
        tmp = rs_mul(tmp, 1 + rs_square(p1, x, precx), x, precx)
        p1 += tmp
    return p1

def rs_tan(p, x, prec):
    if False:
        for i in range(10):
            print('nop')
    "\n    Tangent of a series.\n\n    Return the series expansion of the tan of ``p``, about 0.\n\n    Examples\n    ========\n\n    >>> from sympy.polys.domains import QQ\n    >>> from sympy.polys.rings import ring\n    >>> from sympy.polys.ring_series import rs_tan\n    >>> R, x, y = ring('x, y', QQ)\n    >>> rs_tan(x + x*y, x, 4)\n    1/3*x**3*y**3 + x**3*y**2 + x**3*y + 1/3*x**3 + x*y + x\n\n   See Also\n   ========\n\n   _tan1, tan\n   "
    if rs_is_puiseux(p, x):
        r = rs_puiseux(rs_tan, p, x, prec)
        return r
    R = p.ring
    const = 0
    c = _get_constant_term(p, x)
    if c:
        if R.domain is EX:
            c_expr = c.as_expr()
            const = tan(c_expr)
        elif isinstance(c, PolyElement):
            try:
                c_expr = c.as_expr()
                const = R(tan(c_expr))
            except ValueError:
                R = R.add_gens([tan(c_expr)])
                p = p.set_ring(R)
                x = x.set_ring(R)
                c = c.set_ring(R)
                const = R(tan(c_expr))
        else:
            try:
                const = R(tan(c))
            except ValueError:
                raise DomainError('The given series cannot be expanded in this domain.')
        p1 = p - c
        t2 = rs_tan(p1, x, prec)
        t = rs_series_inversion(1 - const * t2, x, prec)
        return rs_mul(const + t2, t, x, prec)
    if R.ngens == 1:
        return _tan1(p, x, prec)
    else:
        return rs_fun(p, rs_tan, x, prec)

def rs_cot(p, x, prec):
    if False:
        for i in range(10):
            print('nop')
    "\n    Cotangent of a series\n\n    Return the series expansion of the cot of ``p``, about 0.\n\n    Examples\n    ========\n\n    >>> from sympy.polys.domains import QQ\n    >>> from sympy.polys.rings import ring\n    >>> from sympy.polys.ring_series import rs_cot\n    >>> R, x, y = ring('x, y', QQ)\n    >>> rs_cot(x, x, 6)\n    -2/945*x**5 - 1/45*x**3 - 1/3*x + x**(-1)\n\n    See Also\n    ========\n\n    cot\n    "
    if rs_is_puiseux(p, x):
        r = rs_puiseux(rs_cot, p, x, prec)
        return r
    (i, m) = _check_series_var(p, x, 'cot')
    prec1 = prec + 2 * m
    (c, s) = rs_cos_sin(p, x, prec1)
    s = mul_xin(s, i, -m)
    s = rs_series_inversion(s, x, prec1)
    res = rs_mul(c, s, x, prec1)
    res = mul_xin(res, i, -m)
    res = rs_trunc(res, x, prec)
    return res

def rs_sin(p, x, prec):
    if False:
        return 10
    "\n    Sine of a series\n\n    Return the series expansion of the sin of ``p``, about 0.\n\n    Examples\n    ========\n\n    >>> from sympy.polys.domains import QQ\n    >>> from sympy.polys.rings import ring\n    >>> from sympy.polys.ring_series import rs_sin\n    >>> R, x, y = ring('x, y', QQ)\n    >>> rs_sin(x + x*y, x, 4)\n    -1/6*x**3*y**3 - 1/2*x**3*y**2 - 1/2*x**3*y - 1/6*x**3 + x*y + x\n    >>> rs_sin(x**QQ(3, 2) + x*y**QQ(7, 5), x, 4)\n    -1/2*x**(7/2)*y**(14/5) - 1/6*x**3*y**(21/5) + x**(3/2) + x*y**(7/5)\n\n    See Also\n    ========\n\n    sin\n    "
    if rs_is_puiseux(p, x):
        return rs_puiseux(rs_sin, p, x, prec)
    R = x.ring
    if not p:
        return R(0)
    c = _get_constant_term(p, x)
    if c:
        if R.domain is EX:
            c_expr = c.as_expr()
            (t1, t2) = (sin(c_expr), cos(c_expr))
        elif isinstance(c, PolyElement):
            try:
                c_expr = c.as_expr()
                (t1, t2) = (R(sin(c_expr)), R(cos(c_expr)))
            except ValueError:
                R = R.add_gens([sin(c_expr), cos(c_expr)])
                p = p.set_ring(R)
                x = x.set_ring(R)
                c = c.set_ring(R)
                (t1, t2) = (R(sin(c_expr)), R(cos(c_expr)))
        else:
            try:
                (t1, t2) = (R(sin(c)), R(cos(c)))
            except ValueError:
                raise DomainError('The given series cannot be expanded in this domain.')
        p1 = p - c
        return rs_sin(p1, x, prec) * t2 + rs_cos(p1, x, prec) * t1
    if len(p) > 20 and R.ngens == 1:
        t = rs_tan(p / 2, x, prec)
        t2 = rs_square(t, x, prec)
        p1 = rs_series_inversion(1 + t2, x, prec)
        return rs_mul(p1, 2 * t, x, prec)
    one = R(1)
    n = 1
    c = [0]
    for k in range(2, prec + 2, 2):
        c.append(one / n)
        c.append(0)
        n *= -k * (k + 1)
    return rs_series_from_list(p, c, x, prec)

def rs_cos(p, x, prec):
    if False:
        for i in range(10):
            print('nop')
    "\n    Cosine of a series\n\n    Return the series expansion of the cos of ``p``, about 0.\n\n    Examples\n    ========\n\n    >>> from sympy.polys.domains import QQ\n    >>> from sympy.polys.rings import ring\n    >>> from sympy.polys.ring_series import rs_cos\n    >>> R, x, y = ring('x, y', QQ)\n    >>> rs_cos(x + x*y, x, 4)\n    -1/2*x**2*y**2 - x**2*y - 1/2*x**2 + 1\n    >>> rs_cos(x + x*y, x, 4)/x**QQ(7, 5)\n    -1/2*x**(3/5)*y**2 - x**(3/5)*y - 1/2*x**(3/5) + x**(-7/5)\n\n    See Also\n    ========\n\n    cos\n    "
    if rs_is_puiseux(p, x):
        return rs_puiseux(rs_cos, p, x, prec)
    R = p.ring
    c = _get_constant_term(p, x)
    if c:
        if R.domain is EX:
            c_expr = c.as_expr()
            (_, _) = (sin(c_expr), cos(c_expr))
        elif isinstance(c, PolyElement):
            try:
                c_expr = c.as_expr()
                (_, _) = (R(sin(c_expr)), R(cos(c_expr)))
            except ValueError:
                R = R.add_gens([sin(c_expr), cos(c_expr)])
                p = p.set_ring(R)
                x = x.set_ring(R)
                c = c.set_ring(R)
        else:
            try:
                (_, _) = (R(sin(c)), R(cos(c)))
            except ValueError:
                raise DomainError('The given series cannot be expanded in this domain.')
        p1 = p - c
        p_cos = rs_cos(p1, x, prec)
        p_sin = rs_sin(p1, x, prec)
        R = R.compose(p_cos.ring).compose(p_sin.ring)
        p_cos.set_ring(R)
        p_sin.set_ring(R)
        (t1, t2) = (R(sin(c_expr)), R(cos(c_expr)))
        return p_cos * t2 - p_sin * t1
    if len(p) > 20 and R.ngens == 1:
        t = rs_tan(p / 2, x, prec)
        t2 = rs_square(t, x, prec)
        p1 = rs_series_inversion(1 + t2, x, prec)
        return rs_mul(p1, 1 - t2, x, prec)
    one = R(1)
    n = 1
    c = []
    for k in range(2, prec + 2, 2):
        c.append(one / n)
        c.append(0)
        n *= -k * (k - 1)
    return rs_series_from_list(p, c, x, prec)

def rs_cos_sin(p, x, prec):
    if False:
        while True:
            i = 10
    '\n    Return the tuple ``(rs_cos(p, x, prec)`, `rs_sin(p, x, prec))``.\n\n    Is faster than calling rs_cos and rs_sin separately\n    '
    if rs_is_puiseux(p, x):
        return rs_puiseux(rs_cos_sin, p, x, prec)
    t = rs_tan(p / 2, x, prec)
    t2 = rs_square(t, x, prec)
    p1 = rs_series_inversion(1 + t2, x, prec)
    return (rs_mul(p1, 1 - t2, x, prec), rs_mul(p1, 2 * t, x, prec))

def _atanh(p, x, prec):
    if False:
        while True:
            i = 10
    '\n    Expansion using formula\n\n    Faster for very small and univariate series\n    '
    R = p.ring
    one = R(1)
    c = [one]
    p2 = rs_square(p, x, prec)
    for k in range(1, prec):
        c.append(one / (2 * k + 1))
    s = rs_series_from_list(p2, c, x, prec)
    s = rs_mul(s, p, x, prec)
    return s

def rs_atanh(p, x, prec):
    if False:
        return 10
    "\n    Hyperbolic arctangent of a series\n\n    Return the series expansion of the atanh of ``p``, about 0.\n\n    Examples\n    ========\n\n    >>> from sympy.polys.domains import QQ\n    >>> from sympy.polys.rings import ring\n    >>> from sympy.polys.ring_series import rs_atanh\n    >>> R, x, y = ring('x, y', QQ)\n    >>> rs_atanh(x + x*y, x, 4)\n    1/3*x**3*y**3 + x**3*y**2 + x**3*y + 1/3*x**3 + x*y + x\n\n    See Also\n    ========\n\n    atanh\n    "
    if rs_is_puiseux(p, x):
        return rs_puiseux(rs_atanh, p, x, prec)
    R = p.ring
    const = 0
    if _has_constant_term(p, x):
        zm = R.zero_monom
        c = p[zm]
        if R.domain is EX:
            c_expr = c.as_expr()
            const = atanh(c_expr)
        elif isinstance(c, PolyElement):
            try:
                c_expr = c.as_expr()
                const = R(atanh(c_expr))
            except ValueError:
                raise DomainError('The given series cannot be expanded in this domain.')
        else:
            try:
                const = R(atanh(c))
            except ValueError:
                raise DomainError('The given series cannot be expanded in this domain.')
    dp = rs_diff(p, x)
    p1 = -rs_square(p, x, prec) + 1
    p1 = rs_series_inversion(p1, x, prec - 1)
    p1 = rs_mul(dp, p1, x, prec - 1)
    return rs_integrate(p1, x) + const

def rs_sinh(p, x, prec):
    if False:
        i = 10
        return i + 15
    "\n    Hyperbolic sine of a series\n\n    Return the series expansion of the sinh of ``p``, about 0.\n\n    Examples\n    ========\n\n    >>> from sympy.polys.domains import QQ\n    >>> from sympy.polys.rings import ring\n    >>> from sympy.polys.ring_series import rs_sinh\n    >>> R, x, y = ring('x, y', QQ)\n    >>> rs_sinh(x + x*y, x, 4)\n    1/6*x**3*y**3 + 1/2*x**3*y**2 + 1/2*x**3*y + 1/6*x**3 + x*y + x\n\n    See Also\n    ========\n\n    sinh\n    "
    if rs_is_puiseux(p, x):
        return rs_puiseux(rs_sinh, p, x, prec)
    t = rs_exp(p, x, prec)
    t1 = rs_series_inversion(t, x, prec)
    return (t - t1) / 2

def rs_cosh(p, x, prec):
    if False:
        while True:
            i = 10
    "\n    Hyperbolic cosine of a series\n\n    Return the series expansion of the cosh of ``p``, about 0.\n\n    Examples\n    ========\n\n    >>> from sympy.polys.domains import QQ\n    >>> from sympy.polys.rings import ring\n    >>> from sympy.polys.ring_series import rs_cosh\n    >>> R, x, y = ring('x, y', QQ)\n    >>> rs_cosh(x + x*y, x, 4)\n    1/2*x**2*y**2 + x**2*y + 1/2*x**2 + 1\n\n    See Also\n    ========\n\n    cosh\n    "
    if rs_is_puiseux(p, x):
        return rs_puiseux(rs_cosh, p, x, prec)
    t = rs_exp(p, x, prec)
    t1 = rs_series_inversion(t, x, prec)
    return (t + t1) / 2

def _tanh(p, x, prec):
    if False:
        while True:
            i = 10
    "\n    Helper function of :func:`rs_tanh`\n\n    Return the series expansion of tanh of a univariate series using Newton's\n    method. It takes advantage of the fact that series expansion of atanh is\n    easier than that of tanh.\n\n    See Also\n    ========\n\n    _tanh\n    "
    R = p.ring
    p1 = R(0)
    for precx in _giant_steps(prec):
        tmp = p - rs_atanh(p1, x, precx)
        tmp = rs_mul(tmp, 1 - rs_square(p1, x, prec), x, precx)
        p1 += tmp
    return p1

def rs_tanh(p, x, prec):
    if False:
        i = 10
        return i + 15
    "\n    Hyperbolic tangent of a series\n\n    Return the series expansion of the tanh of ``p``, about 0.\n\n    Examples\n    ========\n\n    >>> from sympy.polys.domains import QQ\n    >>> from sympy.polys.rings import ring\n    >>> from sympy.polys.ring_series import rs_tanh\n    >>> R, x, y = ring('x, y', QQ)\n    >>> rs_tanh(x + x*y, x, 4)\n    -1/3*x**3*y**3 - x**3*y**2 - x**3*y - 1/3*x**3 + x*y + x\n\n    See Also\n    ========\n\n    tanh\n    "
    if rs_is_puiseux(p, x):
        return rs_puiseux(rs_tanh, p, x, prec)
    R = p.ring
    const = 0
    if _has_constant_term(p, x):
        zm = R.zero_monom
        c = p[zm]
        if R.domain is EX:
            c_expr = c.as_expr()
            const = tanh(c_expr)
        elif isinstance(c, PolyElement):
            try:
                c_expr = c.as_expr()
                const = R(tanh(c_expr))
            except ValueError:
                raise DomainError('The given series cannot be expanded in this domain.')
        else:
            try:
                const = R(tanh(c))
            except ValueError:
                raise DomainError('The given series cannot be expanded in this domain.')
        p1 = p - c
        t1 = rs_tanh(p1, x, prec)
        t = rs_series_inversion(1 + const * t1, x, prec)
        return rs_mul(const + t1, t, x, prec)
    if R.ngens == 1:
        return _tanh(p, x, prec)
    else:
        return rs_fun(p, _tanh, x, prec)

def rs_newton(p, x, prec):
    if False:
        return 10
    "\n    Compute the truncated Newton sum of the polynomial ``p``\n\n    Examples\n    ========\n\n    >>> from sympy.polys.domains import QQ\n    >>> from sympy.polys.rings import ring\n    >>> from sympy.polys.ring_series import rs_newton\n    >>> R, x = ring('x', QQ)\n    >>> p = x**2 - 2\n    >>> rs_newton(p, x, 5)\n    8*x**4 + 4*x**2 + 2\n    "
    deg = p.degree()
    p1 = _invert_monoms(p)
    p2 = rs_series_inversion(p1, x, prec)
    p3 = rs_mul(p1.diff(x), p2, x, prec)
    res = deg - p3 * x
    return res

def rs_hadamard_exp(p1, inverse=False):
    if False:
        return 10
    "\n    Return ``sum f_i/i!*x**i`` from ``sum f_i*x**i``,\n    where ``x`` is the first variable.\n\n    If ``invers=True`` return ``sum f_i*i!*x**i``\n\n    Examples\n    ========\n\n    >>> from sympy.polys.domains import QQ\n    >>> from sympy.polys.rings import ring\n    >>> from sympy.polys.ring_series import rs_hadamard_exp\n    >>> R, x = ring('x', QQ)\n    >>> p = 1 + x + x**2 + x**3\n    >>> rs_hadamard_exp(p)\n    1/6*x**3 + 1/2*x**2 + x + 1\n    "
    R = p1.ring
    if R.domain != QQ:
        raise NotImplementedError
    p = R.zero
    if not inverse:
        for (exp1, v1) in p1.items():
            p[exp1] = v1 / int(ifac(exp1[0]))
    else:
        for (exp1, v1) in p1.items():
            p[exp1] = v1 * int(ifac(exp1[0]))
    return p

def rs_compose_add(p1, p2):
    if False:
        for i in range(10):
            print('nop')
    '\n    compute the composed sum ``prod(p2(x - beta) for beta root of p1)``\n\n    Examples\n    ========\n\n    >>> from sympy.polys.domains import QQ\n    >>> from sympy.polys.rings import ring\n    >>> from sympy.polys.ring_series import rs_compose_add\n    >>> R, x = ring(\'x\', QQ)\n    >>> f = x**2 - 2\n    >>> g = x**2 - 3\n    >>> rs_compose_add(f, g)\n    x**4 - 10*x**2 + 1\n\n    References\n    ==========\n\n    .. [1] A. Bostan, P. Flajolet, B. Salvy and E. Schost\n           "Fast Computation with Two Algebraic Numbers",\n           (2002) Research Report 4579, Institut\n           National de Recherche en Informatique et en Automatique\n    '
    R = p1.ring
    x = R.gens[0]
    prec = p1.degree() * p2.degree() + 1
    np1 = rs_newton(p1, x, prec)
    np1e = rs_hadamard_exp(np1)
    np2 = rs_newton(p2, x, prec)
    np2e = rs_hadamard_exp(np2)
    np3e = rs_mul(np1e, np2e, x, prec)
    np3 = rs_hadamard_exp(np3e, True)
    np3a = (np3[0,] - np3) / x
    q = rs_integrate(np3a, x)
    q = rs_exp(q, x, prec)
    q = _invert_monoms(q)
    q = q.primitive()[1]
    dp = p1.degree() * p2.degree() - q.degree()
    if dp:
        q = q * x ** dp
    return q
_convert_func = {'sin': 'rs_sin', 'cos': 'rs_cos', 'exp': 'rs_exp', 'tan': 'rs_tan', 'log': 'rs_log'}

def rs_min_pow(expr, series_rs, a):
    if False:
        for i in range(10):
            print('nop')
    'Find the minimum power of `a` in the series expansion of expr'
    series = 0
    n = 2
    while series == 0:
        series = _rs_series(expr, series_rs, a, n)
        n *= 2
    R = series.ring
    a = R(a)
    i = R.gens.index(a)
    return min(series, key=lambda t: t[i])[i]

def _rs_series(expr, series_rs, a, prec):
    if False:
        i = 10
        return i + 15
    args = expr.args
    R = series_rs.ring
    if not any((arg.has(Function) for arg in args)) and (not expr.is_Function):
        return series_rs
    if not expr.has(a):
        return series_rs
    elif expr.is_Function:
        arg = args[0]
        if len(args) > 1:
            raise NotImplementedError
        (R1, series) = sring(arg, domain=QQ, expand=False, series=True)
        series_inner = _rs_series(arg, series, a, prec)
        R = R.compose(R1).compose(series_inner.ring)
        series_inner = series_inner.set_ring(R)
        series = eval(_convert_func[str(expr.func)])(series_inner, R(a), prec)
        return series
    elif expr.is_Mul:
        n = len(args)
        for arg in args:
            if not arg.is_Number:
                (R1, _) = sring(arg, expand=False, series=True)
                R = R.compose(R1)
        min_pows = list(map(rs_min_pow, args, [R(arg) for arg in args], [a] * len(args)))
        sum_pows = sum(min_pows)
        series = R(1)
        for i in range(n):
            _series = _rs_series(args[i], R(args[i]), a, prec - sum_pows + min_pows[i])
            R = R.compose(_series.ring)
            _series = _series.set_ring(R)
            series = series.set_ring(R)
            series *= _series
        series = rs_trunc(series, R(a), prec)
        return series
    elif expr.is_Add:
        n = len(args)
        series = R(0)
        for i in range(n):
            _series = _rs_series(args[i], R(args[i]), a, prec)
            R = R.compose(_series.ring)
            _series = _series.set_ring(R)
            series = series.set_ring(R)
            series += _series
        return series
    elif expr.is_Pow:
        (R1, _) = sring(expr.base, domain=QQ, expand=False, series=True)
        R = R.compose(R1)
        series_inner = _rs_series(expr.base, R(expr.base), a, prec)
        return rs_pow(series_inner, expr.exp, series_inner.ring(a), prec)
    elif isinstance(expr, Expr) and expr.is_constant():
        return sring(expr, domain=QQ, expand=False, series=True)[1]
    else:
        raise NotImplementedError

def rs_series(expr, a, prec):
    if False:
        print('Hello World!')
    "Return the series expansion of an expression about 0.\n\n    Parameters\n    ==========\n\n    expr : :class:`Expr`\n    a : :class:`Symbol` with respect to which expr is to be expanded\n    prec : order of the series expansion\n\n    Currently supports multivariate Taylor series expansion. This is much\n    faster that SymPy's series method as it uses sparse polynomial operations.\n\n    It automatically creates the simplest ring required to represent the series\n    expansion through repeated calls to sring.\n\n    Examples\n    ========\n\n    >>> from sympy.polys.ring_series import rs_series\n    >>> from sympy import sin, cos, exp, tan, symbols, QQ\n    >>> a, b, c = symbols('a, b, c')\n    >>> rs_series(sin(a) + exp(a), a, 5)\n    1/24*a**4 + 1/2*a**2 + 2*a + 1\n    >>> series = rs_series(tan(a + b)*cos(a + c), a, 2)\n    >>> series.as_expr()\n    -a*sin(c)*tan(b) + a*cos(c)*tan(b)**2 + a*cos(c) + cos(c)*tan(b)\n    >>> series = rs_series(exp(a**QQ(1,3) + a**QQ(2, 5)), a, 1)\n    >>> series.as_expr()\n    a**(11/15) + a**(4/5)/2 + a**(2/5) + a**(2/3)/2 + a**(1/3) + 1\n\n    "
    (R, series) = sring(expr, domain=QQ, expand=False, series=True)
    if a not in R.symbols:
        R = R.add_gens([a])
    series = series.set_ring(R)
    series = _rs_series(expr, series, a, prec)
    R = series.ring
    gen = R(a)
    prec_got = series.degree(gen) + 1
    if prec_got >= prec:
        return rs_trunc(series, gen, prec)
    else:
        for more in range(1, 9):
            p1 = _rs_series(expr, series, a, prec=prec + more)
            gen = gen.set_ring(p1.ring)
            new_prec = p1.degree(gen) + 1
            if new_prec != prec_got:
                prec_do = ceiling(prec + (prec - prec_got) * more / (new_prec - prec_got))
                p1 = _rs_series(expr, series, a, prec=prec_do)
                while p1.degree(gen) + 1 < prec:
                    p1 = _rs_series(expr, series, a, prec=prec_do)
                    gen = gen.set_ring(p1.ring)
                    prec_do *= 2
                break
            else:
                break
        else:
            raise ValueError('Could not calculate %s terms for %s' % (str(prec), expr))
        return rs_trunc(p1, gen, prec)