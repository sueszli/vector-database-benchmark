"""Dense univariate polynomials with coefficients in Galois fields. """
from math import ceil as _ceil, sqrt as _sqrt, prod
from sympy.core.random import uniform, _randint
from sympy.external.gmpy import SYMPY_INTS, MPZ, invert
from sympy.polys.polyconfig import query
from sympy.polys.polyerrors import ExactQuotientFailed
from sympy.polys.polyutils import _sort_factors

def gf_crt(U, M, K=None):
    if False:
        while True:
            i = 10
    '\n    Chinese Remainder Theorem.\n\n    Given a set of integer residues ``u_0,...,u_n`` and a set of\n    co-prime integer moduli ``m_0,...,m_n``, returns an integer\n    ``u``, such that ``u = u_i mod m_i`` for ``i = ``0,...,n``.\n\n    Examples\n    ========\n\n    Consider a set of residues ``U = [49, 76, 65]``\n    and a set of moduli ``M = [99, 97, 95]``. Then we have::\n\n       >>> from sympy.polys.domains import ZZ\n       >>> from sympy.polys.galoistools import gf_crt\n\n       >>> gf_crt([49, 76, 65], [99, 97, 95], ZZ)\n       639985\n\n    This is the correct result because::\n\n       >>> [639985 % m for m in [99, 97, 95]]\n       [49, 76, 65]\n\n    Note: this is a low-level routine with no error checking.\n\n    See Also\n    ========\n\n    sympy.ntheory.modular.crt : a higher level crt routine\n    sympy.ntheory.modular.solve_congruence\n\n    '
    p = prod(M, start=K.one)
    v = K.zero
    for (u, m) in zip(U, M):
        e = p // m
        (s, _, _) = K.gcdex(e, m)
        v += e * (u * s % m)
    return v % p

def gf_crt1(M, K):
    if False:
        i = 10
        return i + 15
    '\n    First part of the Chinese Remainder Theorem.\n\n    Examples\n    ========\n\n    >>> from sympy.polys.domains import ZZ\n    >>> from sympy.polys.galoistools import gf_crt, gf_crt1, gf_crt2\n    >>> U = [49, 76, 65]\n    >>> M = [99, 97, 95]\n\n    The following two codes have the same result.\n\n    >>> gf_crt(U, M, ZZ)\n    639985\n\n    >>> p, E, S = gf_crt1(M, ZZ)\n    >>> gf_crt2(U, M, p, E, S, ZZ)\n    639985\n\n    However, it is faster when we want to fix ``M`` and\n    compute for multiple U, i.e. the following cases:\n\n    >>> p, E, S = gf_crt1(M, ZZ)\n    >>> Us = [[49, 76, 65], [23, 42, 67]]\n    >>> for U in Us:\n    ...     print(gf_crt2(U, M, p, E, S, ZZ))\n    639985\n    236237\n\n    See Also\n    ========\n\n    sympy.ntheory.modular.crt1 : a higher level crt routine\n    sympy.polys.galoistools.gf_crt\n    sympy.polys.galoistools.gf_crt2\n\n    '
    (E, S) = ([], [])
    p = prod(M, start=K.one)
    for m in M:
        E.append(p // m)
        S.append(K.gcdex(E[-1], m)[0] % m)
    return (p, E, S)

def gf_crt2(U, M, p, E, S, K):
    if False:
        for i in range(10):
            print('nop')
    '\n    Second part of the Chinese Remainder Theorem.\n\n    See ``gf_crt1`` for usage.\n\n    Examples\n    ========\n\n    >>> from sympy.polys.domains import ZZ\n    >>> from sympy.polys.galoistools import gf_crt2\n\n    >>> U = [49, 76, 65]\n    >>> M = [99, 97, 95]\n    >>> p = 912285\n    >>> E = [9215, 9405, 9603]\n    >>> S = [62, 24, 12]\n\n    >>> gf_crt2(U, M, p, E, S, ZZ)\n    639985\n\n    See Also\n    ========\n\n    sympy.ntheory.modular.crt2 : a higher level crt routine\n    sympy.polys.galoistools.gf_crt\n    sympy.polys.galoistools.gf_crt1\n\n    '
    v = K.zero
    for (u, m, e, s) in zip(U, M, E, S):
        v += e * (u * s % m)
    return v % p

def gf_int(a, p):
    if False:
        for i in range(10):
            print('nop')
    '\n    Coerce ``a mod p`` to an integer in the range ``[-p/2, p/2]``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys.galoistools import gf_int\n\n    >>> gf_int(2, 7)\n    2\n    >>> gf_int(5, 7)\n    -2\n\n    '
    if a <= p // 2:
        return a
    else:
        return a - p

def gf_degree(f):
    if False:
        while True:
            i = 10
    '\n    Return the leading degree of ``f``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys.galoistools import gf_degree\n\n    >>> gf_degree([1, 1, 2, 0])\n    3\n    >>> gf_degree([])\n    -1\n\n    '
    return len(f) - 1

def gf_LC(f, K):
    if False:
        return 10
    '\n    Return the leading coefficient of ``f``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys.domains import ZZ\n    >>> from sympy.polys.galoistools import gf_LC\n\n    >>> gf_LC([3, 0, 1], ZZ)\n    3\n\n    '
    if not f:
        return K.zero
    else:
        return f[0]

def gf_TC(f, K):
    if False:
        return 10
    '\n    Return the trailing coefficient of ``f``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys.domains import ZZ\n    >>> from sympy.polys.galoistools import gf_TC\n\n    >>> gf_TC([3, 0, 1], ZZ)\n    1\n\n    '
    if not f:
        return K.zero
    else:
        return f[-1]

def gf_strip(f):
    if False:
        for i in range(10):
            print('nop')
    '\n    Remove leading zeros from ``f``.\n\n\n    Examples\n    ========\n\n    >>> from sympy.polys.galoistools import gf_strip\n\n    >>> gf_strip([0, 0, 0, 3, 0, 1])\n    [3, 0, 1]\n\n    '
    if not f or f[0]:
        return f
    k = 0
    for coeff in f:
        if coeff:
            break
        else:
            k += 1
    return f[k:]

def gf_trunc(f, p):
    if False:
        return 10
    '\n    Reduce all coefficients modulo ``p``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys.galoistools import gf_trunc\n\n    >>> gf_trunc([7, -2, 3], 5)\n    [2, 3, 3]\n\n    '
    return gf_strip([a % p for a in f])

def gf_normal(f, p, K):
    if False:
        return 10
    '\n    Normalize all coefficients in ``K``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys.domains import ZZ\n    >>> from sympy.polys.galoistools import gf_normal\n\n    >>> gf_normal([5, 10, 21, -3], 5, ZZ)\n    [1, 2]\n\n    '
    return gf_trunc(list(map(K, f)), p)

def gf_from_dict(f, p, K):
    if False:
        return 10
    '\n    Create a ``GF(p)[x]`` polynomial from a dict.\n\n    Examples\n    ========\n\n    >>> from sympy.polys.domains import ZZ\n    >>> from sympy.polys.galoistools import gf_from_dict\n\n    >>> gf_from_dict({10: ZZ(4), 4: ZZ(33), 0: ZZ(-1)}, 5, ZZ)\n    [4, 0, 0, 0, 0, 0, 3, 0, 0, 0, 4]\n\n    '
    (n, h) = (max(f.keys()), [])
    if isinstance(n, SYMPY_INTS):
        for k in range(n, -1, -1):
            h.append(f.get(k, K.zero) % p)
    else:
        (n,) = n
        for k in range(n, -1, -1):
            h.append(f.get((k,), K.zero) % p)
    return gf_trunc(h, p)

def gf_to_dict(f, p, symmetric=True):
    if False:
        return 10
    '\n    Convert a ``GF(p)[x]`` polynomial to a dict.\n\n    Examples\n    ========\n\n    >>> from sympy.polys.galoistools import gf_to_dict\n\n    >>> gf_to_dict([4, 0, 0, 0, 0, 0, 3, 0, 0, 0, 4], 5)\n    {0: -1, 4: -2, 10: -1}\n    >>> gf_to_dict([4, 0, 0, 0, 0, 0, 3, 0, 0, 0, 4], 5, symmetric=False)\n    {0: 4, 4: 3, 10: 4}\n\n    '
    (n, result) = (gf_degree(f), {})
    for k in range(0, n + 1):
        if symmetric:
            a = gf_int(f[n - k], p)
        else:
            a = f[n - k]
        if a:
            result[k] = a
    return result

def gf_from_int_poly(f, p):
    if False:
        return 10
    '\n    Create a ``GF(p)[x]`` polynomial from ``Z[x]``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys.galoistools import gf_from_int_poly\n\n    >>> gf_from_int_poly([7, -2, 3], 5)\n    [2, 3, 3]\n\n    '
    return gf_trunc(f, p)

def gf_to_int_poly(f, p, symmetric=True):
    if False:
        i = 10
        return i + 15
    '\n    Convert a ``GF(p)[x]`` polynomial to ``Z[x]``.\n\n\n    Examples\n    ========\n\n    >>> from sympy.polys.galoistools import gf_to_int_poly\n\n    >>> gf_to_int_poly([2, 3, 3], 5)\n    [2, -2, -2]\n    >>> gf_to_int_poly([2, 3, 3], 5, symmetric=False)\n    [2, 3, 3]\n\n    '
    if symmetric:
        return [gf_int(c, p) for c in f]
    else:
        return f

def gf_neg(f, p, K):
    if False:
        for i in range(10):
            print('nop')
    '\n    Negate a polynomial in ``GF(p)[x]``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys.domains import ZZ\n    >>> from sympy.polys.galoistools import gf_neg\n\n    >>> gf_neg([3, 2, 1, 0], 5, ZZ)\n    [2, 3, 4, 0]\n\n    '
    return [-coeff % p for coeff in f]

def gf_add_ground(f, a, p, K):
    if False:
        return 10
    '\n    Compute ``f + a`` where ``f`` in ``GF(p)[x]`` and ``a`` in ``GF(p)``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys.domains import ZZ\n    >>> from sympy.polys.galoistools import gf_add_ground\n\n    >>> gf_add_ground([3, 2, 4], 2, 5, ZZ)\n    [3, 2, 1]\n\n    '
    if not f:
        a = a % p
    else:
        a = (f[-1] + a) % p
        if len(f) > 1:
            return f[:-1] + [a]
    if not a:
        return []
    else:
        return [a]

def gf_sub_ground(f, a, p, K):
    if False:
        print('Hello World!')
    '\n    Compute ``f - a`` where ``f`` in ``GF(p)[x]`` and ``a`` in ``GF(p)``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys.domains import ZZ\n    >>> from sympy.polys.galoistools import gf_sub_ground\n\n    >>> gf_sub_ground([3, 2, 4], 2, 5, ZZ)\n    [3, 2, 2]\n\n    '
    if not f:
        a = -a % p
    else:
        a = (f[-1] - a) % p
        if len(f) > 1:
            return f[:-1] + [a]
    if not a:
        return []
    else:
        return [a]

def gf_mul_ground(f, a, p, K):
    if False:
        for i in range(10):
            print('nop')
    '\n    Compute ``f * a`` where ``f`` in ``GF(p)[x]`` and ``a`` in ``GF(p)``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys.domains import ZZ\n    >>> from sympy.polys.galoistools import gf_mul_ground\n\n    >>> gf_mul_ground([3, 2, 4], 2, 5, ZZ)\n    [1, 4, 3]\n\n    '
    if not a:
        return []
    else:
        return [a * b % p for b in f]

def gf_quo_ground(f, a, p, K):
    if False:
        for i in range(10):
            print('nop')
    '\n    Compute ``f/a`` where ``f`` in ``GF(p)[x]`` and ``a`` in ``GF(p)``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys.domains import ZZ\n    >>> from sympy.polys.galoistools import gf_quo_ground\n\n    >>> gf_quo_ground(ZZ.map([3, 2, 4]), ZZ(2), 5, ZZ)\n    [4, 1, 2]\n\n    '
    return gf_mul_ground(f, K.invert(a, p), p, K)

def gf_add(f, g, p, K):
    if False:
        for i in range(10):
            print('nop')
    '\n    Add polynomials in ``GF(p)[x]``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys.domains import ZZ\n    >>> from sympy.polys.galoistools import gf_add\n\n    >>> gf_add([3, 2, 4], [2, 2, 2], 5, ZZ)\n    [4, 1]\n\n    '
    if not f:
        return g
    if not g:
        return f
    df = gf_degree(f)
    dg = gf_degree(g)
    if df == dg:
        return gf_strip([(a + b) % p for (a, b) in zip(f, g)])
    else:
        k = abs(df - dg)
        if df > dg:
            (h, f) = (f[:k], f[k:])
        else:
            (h, g) = (g[:k], g[k:])
        return h + [(a + b) % p for (a, b) in zip(f, g)]

def gf_sub(f, g, p, K):
    if False:
        return 10
    '\n    Subtract polynomials in ``GF(p)[x]``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys.domains import ZZ\n    >>> from sympy.polys.galoistools import gf_sub\n\n    >>> gf_sub([3, 2, 4], [2, 2, 2], 5, ZZ)\n    [1, 0, 2]\n\n    '
    if not g:
        return f
    if not f:
        return gf_neg(g, p, K)
    df = gf_degree(f)
    dg = gf_degree(g)
    if df == dg:
        return gf_strip([(a - b) % p for (a, b) in zip(f, g)])
    else:
        k = abs(df - dg)
        if df > dg:
            (h, f) = (f[:k], f[k:])
        else:
            (h, g) = (gf_neg(g[:k], p, K), g[k:])
        return h + [(a - b) % p for (a, b) in zip(f, g)]

def gf_mul(f, g, p, K):
    if False:
        while True:
            i = 10
    '\n    Multiply polynomials in ``GF(p)[x]``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys.domains import ZZ\n    >>> from sympy.polys.galoistools import gf_mul\n\n    >>> gf_mul([3, 2, 4], [2, 2, 2], 5, ZZ)\n    [1, 0, 3, 2, 3]\n\n    '
    df = gf_degree(f)
    dg = gf_degree(g)
    dh = df + dg
    h = [0] * (dh + 1)
    for i in range(0, dh + 1):
        coeff = K.zero
        for j in range(max(0, i - dg), min(i, df) + 1):
            coeff += f[j] * g[i - j]
        h[i] = coeff % p
    return gf_strip(h)

def gf_sqr(f, p, K):
    if False:
        while True:
            i = 10
    '\n    Square polynomials in ``GF(p)[x]``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys.domains import ZZ\n    >>> from sympy.polys.galoistools import gf_sqr\n\n    >>> gf_sqr([3, 2, 4], 5, ZZ)\n    [4, 2, 3, 1, 1]\n\n    '
    df = gf_degree(f)
    dh = 2 * df
    h = [0] * (dh + 1)
    for i in range(0, dh + 1):
        coeff = K.zero
        jmin = max(0, i - df)
        jmax = min(i, df)
        n = jmax - jmin + 1
        jmax = jmin + n // 2 - 1
        for j in range(jmin, jmax + 1):
            coeff += f[j] * f[i - j]
        coeff += coeff
        if n & 1:
            elem = f[jmax + 1]
            coeff += elem ** 2
        h[i] = coeff % p
    return gf_strip(h)

def gf_add_mul(f, g, h, p, K):
    if False:
        return 10
    '\n    Returns ``f + g*h`` where ``f``, ``g``, ``h`` in ``GF(p)[x]``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys.domains import ZZ\n    >>> from sympy.polys.galoistools import gf_add_mul\n    >>> gf_add_mul([3, 2, 4], [2, 2, 2], [1, 4], 5, ZZ)\n    [2, 3, 2, 2]\n    '
    return gf_add(f, gf_mul(g, h, p, K), p, K)

def gf_sub_mul(f, g, h, p, K):
    if False:
        return 10
    '\n    Compute ``f - g*h`` where ``f``, ``g``, ``h`` in ``GF(p)[x]``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys.domains import ZZ\n    >>> from sympy.polys.galoistools import gf_sub_mul\n\n    >>> gf_sub_mul([3, 2, 4], [2, 2, 2], [1, 4], 5, ZZ)\n    [3, 3, 2, 1]\n\n    '
    return gf_sub(f, gf_mul(g, h, p, K), p, K)

def gf_expand(F, p, K):
    if False:
        print('Hello World!')
    '\n    Expand results of :func:`~.factor` in ``GF(p)[x]``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys.domains import ZZ\n    >>> from sympy.polys.galoistools import gf_expand\n\n    >>> gf_expand([([3, 2, 4], 1), ([2, 2], 2), ([3, 1], 3)], 5, ZZ)\n    [4, 3, 0, 3, 0, 1, 4, 1]\n\n    '
    if isinstance(F, tuple):
        (lc, F) = F
    else:
        lc = K.one
    g = [lc]
    for (f, k) in F:
        f = gf_pow(f, k, p, K)
        g = gf_mul(g, f, p, K)
    return g

def gf_div(f, g, p, K):
    if False:
        for i in range(10):
            print('nop')
    '\n    Division with remainder in ``GF(p)[x]``.\n\n    Given univariate polynomials ``f`` and ``g`` with coefficients in a\n    finite field with ``p`` elements, returns polynomials ``q`` and ``r``\n    (quotient and remainder) such that ``f = q*g + r``.\n\n    Consider polynomials ``x**3 + x + 1`` and ``x**2 + x`` in GF(2)::\n\n       >>> from sympy.polys.domains import ZZ\n       >>> from sympy.polys.galoistools import gf_div, gf_add_mul\n\n       >>> gf_div(ZZ.map([1, 0, 1, 1]), ZZ.map([1, 1, 0]), 2, ZZ)\n       ([1, 1], [1])\n\n    As result we obtained quotient ``x + 1`` and remainder ``1``, thus::\n\n       >>> gf_add_mul(ZZ.map([1]), ZZ.map([1, 1]), ZZ.map([1, 1, 0]), 2, ZZ)\n       [1, 0, 1, 1]\n\n    References\n    ==========\n\n    .. [1] [Monagan93]_\n    .. [2] [Gathen99]_\n\n    '
    df = gf_degree(f)
    dg = gf_degree(g)
    if not g:
        raise ZeroDivisionError('polynomial division')
    elif df < dg:
        return ([], f)
    inv = K.invert(g[0], p)
    (h, dq, dr) = (list(f), df - dg, dg - 1)
    for i in range(0, df + 1):
        coeff = h[i]
        for j in range(max(0, dg - i), min(df - i, dr) + 1):
            coeff -= h[i + j - dg] * g[dg - j]
        if i <= dq:
            coeff *= inv
        h[i] = coeff % p
    return (h[:dq + 1], gf_strip(h[dq + 1:]))

def gf_rem(f, g, p, K):
    if False:
        print('Hello World!')
    '\n    Compute polynomial remainder in ``GF(p)[x]``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys.domains import ZZ\n    >>> from sympy.polys.galoistools import gf_rem\n\n    >>> gf_rem(ZZ.map([1, 0, 1, 1]), ZZ.map([1, 1, 0]), 2, ZZ)\n    [1]\n\n    '
    return gf_div(f, g, p, K)[1]

def gf_quo(f, g, p, K):
    if False:
        return 10
    '\n    Compute exact quotient in ``GF(p)[x]``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys.domains import ZZ\n    >>> from sympy.polys.galoistools import gf_quo\n\n    >>> gf_quo(ZZ.map([1, 0, 1, 1]), ZZ.map([1, 1, 0]), 2, ZZ)\n    [1, 1]\n    >>> gf_quo(ZZ.map([1, 0, 3, 2, 3]), ZZ.map([2, 2, 2]), 5, ZZ)\n    [3, 2, 4]\n\n    '
    df = gf_degree(f)
    dg = gf_degree(g)
    if not g:
        raise ZeroDivisionError('polynomial division')
    elif df < dg:
        return []
    inv = K.invert(g[0], p)
    (h, dq, dr) = (f[:], df - dg, dg - 1)
    for i in range(0, dq + 1):
        coeff = h[i]
        for j in range(max(0, dg - i), min(df - i, dr) + 1):
            coeff -= h[i + j - dg] * g[dg - j]
        h[i] = coeff * inv % p
    return h[:dq + 1]

def gf_exquo(f, g, p, K):
    if False:
        for i in range(10):
            print('nop')
    '\n    Compute polynomial quotient in ``GF(p)[x]``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys.domains import ZZ\n    >>> from sympy.polys.galoistools import gf_exquo\n\n    >>> gf_exquo(ZZ.map([1, 0, 3, 2, 3]), ZZ.map([2, 2, 2]), 5, ZZ)\n    [3, 2, 4]\n\n    >>> gf_exquo(ZZ.map([1, 0, 1, 1]), ZZ.map([1, 1, 0]), 2, ZZ)\n    Traceback (most recent call last):\n    ...\n    ExactQuotientFailed: [1, 1, 0] does not divide [1, 0, 1, 1]\n\n    '
    (q, r) = gf_div(f, g, p, K)
    if not r:
        return q
    else:
        raise ExactQuotientFailed(f, g)

def gf_lshift(f, n, K):
    if False:
        for i in range(10):
            print('nop')
    '\n    Efficiently multiply ``f`` by ``x**n``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys.domains import ZZ\n    >>> from sympy.polys.galoistools import gf_lshift\n\n    >>> gf_lshift([3, 2, 4], 4, ZZ)\n    [3, 2, 4, 0, 0, 0, 0]\n\n    '
    if not f:
        return f
    else:
        return f + [K.zero] * n

def gf_rshift(f, n, K):
    if False:
        i = 10
        return i + 15
    '\n    Efficiently divide ``f`` by ``x**n``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys.domains import ZZ\n    >>> from sympy.polys.galoistools import gf_rshift\n\n    >>> gf_rshift([1, 2, 3, 4, 0], 3, ZZ)\n    ([1, 2], [3, 4, 0])\n\n    '
    if not n:
        return (f, [])
    else:
        return (f[:-n], f[-n:])

def gf_pow(f, n, p, K):
    if False:
        i = 10
        return i + 15
    '\n    Compute ``f**n`` in ``GF(p)[x]`` using repeated squaring.\n\n    Examples\n    ========\n\n    >>> from sympy.polys.domains import ZZ\n    >>> from sympy.polys.galoistools import gf_pow\n\n    >>> gf_pow([3, 2, 4], 3, 5, ZZ)\n    [2, 4, 4, 2, 2, 1, 4]\n\n    '
    if not n:
        return [K.one]
    elif n == 1:
        return f
    elif n == 2:
        return gf_sqr(f, p, K)
    h = [K.one]
    while True:
        if n & 1:
            h = gf_mul(h, f, p, K)
            n -= 1
        n >>= 1
        if not n:
            break
        f = gf_sqr(f, p, K)
    return h

def gf_frobenius_monomial_base(g, p, K):
    if False:
        return 10
    '\n    return the list of ``x**(i*p) mod g in Z_p`` for ``i = 0, .., n - 1``\n    where ``n = gf_degree(g)``\n\n    Examples\n    ========\n\n    >>> from sympy.polys.domains import ZZ\n    >>> from sympy.polys.galoistools import gf_frobenius_monomial_base\n    >>> g = ZZ.map([1, 0, 2, 1])\n    >>> gf_frobenius_monomial_base(g, 5, ZZ)\n    [[1], [4, 4, 2], [1, 2]]\n\n    '
    n = gf_degree(g)
    if n == 0:
        return []
    b = [0] * n
    b[0] = [1]
    if p < n:
        for i in range(1, n):
            mon = gf_lshift(b[i - 1], p, K)
            b[i] = gf_rem(mon, g, p, K)
    elif n > 1:
        b[1] = gf_pow_mod([K.one, K.zero], p, g, p, K)
        for i in range(2, n):
            b[i] = gf_mul(b[i - 1], b[1], p, K)
            b[i] = gf_rem(b[i], g, p, K)
    return b

def gf_frobenius_map(f, g, b, p, K):
    if False:
        return 10
    '\n    compute gf_pow_mod(f, p, g, p, K) using the Frobenius map\n\n    Parameters\n    ==========\n\n    f, g : polynomials in ``GF(p)[x]``\n    b : frobenius monomial base\n    p : prime number\n    K : domain\n\n    Examples\n    ========\n\n    >>> from sympy.polys.domains import ZZ\n    >>> from sympy.polys.galoistools import gf_frobenius_monomial_base, gf_frobenius_map\n    >>> f = ZZ.map([2, 1, 0, 1])\n    >>> g = ZZ.map([1, 0, 2, 1])\n    >>> p = 5\n    >>> b = gf_frobenius_monomial_base(g, p, ZZ)\n    >>> r = gf_frobenius_map(f, g, b, p, ZZ)\n    >>> gf_frobenius_map(f, g, b, p, ZZ)\n    [4, 0, 3]\n    '
    m = gf_degree(g)
    if gf_degree(f) >= m:
        f = gf_rem(f, g, p, K)
    if not f:
        return []
    n = gf_degree(f)
    sf = [f[-1]]
    for i in range(1, n + 1):
        v = gf_mul_ground(b[i], f[n - i], p, K)
        sf = gf_add(sf, v, p, K)
    return sf

def _gf_pow_pnm1d2(f, n, g, b, p, K):
    if False:
        return 10
    '\n    utility function for ``gf_edf_zassenhaus``\n    Compute ``f**((p**n - 1) // 2)`` in ``GF(p)[x]/(g)``\n    ``f**((p**n - 1) // 2) = (f*f**p*...*f**(p**n - 1))**((p - 1) // 2)``\n    '
    f = gf_rem(f, g, p, K)
    h = f
    r = f
    for i in range(1, n):
        h = gf_frobenius_map(h, g, b, p, K)
        r = gf_mul(r, h, p, K)
        r = gf_rem(r, g, p, K)
    res = gf_pow_mod(r, (p - 1) // 2, g, p, K)
    return res

def gf_pow_mod(f, n, g, p, K):
    if False:
        while True:
            i = 10
    '\n    Compute ``f**n`` in ``GF(p)[x]/(g)`` using repeated squaring.\n\n    Given polynomials ``f`` and ``g`` in ``GF(p)[x]`` and a non-negative\n    integer ``n``, efficiently computes ``f**n (mod g)`` i.e. the remainder\n    of ``f**n`` from division by ``g``, using the repeated squaring algorithm.\n\n    Examples\n    ========\n\n    >>> from sympy.polys.domains import ZZ\n    >>> from sympy.polys.galoistools import gf_pow_mod\n\n    >>> gf_pow_mod(ZZ.map([3, 2, 4]), 3, ZZ.map([1, 1]), 5, ZZ)\n    []\n\n    References\n    ==========\n\n    .. [1] [Gathen99]_\n\n    '
    if not n:
        return [K.one]
    elif n == 1:
        return gf_rem(f, g, p, K)
    elif n == 2:
        return gf_rem(gf_sqr(f, p, K), g, p, K)
    h = [K.one]
    while True:
        if n & 1:
            h = gf_mul(h, f, p, K)
            h = gf_rem(h, g, p, K)
            n -= 1
        n >>= 1
        if not n:
            break
        f = gf_sqr(f, p, K)
        f = gf_rem(f, g, p, K)
    return h

def gf_gcd(f, g, p, K):
    if False:
        return 10
    '\n    Euclidean Algorithm in ``GF(p)[x]``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys.domains import ZZ\n    >>> from sympy.polys.galoistools import gf_gcd\n\n    >>> gf_gcd(ZZ.map([3, 2, 4]), ZZ.map([2, 2, 3]), 5, ZZ)\n    [1, 3]\n\n    '
    while g:
        (f, g) = (g, gf_rem(f, g, p, K))
    return gf_monic(f, p, K)[1]

def gf_lcm(f, g, p, K):
    if False:
        i = 10
        return i + 15
    '\n    Compute polynomial LCM in ``GF(p)[x]``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys.domains import ZZ\n    >>> from sympy.polys.galoistools import gf_lcm\n\n    >>> gf_lcm(ZZ.map([3, 2, 4]), ZZ.map([2, 2, 3]), 5, ZZ)\n    [1, 2, 0, 4]\n\n    '
    if not f or not g:
        return []
    h = gf_quo(gf_mul(f, g, p, K), gf_gcd(f, g, p, K), p, K)
    return gf_monic(h, p, K)[1]

def gf_cofactors(f, g, p, K):
    if False:
        return 10
    '\n    Compute polynomial GCD and cofactors in ``GF(p)[x]``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys.domains import ZZ\n    >>> from sympy.polys.galoistools import gf_cofactors\n\n    >>> gf_cofactors(ZZ.map([3, 2, 4]), ZZ.map([2, 2, 3]), 5, ZZ)\n    ([1, 3], [3, 3], [2, 1])\n\n    '
    if not f and (not g):
        return ([], [], [])
    h = gf_gcd(f, g, p, K)
    return (h, gf_quo(f, h, p, K), gf_quo(g, h, p, K))

def gf_gcdex(f, g, p, K):
    if False:
        i = 10
        return i + 15
    '\n    Extended Euclidean Algorithm in ``GF(p)[x]``.\n\n    Given polynomials ``f`` and ``g`` in ``GF(p)[x]``, computes polynomials\n    ``s``, ``t`` and ``h``, such that ``h = gcd(f, g)`` and ``s*f + t*g = h``.\n    The typical application of EEA is solving polynomial diophantine equations.\n\n    Consider polynomials ``f = (x + 7) (x + 1)``, ``g = (x + 7) (x**2 + 1)``\n    in ``GF(11)[x]``. Application of Extended Euclidean Algorithm gives::\n\n       >>> from sympy.polys.domains import ZZ\n       >>> from sympy.polys.galoistools import gf_gcdex, gf_mul, gf_add\n\n       >>> s, t, g = gf_gcdex(ZZ.map([1, 8, 7]), ZZ.map([1, 7, 1, 7]), 11, ZZ)\n       >>> s, t, g\n       ([5, 6], [6], [1, 7])\n\n    As result we obtained polynomials ``s = 5*x + 6`` and ``t = 6``, and\n    additionally ``gcd(f, g) = x + 7``. This is correct because::\n\n       >>> S = gf_mul(s, ZZ.map([1, 8, 7]), 11, ZZ)\n       >>> T = gf_mul(t, ZZ.map([1, 7, 1, 7]), 11, ZZ)\n\n       >>> gf_add(S, T, 11, ZZ) == [1, 7]\n       True\n\n    References\n    ==========\n\n    .. [1] [Gathen99]_\n\n    '
    if not (f or g):
        return ([K.one], [], [])
    (p0, r0) = gf_monic(f, p, K)
    (p1, r1) = gf_monic(g, p, K)
    if not f:
        return ([], [K.invert(p1, p)], r1)
    if not g:
        return ([K.invert(p0, p)], [], r0)
    (s0, s1) = ([K.invert(p0, p)], [])
    (t0, t1) = ([], [K.invert(p1, p)])
    while True:
        (Q, R) = gf_div(r0, r1, p, K)
        if not R:
            break
        ((lc, r1), r0) = (gf_monic(R, p, K), r1)
        inv = K.invert(lc, p)
        s = gf_sub_mul(s0, s1, Q, p, K)
        t = gf_sub_mul(t0, t1, Q, p, K)
        (s1, s0) = (gf_mul_ground(s, inv, p, K), s1)
        (t1, t0) = (gf_mul_ground(t, inv, p, K), t1)
    return (s1, t1, r1)

def gf_monic(f, p, K):
    if False:
        i = 10
        return i + 15
    '\n    Compute LC and a monic polynomial in ``GF(p)[x]``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys.domains import ZZ\n    >>> from sympy.polys.galoistools import gf_monic\n\n    >>> gf_monic(ZZ.map([3, 2, 4]), 5, ZZ)\n    (3, [1, 4, 3])\n\n    '
    if not f:
        return (K.zero, [])
    else:
        lc = f[0]
        if K.is_one(lc):
            return (lc, list(f))
        else:
            return (lc, gf_quo_ground(f, lc, p, K))

def gf_diff(f, p, K):
    if False:
        return 10
    '\n    Differentiate polynomial in ``GF(p)[x]``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys.domains import ZZ\n    >>> from sympy.polys.galoistools import gf_diff\n\n    >>> gf_diff([3, 2, 4], 5, ZZ)\n    [1, 2]\n\n    '
    df = gf_degree(f)
    (h, n) = ([K.zero] * df, df)
    for coeff in f[:-1]:
        coeff *= K(n)
        coeff %= p
        if coeff:
            h[df - n] = coeff
        n -= 1
    return gf_strip(h)

def gf_eval(f, a, p, K):
    if False:
        for i in range(10):
            print('nop')
    '\n    Evaluate ``f(a)`` in ``GF(p)`` using Horner scheme.\n\n    Examples\n    ========\n\n    >>> from sympy.polys.domains import ZZ\n    >>> from sympy.polys.galoistools import gf_eval\n\n    >>> gf_eval([3, 2, 4], 2, 5, ZZ)\n    0\n\n    '
    result = K.zero
    for c in f:
        result *= a
        result += c
        result %= p
    return result

def gf_multi_eval(f, A, p, K):
    if False:
        for i in range(10):
            print('nop')
    '\n    Evaluate ``f(a)`` for ``a`` in ``[a_1, ..., a_n]``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys.domains import ZZ\n    >>> from sympy.polys.galoistools import gf_multi_eval\n\n    >>> gf_multi_eval([3, 2, 4], [0, 1, 2, 3, 4], 5, ZZ)\n    [4, 4, 0, 2, 0]\n\n    '
    return [gf_eval(f, a, p, K) for a in A]

def gf_compose(f, g, p, K):
    if False:
        i = 10
        return i + 15
    '\n    Compute polynomial composition ``f(g)`` in ``GF(p)[x]``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys.domains import ZZ\n    >>> from sympy.polys.galoistools import gf_compose\n\n    >>> gf_compose([3, 2, 4], [2, 2, 2], 5, ZZ)\n    [2, 4, 0, 3, 0]\n\n    '
    if len(g) <= 1:
        return gf_strip([gf_eval(f, gf_LC(g, K), p, K)])
    if not f:
        return []
    h = [f[0]]
    for c in f[1:]:
        h = gf_mul(h, g, p, K)
        h = gf_add_ground(h, c, p, K)
    return h

def gf_compose_mod(g, h, f, p, K):
    if False:
        print('Hello World!')
    '\n    Compute polynomial composition ``g(h)`` in ``GF(p)[x]/(f)``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys.domains import ZZ\n    >>> from sympy.polys.galoistools import gf_compose_mod\n\n    >>> gf_compose_mod(ZZ.map([3, 2, 4]), ZZ.map([2, 2, 2]), ZZ.map([4, 3]), 5, ZZ)\n    [4]\n\n    '
    if not g:
        return []
    comp = [g[0]]
    for a in g[1:]:
        comp = gf_mul(comp, h, p, K)
        comp = gf_add_ground(comp, a, p, K)
        comp = gf_rem(comp, f, p, K)
    return comp

def gf_trace_map(a, b, c, n, f, p, K):
    if False:
        return 10
    '\n    Compute polynomial trace map in ``GF(p)[x]/(f)``.\n\n    Given a polynomial ``f`` in ``GF(p)[x]``, polynomials ``a``, ``b``,\n    ``c`` in the quotient ring ``GF(p)[x]/(f)`` such that ``b = c**t\n    (mod f)`` for some positive power ``t`` of ``p``, and a positive\n    integer ``n``, returns a mapping::\n\n       a -> a**t**n, a + a**t + a**t**2 + ... + a**t**n (mod f)\n\n    In factorization context, ``b = x**p mod f`` and ``c = x mod f``.\n    This way we can efficiently compute trace polynomials in equal\n    degree factorization routine, much faster than with other methods,\n    like iterated Frobenius algorithm, for large degrees.\n\n    Examples\n    ========\n\n    >>> from sympy.polys.domains import ZZ\n    >>> from sympy.polys.galoistools import gf_trace_map\n\n    >>> gf_trace_map([1, 2], [4, 4], [1, 1], 4, [3, 2, 4], 5, ZZ)\n    ([1, 3], [1, 3])\n\n    References\n    ==========\n\n    .. [1] [Gathen92]_\n\n    '
    u = gf_compose_mod(a, b, f, p, K)
    v = b
    if n & 1:
        U = gf_add(a, u, p, K)
        V = b
    else:
        U = a
        V = c
    n >>= 1
    while n:
        u = gf_add(u, gf_compose_mod(u, v, f, p, K), p, K)
        v = gf_compose_mod(v, v, f, p, K)
        if n & 1:
            U = gf_add(U, gf_compose_mod(u, V, f, p, K), p, K)
            V = gf_compose_mod(v, V, f, p, K)
        n >>= 1
    return (gf_compose_mod(a, V, f, p, K), U)

def _gf_trace_map(f, n, g, b, p, K):
    if False:
        return 10
    '\n    utility for ``gf_edf_shoup``\n    '
    f = gf_rem(f, g, p, K)
    h = f
    r = f
    for i in range(1, n):
        h = gf_frobenius_map(h, g, b, p, K)
        r = gf_add(r, h, p, K)
        r = gf_rem(r, g, p, K)
    return r

def gf_random(n, p, K):
    if False:
        while True:
            i = 10
    '\n    Generate a random polynomial in ``GF(p)[x]`` of degree ``n``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys.domains import ZZ\n    >>> from sympy.polys.galoistools import gf_random\n    >>> gf_random(10, 5, ZZ) #doctest: +SKIP\n    [1, 2, 3, 2, 1, 1, 1, 2, 0, 4, 2]\n\n    '
    pi = int(p)
    return [K.one] + [K(int(uniform(0, pi))) for i in range(0, n)]

def gf_irreducible(n, p, K):
    if False:
        i = 10
        return i + 15
    '\n    Generate random irreducible polynomial of degree ``n`` in ``GF(p)[x]``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys.domains import ZZ\n    >>> from sympy.polys.galoistools import gf_irreducible\n    >>> gf_irreducible(10, 5, ZZ) #doctest: +SKIP\n    [1, 4, 2, 2, 3, 2, 4, 1, 4, 0, 4]\n\n    '
    while True:
        f = gf_random(n, p, K)
        if gf_irreducible_p(f, p, K):
            return f

def gf_irred_p_ben_or(f, p, K):
    if False:
        for i in range(10):
            print('nop')
    "\n    Ben-Or's polynomial irreducibility test over finite fields.\n\n    Examples\n    ========\n\n    >>> from sympy.polys.domains import ZZ\n    >>> from sympy.polys.galoistools import gf_irred_p_ben_or\n\n    >>> gf_irred_p_ben_or(ZZ.map([1, 4, 2, 2, 3, 2, 4, 1, 4, 0, 4]), 5, ZZ)\n    True\n    >>> gf_irred_p_ben_or(ZZ.map([3, 2, 4]), 5, ZZ)\n    False\n\n    "
    n = gf_degree(f)
    if n <= 1:
        return True
    (_, f) = gf_monic(f, p, K)
    if n < 5:
        H = h = gf_pow_mod([K.one, K.zero], p, f, p, K)
        for i in range(0, n // 2):
            g = gf_sub(h, [K.one, K.zero], p, K)
            if gf_gcd(f, g, p, K) == [K.one]:
                h = gf_compose_mod(h, H, f, p, K)
            else:
                return False
    else:
        b = gf_frobenius_monomial_base(f, p, K)
        H = h = gf_frobenius_map([K.one, K.zero], f, b, p, K)
        for i in range(0, n // 2):
            g = gf_sub(h, [K.one, K.zero], p, K)
            if gf_gcd(f, g, p, K) == [K.one]:
                h = gf_frobenius_map(h, f, b, p, K)
            else:
                return False
    return True

def gf_irred_p_rabin(f, p, K):
    if False:
        i = 10
        return i + 15
    "\n    Rabin's polynomial irreducibility test over finite fields.\n\n    Examples\n    ========\n\n    >>> from sympy.polys.domains import ZZ\n    >>> from sympy.polys.galoistools import gf_irred_p_rabin\n\n    >>> gf_irred_p_rabin(ZZ.map([1, 4, 2, 2, 3, 2, 4, 1, 4, 0, 4]), 5, ZZ)\n    True\n    >>> gf_irred_p_rabin(ZZ.map([3, 2, 4]), 5, ZZ)\n    False\n\n    "
    n = gf_degree(f)
    if n <= 1:
        return True
    (_, f) = gf_monic(f, p, K)
    x = [K.one, K.zero]
    from sympy.ntheory import factorint
    indices = {n // d for d in factorint(n)}
    b = gf_frobenius_monomial_base(f, p, K)
    h = b[1]
    for i in range(1, n):
        if i in indices:
            g = gf_sub(h, x, p, K)
            if gf_gcd(f, g, p, K) != [K.one]:
                return False
        h = gf_frobenius_map(h, f, b, p, K)
    return h == x
_irred_methods = {'ben-or': gf_irred_p_ben_or, 'rabin': gf_irred_p_rabin}

def gf_irreducible_p(f, p, K):
    if False:
        return 10
    '\n    Test irreducibility of a polynomial ``f`` in ``GF(p)[x]``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys.domains import ZZ\n    >>> from sympy.polys.galoistools import gf_irreducible_p\n\n    >>> gf_irreducible_p(ZZ.map([1, 4, 2, 2, 3, 2, 4, 1, 4, 0, 4]), 5, ZZ)\n    True\n    >>> gf_irreducible_p(ZZ.map([3, 2, 4]), 5, ZZ)\n    False\n\n    '
    method = query('GF_IRRED_METHOD')
    if method is not None:
        irred = _irred_methods[method](f, p, K)
    else:
        irred = gf_irred_p_rabin(f, p, K)
    return irred

def gf_sqf_p(f, p, K):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return ``True`` if ``f`` is square-free in ``GF(p)[x]``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys.domains import ZZ\n    >>> from sympy.polys.galoistools import gf_sqf_p\n\n    >>> gf_sqf_p(ZZ.map([3, 2, 4]), 5, ZZ)\n    True\n    >>> gf_sqf_p(ZZ.map([2, 4, 4, 2, 2, 1, 4]), 5, ZZ)\n    False\n\n    '
    (_, f) = gf_monic(f, p, K)
    if not f:
        return True
    else:
        return gf_gcd(f, gf_diff(f, p, K), p, K) == [K.one]

def gf_sqf_part(f, p, K):
    if False:
        print('Hello World!')
    '\n    Return square-free part of a ``GF(p)[x]`` polynomial.\n\n    Examples\n    ========\n\n    >>> from sympy.polys.domains import ZZ\n    >>> from sympy.polys.galoistools import gf_sqf_part\n\n    >>> gf_sqf_part(ZZ.map([1, 1, 3, 0, 1, 0, 2, 2, 1]), 5, ZZ)\n    [1, 4, 3]\n\n    '
    (_, sqf) = gf_sqf_list(f, p, K)
    g = [K.one]
    for (f, _) in sqf:
        g = gf_mul(g, f, p, K)
    return g

def gf_sqf_list(f, p, K, all=False):
    if False:
        return 10
    "\n    Return the square-free decomposition of a ``GF(p)[x]`` polynomial.\n\n    Given a polynomial ``f`` in ``GF(p)[x]``, returns the leading coefficient\n    of ``f`` and a square-free decomposition ``f_1**e_1 f_2**e_2 ... f_k**e_k``\n    such that all ``f_i`` are monic polynomials and ``(f_i, f_j)`` for ``i != j``\n    are co-prime and ``e_1 ... e_k`` are given in increasing order. All trivial\n    terms (i.e. ``f_i = 1``) are not included in the output.\n\n    Consider polynomial ``f = x**11 + 1`` over ``GF(11)[x]``::\n\n       >>> from sympy.polys.domains import ZZ\n\n       >>> from sympy.polys.galoistools import (\n       ...     gf_from_dict, gf_diff, gf_sqf_list, gf_pow,\n       ... )\n       ... # doctest: +NORMALIZE_WHITESPACE\n\n       >>> f = gf_from_dict({11: ZZ(1), 0: ZZ(1)}, 11, ZZ)\n\n    Note that ``f'(x) = 0``::\n\n       >>> gf_diff(f, 11, ZZ)\n       []\n\n    This phenomenon does not happen in characteristic zero. However we can\n    still compute square-free decomposition of ``f`` using ``gf_sqf()``::\n\n       >>> gf_sqf_list(f, 11, ZZ)\n       (1, [([1, 1], 11)])\n\n    We obtained factorization ``f = (x + 1)**11``. This is correct because::\n\n       >>> gf_pow([1, 1], 11, 11, ZZ) == f\n       True\n\n    References\n    ==========\n\n    .. [1] [Geddes92]_\n\n    "
    (n, sqf, factors, r) = (1, False, [], int(p))
    (lc, f) = gf_monic(f, p, K)
    if gf_degree(f) < 1:
        return (lc, [])
    while True:
        F = gf_diff(f, p, K)
        if F != []:
            g = gf_gcd(f, F, p, K)
            h = gf_quo(f, g, p, K)
            i = 1
            while h != [K.one]:
                G = gf_gcd(g, h, p, K)
                H = gf_quo(h, G, p, K)
                if gf_degree(H) > 0:
                    factors.append((H, i * n))
                (g, h, i) = (gf_quo(g, G, p, K), G, i + 1)
            if g == [K.one]:
                sqf = True
            else:
                f = g
        if not sqf:
            d = gf_degree(f) // r
            for i in range(0, d + 1):
                f[i] = f[i * r]
            (f, n) = (f[:d + 1], n * r)
        else:
            break
    if all:
        raise ValueError("'all=True' is not supported yet")
    return (lc, factors)

def gf_Qmatrix(f, p, K):
    if False:
        print('Hello World!')
    "\n    Calculate Berlekamp's ``Q`` matrix.\n\n    Examples\n    ========\n\n    >>> from sympy.polys.domains import ZZ\n    >>> from sympy.polys.galoistools import gf_Qmatrix\n\n    >>> gf_Qmatrix([3, 2, 4], 5, ZZ)\n    [[1, 0],\n     [3, 4]]\n\n    >>> gf_Qmatrix([1, 0, 0, 0, 1], 5, ZZ)\n    [[1, 0, 0, 0],\n     [0, 4, 0, 0],\n     [0, 0, 1, 0],\n     [0, 0, 0, 4]]\n\n    "
    (n, r) = (gf_degree(f), int(p))
    q = [K.one] + [K.zero] * (n - 1)
    Q = [list(q)] + [[]] * (n - 1)
    for i in range(1, (n - 1) * r + 1):
        (qq, c) = ([-q[-1] * f[-1] % p], q[-1])
        for j in range(1, n):
            qq.append((q[j - 1] - c * f[-j - 1]) % p)
        if not i % r:
            Q[i // r] = list(qq)
        q = qq
    return Q

def gf_Qbasis(Q, p, K):
    if False:
        for i in range(10):
            print('nop')
    '\n    Compute a basis of the kernel of ``Q``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys.domains import ZZ\n    >>> from sympy.polys.galoistools import gf_Qmatrix, gf_Qbasis\n\n    >>> gf_Qbasis(gf_Qmatrix([1, 0, 0, 0, 1], 5, ZZ), 5, ZZ)\n    [[1, 0, 0, 0], [0, 0, 1, 0]]\n\n    >>> gf_Qbasis(gf_Qmatrix([3, 2, 4], 5, ZZ), 5, ZZ)\n    [[1, 0]]\n\n    '
    (Q, n) = ([list(q) for q in Q], len(Q))
    for k in range(0, n):
        Q[k][k] = (Q[k][k] - K.one) % p
    for k in range(0, n):
        for i in range(k, n):
            if Q[k][i]:
                break
        else:
            continue
        inv = K.invert(Q[k][i], p)
        for j in range(0, n):
            Q[j][i] = Q[j][i] * inv % p
        for j in range(0, n):
            t = Q[j][k]
            Q[j][k] = Q[j][i]
            Q[j][i] = t
        for i in range(0, n):
            if i != k:
                q = Q[k][i]
                for j in range(0, n):
                    Q[j][i] = (Q[j][i] - Q[j][k] * q) % p
    for i in range(0, n):
        for j in range(0, n):
            if i == j:
                Q[i][j] = (K.one - Q[i][j]) % p
            else:
                Q[i][j] = -Q[i][j] % p
    basis = []
    for q in Q:
        if any(q):
            basis.append(q)
    return basis

def gf_berlekamp(f, p, K):
    if False:
        while True:
            i = 10
    '\n    Factor a square-free ``f`` in ``GF(p)[x]`` for small ``p``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys.domains import ZZ\n    >>> from sympy.polys.galoistools import gf_berlekamp\n\n    >>> gf_berlekamp([1, 0, 0, 0, 1], 5, ZZ)\n    [[1, 0, 2], [1, 0, 3]]\n\n    '
    Q = gf_Qmatrix(f, p, K)
    V = gf_Qbasis(Q, p, K)
    for (i, v) in enumerate(V):
        V[i] = gf_strip(list(reversed(v)))
    factors = [f]
    for k in range(1, len(V)):
        for f in list(factors):
            s = K.zero
            while s < p:
                g = gf_sub_ground(V[k], s, p, K)
                h = gf_gcd(f, g, p, K)
                if h != [K.one] and h != f:
                    factors.remove(f)
                    f = gf_quo(f, h, p, K)
                    factors.extend([f, h])
                if len(factors) == len(V):
                    return _sort_factors(factors, multiple=False)
                s += K.one
    return _sort_factors(factors, multiple=False)

def gf_ddf_zassenhaus(f, p, K):
    if False:
        return 10
    '\n    Cantor-Zassenhaus: Deterministic Distinct Degree Factorization\n\n    Given a monic square-free polynomial ``f`` in ``GF(p)[x]``, computes\n    partial distinct degree factorization ``f_1 ... f_d`` of ``f`` where\n    ``deg(f_i) != deg(f_j)`` for ``i != j``. The result is returned as a\n    list of pairs ``(f_i, e_i)`` where ``deg(f_i) > 0`` and ``e_i > 0``\n    is an argument to the equal degree factorization routine.\n\n    Consider the polynomial ``x**15 - 1`` in ``GF(11)[x]``::\n\n       >>> from sympy.polys.domains import ZZ\n       >>> from sympy.polys.galoistools import gf_from_dict\n\n       >>> f = gf_from_dict({15: ZZ(1), 0: ZZ(-1)}, 11, ZZ)\n\n    Distinct degree factorization gives::\n\n       >>> from sympy.polys.galoistools import gf_ddf_zassenhaus\n\n       >>> gf_ddf_zassenhaus(f, 11, ZZ)\n       [([1, 0, 0, 0, 0, 10], 1), ([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], 2)]\n\n    which means ``x**15 - 1 = (x**5 - 1) (x**10 + x**5 + 1)``. To obtain\n    factorization into irreducibles, use equal degree factorization\n    procedure (EDF) with each of the factors.\n\n    References\n    ==========\n\n    .. [1] [Gathen99]_\n    .. [2] [Geddes92]_\n\n    '
    (i, g, factors) = (1, [K.one, K.zero], [])
    b = gf_frobenius_monomial_base(f, p, K)
    while 2 * i <= gf_degree(f):
        g = gf_frobenius_map(g, f, b, p, K)
        h = gf_gcd(f, gf_sub(g, [K.one, K.zero], p, K), p, K)
        if h != [K.one]:
            factors.append((h, i))
            f = gf_quo(f, h, p, K)
            g = gf_rem(g, f, p, K)
            b = gf_frobenius_monomial_base(f, p, K)
        i += 1
    if f != [K.one]:
        return factors + [(f, gf_degree(f))]
    else:
        return factors

def gf_edf_zassenhaus(f, n, p, K):
    if False:
        for i in range(10):
            print('nop')
    "\n    Cantor-Zassenhaus: Probabilistic Equal Degree Factorization\n\n    Given a monic square-free polynomial ``f`` in ``GF(p)[x]`` and\n    an integer ``n``, such that ``n`` divides ``deg(f)``, returns all\n    irreducible factors ``f_1,...,f_d`` of ``f``, each of degree ``n``.\n    EDF procedure gives complete factorization over Galois fields.\n\n    Consider the square-free polynomial ``f = x**3 + x**2 + x + 1`` in\n    ``GF(5)[x]``. Let's compute its irreducible factors of degree one::\n\n       >>> from sympy.polys.domains import ZZ\n       >>> from sympy.polys.galoistools import gf_edf_zassenhaus\n\n       >>> gf_edf_zassenhaus([1,1,1,1], 1, 5, ZZ)\n       [[1, 1], [1, 2], [1, 3]]\n\n    Notes\n    =====\n\n    The case p == 2 is handled by Cohen's Algorithm 3.4.8. The case p odd is\n    as in Geddes Algorithm 8.9 (or Cohen's Algorithm 3.4.6).\n\n    References\n    ==========\n\n    .. [1] [Gathen99]_\n    .. [2] [Geddes92]_ Algorithm 8.9\n    .. [3] [Cohen93]_ Algorithm 3.4.8\n\n    "
    factors = [f]
    if gf_degree(f) <= n:
        return factors
    N = gf_degree(f) // n
    if p != 2:
        b = gf_frobenius_monomial_base(f, p, K)
    t = [K.one, K.zero]
    while len(factors) < N:
        if p == 2:
            h = r = t
            for i in range(n - 1):
                r = gf_pow_mod(r, 2, f, p, K)
                h = gf_add(h, r, p, K)
            g = gf_gcd(f, h, p, K)
            t += [K.zero, K.zero]
        else:
            r = gf_random(2 * n - 1, p, K)
            h = _gf_pow_pnm1d2(r, n, f, b, p, K)
            g = gf_gcd(f, gf_sub_ground(h, K.one, p, K), p, K)
        if g != [K.one] and g != f:
            factors = gf_edf_zassenhaus(g, n, p, K) + gf_edf_zassenhaus(gf_quo(f, g, p, K), n, p, K)
    return _sort_factors(factors, multiple=False)

def gf_ddf_shoup(f, p, K):
    if False:
        for i in range(10):
            print('nop')
    '\n    Kaltofen-Shoup: Deterministic Distinct Degree Factorization\n\n    Given a monic square-free polynomial ``f`` in ``GF(p)[x]``, computes\n    partial distinct degree factorization ``f_1,...,f_d`` of ``f`` where\n    ``deg(f_i) != deg(f_j)`` for ``i != j``. The result is returned as a\n    list of pairs ``(f_i, e_i)`` where ``deg(f_i) > 0`` and ``e_i > 0``\n    is an argument to the equal degree factorization routine.\n\n    This algorithm is an improved version of Zassenhaus algorithm for\n    large ``deg(f)`` and modulus ``p`` (especially for ``deg(f) ~ lg(p)``).\n\n    Examples\n    ========\n\n    >>> from sympy.polys.domains import ZZ\n    >>> from sympy.polys.galoistools import gf_ddf_shoup, gf_from_dict\n\n    >>> f = gf_from_dict({6: ZZ(1), 5: ZZ(-1), 4: ZZ(1), 3: ZZ(1), 1: ZZ(-1)}, 3, ZZ)\n\n    >>> gf_ddf_shoup(f, 3, ZZ)\n    [([1, 1, 0], 1), ([1, 1, 0, 1, 2], 2)]\n\n    References\n    ==========\n\n    .. [1] [Kaltofen98]_\n    .. [2] [Shoup95]_\n    .. [3] [Gathen92]_\n\n    '
    n = gf_degree(f)
    k = int(_ceil(_sqrt(n // 2)))
    b = gf_frobenius_monomial_base(f, p, K)
    h = gf_frobenius_map([K.one, K.zero], f, b, p, K)
    U = [[K.one, K.zero], h] + [K.zero] * (k - 1)
    for i in range(2, k + 1):
        U[i] = gf_frobenius_map(U[i - 1], f, b, p, K)
    (h, U) = (U[k], U[:k])
    V = [h] + [K.zero] * (k - 1)
    for i in range(1, k):
        V[i] = gf_compose_mod(V[i - 1], h, f, p, K)
    factors = []
    for (i, v) in enumerate(V):
        (h, j) = ([K.one], k - 1)
        for u in U:
            g = gf_sub(v, u, p, K)
            h = gf_mul(h, g, p, K)
            h = gf_rem(h, f, p, K)
        g = gf_gcd(f, h, p, K)
        f = gf_quo(f, g, p, K)
        for u in reversed(U):
            h = gf_sub(v, u, p, K)
            F = gf_gcd(g, h, p, K)
            if F != [K.one]:
                factors.append((F, k * (i + 1) - j))
            (g, j) = (gf_quo(g, F, p, K), j - 1)
    if f != [K.one]:
        factors.append((f, gf_degree(f)))
    return factors

def gf_edf_shoup(f, n, p, K):
    if False:
        while True:
            i = 10
    '\n    Gathen-Shoup: Probabilistic Equal Degree Factorization\n\n    Given a monic square-free polynomial ``f`` in ``GF(p)[x]`` and integer\n    ``n`` such that ``n`` divides ``deg(f)``, returns all irreducible factors\n    ``f_1,...,f_d`` of ``f``, each of degree ``n``. This is a complete\n    factorization over Galois fields.\n\n    This algorithm is an improved version of Zassenhaus algorithm for\n    large ``deg(f)`` and modulus ``p`` (especially for ``deg(f) ~ lg(p)``).\n\n    Examples\n    ========\n\n    >>> from sympy.polys.domains import ZZ\n    >>> from sympy.polys.galoistools import gf_edf_shoup\n\n    >>> gf_edf_shoup(ZZ.map([1, 2837, 2277]), 1, 2917, ZZ)\n    [[1, 852], [1, 1985]]\n\n    References\n    ==========\n\n    .. [1] [Shoup91]_\n    .. [2] [Gathen92]_\n\n    '
    (N, q) = (gf_degree(f), int(p))
    if not N:
        return []
    if N <= n:
        return [f]
    (factors, x) = ([f], [K.one, K.zero])
    r = gf_random(N - 1, p, K)
    if p == 2:
        h = gf_pow_mod(x, q, f, p, K)
        H = gf_trace_map(r, h, x, n - 1, f, p, K)[1]
        h1 = gf_gcd(f, H, p, K)
        h2 = gf_quo(f, h1, p, K)
        factors = gf_edf_shoup(h1, n, p, K) + gf_edf_shoup(h2, n, p, K)
    else:
        b = gf_frobenius_monomial_base(f, p, K)
        H = _gf_trace_map(r, n, f, b, p, K)
        h = gf_pow_mod(H, (q - 1) // 2, f, p, K)
        h1 = gf_gcd(f, h, p, K)
        h2 = gf_gcd(f, gf_sub_ground(h, K.one, p, K), p, K)
        h3 = gf_quo(f, gf_mul(h1, h2, p, K), p, K)
        factors = gf_edf_shoup(h1, n, p, K) + gf_edf_shoup(h2, n, p, K) + gf_edf_shoup(h3, n, p, K)
    return _sort_factors(factors, multiple=False)

def gf_zassenhaus(f, p, K):
    if False:
        for i in range(10):
            print('nop')
    '\n    Factor a square-free ``f`` in ``GF(p)[x]`` for medium ``p``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys.domains import ZZ\n    >>> from sympy.polys.galoistools import gf_zassenhaus\n\n    >>> gf_zassenhaus(ZZ.map([1, 4, 3]), 5, ZZ)\n    [[1, 1], [1, 3]]\n\n    '
    factors = []
    for (factor, n) in gf_ddf_zassenhaus(f, p, K):
        factors += gf_edf_zassenhaus(factor, n, p, K)
    return _sort_factors(factors, multiple=False)

def gf_shoup(f, p, K):
    if False:
        while True:
            i = 10
    '\n    Factor a square-free ``f`` in ``GF(p)[x]`` for large ``p``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys.domains import ZZ\n    >>> from sympy.polys.galoistools import gf_shoup\n\n    >>> gf_shoup(ZZ.map([1, 4, 3]), 5, ZZ)\n    [[1, 1], [1, 3]]\n\n    '
    factors = []
    for (factor, n) in gf_ddf_shoup(f, p, K):
        factors += gf_edf_shoup(factor, n, p, K)
    return _sort_factors(factors, multiple=False)
_factor_methods = {'berlekamp': gf_berlekamp, 'zassenhaus': gf_zassenhaus, 'shoup': gf_shoup}

def gf_factor_sqf(f, p, K, method=None):
    if False:
        while True:
            i = 10
    '\n    Factor a square-free polynomial ``f`` in ``GF(p)[x]``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys.domains import ZZ\n    >>> from sympy.polys.galoistools import gf_factor_sqf\n\n    >>> gf_factor_sqf(ZZ.map([3, 2, 4]), 5, ZZ)\n    (3, [[1, 1], [1, 3]])\n\n    '
    (lc, f) = gf_monic(f, p, K)
    if gf_degree(f) < 1:
        return (lc, [])
    method = method or query('GF_FACTOR_METHOD')
    if method is not None:
        factors = _factor_methods[method](f, p, K)
    else:
        factors = gf_zassenhaus(f, p, K)
    return (lc, factors)

def gf_factor(f, p, K):
    if False:
        print('Hello World!')
    '\n    Factor (non square-free) polynomials in ``GF(p)[x]``.\n\n    Given a possibly non square-free polynomial ``f`` in ``GF(p)[x]``,\n    returns its complete factorization into irreducibles::\n\n                 f_1(x)**e_1 f_2(x)**e_2 ... f_d(x)**e_d\n\n    where each ``f_i`` is a monic polynomial and ``gcd(f_i, f_j) == 1``,\n    for ``i != j``.  The result is given as a tuple consisting of the\n    leading coefficient of ``f`` and a list of factors of ``f`` with\n    their multiplicities.\n\n    The algorithm proceeds by first computing square-free decomposition\n    of ``f`` and then iteratively factoring each of square-free factors.\n\n    Consider a non square-free polynomial ``f = (7*x + 1) (x + 2)**2`` in\n    ``GF(11)[x]``. We obtain its factorization into irreducibles as follows::\n\n       >>> from sympy.polys.domains import ZZ\n       >>> from sympy.polys.galoistools import gf_factor\n\n       >>> gf_factor(ZZ.map([5, 2, 7, 2]), 11, ZZ)\n       (5, [([1, 2], 1), ([1, 8], 2)])\n\n    We arrived with factorization ``f = 5 (x + 2) (x + 8)**2``. We did not\n    recover the exact form of the input polynomial because we requested to\n    get monic factors of ``f`` and its leading coefficient separately.\n\n    Square-free factors of ``f`` can be factored into irreducibles over\n    ``GF(p)`` using three very different methods:\n\n    Berlekamp\n        efficient for very small values of ``p`` (usually ``p < 25``)\n    Cantor-Zassenhaus\n        efficient on average input and with "typical" ``p``\n    Shoup-Kaltofen-Gathen\n        efficient with very large inputs and modulus\n\n    If you want to use a specific factorization method, instead of the default\n    one, set ``GF_FACTOR_METHOD`` with one of ``berlekamp``, ``zassenhaus`` or\n    ``shoup`` values.\n\n    References\n    ==========\n\n    .. [1] [Gathen99]_\n\n    '
    (lc, f) = gf_monic(f, p, K)
    if gf_degree(f) < 1:
        return (lc, [])
    factors = []
    for (g, n) in gf_sqf_list(f, p, K)[1]:
        for h in gf_factor_sqf(g, p, K)[1]:
            factors.append((h, n))
    return (lc, _sort_factors(factors))

def gf_value(f, a):
    if False:
        print('Hello World!')
    "\n    Value of polynomial 'f' at 'a' in field R.\n\n    Examples\n    ========\n\n    >>> from sympy.polys.galoistools import gf_value\n\n    >>> gf_value([1, 7, 2, 4], 11)\n    2204\n\n    "
    result = 0
    for c in f:
        result *= a
        result += c
    return result

def linear_congruence(a, b, m):
    if False:
        while True:
            i = 10
    '\n    Returns the values of x satisfying a*x congruent b mod(m)\n\n    Here m is positive integer and a, b are natural numbers.\n    This function returns only those values of x which are distinct mod(m).\n\n    Examples\n    ========\n\n    >>> from sympy.polys.galoistools import linear_congruence\n\n    >>> linear_congruence(3, 12, 15)\n    [4, 9, 14]\n\n    There are 3 solutions distinct mod(15) since gcd(a, m) = gcd(3, 15) = 3.\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Linear_congruence_theorem\n\n    '
    from sympy.polys.polytools import gcdex
    if a % m == 0:
        if b % m == 0:
            return list(range(m))
        else:
            return []
    (r, _, g) = gcdex(a, m)
    if b % g != 0:
        return []
    return [(r * b // g + t * m // g) % m for t in range(g)]

def _raise_mod_power(x, s, p, f):
    if False:
        print('Hello World!')
    '\n    Used in gf_csolve to generate solutions of f(x) cong 0 mod(p**(s + 1))\n    from the solutions of f(x) cong 0 mod(p**s).\n\n    Examples\n    ========\n\n    >>> from sympy.polys.galoistools import _raise_mod_power\n    >>> from sympy.polys.galoistools import csolve_prime\n\n    These is the solutions of f(x) = x**2 + x + 7 cong 0 mod(3)\n\n    >>> f = [1, 1, 7]\n    >>> csolve_prime(f, 3)\n    [1]\n    >>> [ i for i in range(3) if not (i**2 + i + 7) % 3]\n    [1]\n\n    The solutions of f(x) cong 0 mod(9) are constructed from the\n    values returned from _raise_mod_power:\n\n    >>> x, s, p = 1, 1, 3\n    >>> V = _raise_mod_power(x, s, p, f)\n    >>> [x + v * p**s for v in V]\n    [1, 4, 7]\n\n    And these are confirmed with the following:\n\n    >>> [ i for i in range(3**2) if not (i**2 + i + 7) % 3**2]\n    [1, 4, 7]\n\n    '
    from sympy.polys.domains import ZZ
    f_f = gf_diff(f, p, ZZ)
    alpha = gf_value(f_f, x)
    beta = -gf_value(f, x) // p ** s
    return linear_congruence(alpha, beta, p)

def _csolve_prime_las_vegas(f, p, seed=None):
    if False:
        while True:
            i = 10
    ' Solutions of `f(x) \\equiv 0 \\pmod{p}`, `f(0) \\not\\equiv 0 \\pmod{p}`.\n\n    Explanation\n    ===========\n\n    This algorithm is classified as the Las Vegas method.\n    That is, it always returns the correct answer and solves the problem\n    fast in many cases, but if it is unlucky, it does not answer forever.\n\n    Suppose the polynomial f is not a zero polynomial. Assume further\n    that it is of degree at most p-1 and `f(0)\\not\\equiv 0 \\pmod{p}`.\n    These assumptions are not an essential part of the algorithm,\n    only that it is more convenient for the function calling this\n    function to resolve them.\n\n    Note that `x^{p-1} - 1 \\equiv \\prod_{a=1}^{p-1}(x - a) \\pmod{p}`.\n    Thus, the greatest common divisor with f is `\\prod_{s \\in S}(x - s)`,\n    with S being the set of solutions to f. Furthermore,\n    when a is randomly determined, `(x+a)^{(p-1)/2}-1` is\n    a polynomial with (p-1)/2 randomly chosen solutions.\n    The greatest common divisor of f may be a nontrivial factor of f.\n\n    When p is large and the degree of f is small,\n    it is faster than naive solution methods.\n\n    Parameters\n    ==========\n\n    f : polynomial\n    p : prime number\n\n    Returns\n    =======\n\n    list[int]\n        a list of solutions, sorted in ascending order\n        by integers in the range [1, p). The same value\n        does not exist in the list even if there is\n        a multiple solution. If no solution exists, returns [].\n\n    Examples\n    ========\n\n    >>> from sympy.polys.galoistools import _csolve_prime_las_vegas\n    >>> _csolve_prime_las_vegas([1, 4, 3], 7) # x^2 + 4x + 3 = 0 (mod 7)\n    [4, 6]\n    >>> _csolve_prime_las_vegas([5, 7, 1, 9], 11) # 5x^3 + 7x^2 + x + 9 = 0 (mod 11)\n    [1, 5, 8]\n\n    References\n    ==========\n\n    .. [1] R. Crandall and C. Pomerance "Prime Numbers", 2nd Ed., Algorithm 2.3.10\n\n    '
    from sympy.polys.domains import ZZ
    from sympy.ntheory import sqrt_mod
    randint = _randint(seed)
    root = set()
    g = gf_pow_mod([1, 0], p - 1, f, p, ZZ)
    g = gf_sub_ground(g, 1, p, ZZ)
    factors = [gf_gcd(f, g, p, ZZ)]
    while factors:
        f = factors.pop()
        if len(f) <= 1:
            continue
        if len(f) == 2:
            root.add(-invert(f[0], p) * f[1] % p)
            continue
        if len(f) == 3:
            inv = invert(f[0], p)
            b = f[1] * inv % p
            b = (b + p * (b % 2)) // 2
            root.update(((r - b) % p for r in sqrt_mod(b ** 2 - f[2] * inv, p, all_roots=True)))
            continue
        while True:
            a = randint(0, p - 1)
            g = gf_pow_mod([1, a], (p - 1) // 2, f, p, ZZ)
            g = gf_sub_ground(g, 1, p, ZZ)
            g = gf_gcd(f, g, p, ZZ)
            if 1 < len(g) < len(f):
                factors.append(g)
                factors.append(gf_div(f, g, p, ZZ)[0])
                break
    return sorted(root)

def csolve_prime(f, p, e=1):
    if False:
        while True:
            i = 10
    ' Solutions of `f(x) \\equiv 0 \\pmod{p^e}`.\n\n    Parameters\n    ==========\n\n    f : polynomial\n    p : prime number\n    e : positive integer\n\n    Returns\n    =======\n\n    list[int]\n        a list of solutions, sorted in ascending order\n        by integers in the range [1, p**e). The same value\n        does not exist in the list even if there is\n        a multiple solution. If no solution exists, returns [].\n\n    Examples\n    ========\n\n    >>> from sympy.polys.galoistools import csolve_prime\n    >>> csolve_prime([1, 1, 7], 3, 1)\n    [1]\n    >>> csolve_prime([1, 1, 7], 3, 2)\n    [1, 4, 7]\n\n    Solutions [7, 4, 1] (mod 3**2) are generated by ``_raise_mod_power()``\n    from solution [1] (mod 3).\n    '
    from sympy.polys.domains import ZZ
    g = [MPZ(int(c)) for c in f]
    for i in range(len(g) - p):
        g[i + p - 1] += g[i]
        g[i] = 0
    g = gf_trunc(g, p)
    k = 0
    while k < len(g) and g[len(g) - k - 1] == 0:
        k += 1
    if k:
        g = g[:-k]
        root_zero = [0]
    else:
        root_zero = []
    if g == []:
        X1 = list(range(p))
    elif len(g) ** 2 < p:
        X1 = root_zero + _csolve_prime_las_vegas(g, p)
    else:
        X1 = root_zero + [i for i in range(p) if gf_eval(g, i, p, ZZ) == 0]
    if e == 1:
        return X1
    X = []
    S = list(zip(X1, [1] * len(X1)))
    while S:
        (x, s) = S.pop()
        if s == e:
            X.append(x)
        else:
            s1 = s + 1
            ps = p ** s
            S.extend([(x + v * ps, s1) for v in _raise_mod_power(x, s, p, f)])
    return sorted(X)

def gf_csolve(f, n):
    if False:
        print('Hello World!')
    "\n    To solve f(x) congruent 0 mod(n).\n\n    n is divided into canonical factors and f(x) cong 0 mod(p**e) will be\n    solved for each factor. Applying the Chinese Remainder Theorem to the\n    results returns the final answers.\n\n    Examples\n    ========\n\n    Solve [1, 1, 7] congruent 0 mod(189):\n\n    >>> from sympy.polys.galoistools import gf_csolve\n    >>> gf_csolve([1, 1, 7], 189)\n    [13, 49, 76, 112, 139, 175]\n\n    See Also\n    ========\n\n    sympy.ntheory.residue_ntheory.polynomial_congruence : a higher level solving routine\n\n    References\n    ==========\n\n    .. [1] 'An introduction to the Theory of Numbers' 5th Edition by Ivan Niven,\n           Zuckerman and Montgomery.\n\n    "
    from sympy.polys.domains import ZZ
    from sympy.ntheory import factorint
    P = factorint(n)
    X = [csolve_prime(f, p, e) for (p, e) in P.items()]
    pools = list(map(tuple, X))
    perms = [[]]
    for pool in pools:
        perms = [x + [y] for x in perms for y in pool]
    dist_factors = [pow(p, e) for (p, e) in P.items()]
    return sorted([gf_crt(per, dist_factors, ZZ) for per in perms])