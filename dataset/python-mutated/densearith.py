"""Arithmetics for dense recursive polynomials in ``K[x]`` or ``K[X]``. """
from sympy.polys.densebasic import dup_slice, dup_LC, dmp_LC, dup_degree, dmp_degree, dup_strip, dmp_strip, dmp_zero_p, dmp_zero, dmp_one_p, dmp_one, dmp_ground, dmp_zeros
from sympy.polys.polyerrors import ExactQuotientFailed, PolynomialDivisionFailed

def dup_add_term(f, c, i, K):
    if False:
        for i in range(10):
            print('nop')
    '\n    Add ``c*x**i`` to ``f`` in ``K[x]``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys import ring, ZZ\n    >>> R, x = ring("x", ZZ)\n\n    >>> R.dup_add_term(x**2 - 1, ZZ(2), 4)\n    2*x**4 + x**2 - 1\n\n    '
    if not c:
        return f
    n = len(f)
    m = n - i - 1
    if i == n - 1:
        return dup_strip([f[0] + c] + f[1:])
    elif i >= n:
        return [c] + [K.zero] * (i - n) + f
    else:
        return f[:m] + [f[m] + c] + f[m + 1:]

def dmp_add_term(f, c, i, u, K):
    if False:
        for i in range(10):
            print('nop')
    '\n    Add ``c(x_2..x_u)*x_0**i`` to ``f`` in ``K[X]``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys import ring, ZZ\n    >>> R, x,y = ring("x,y", ZZ)\n\n    >>> R.dmp_add_term(x*y + 1, 2, 2)\n    2*x**2 + x*y + 1\n\n    '
    if not u:
        return dup_add_term(f, c, i, K)
    v = u - 1
    if dmp_zero_p(c, v):
        return f
    n = len(f)
    m = n - i - 1
    if i == n - 1:
        return dmp_strip([dmp_add(f[0], c, v, K)] + f[1:], u)
    elif i >= n:
        return [c] + dmp_zeros(i - n, v, K) + f
    else:
        return f[:m] + [dmp_add(f[m], c, v, K)] + f[m + 1:]

def dup_sub_term(f, c, i, K):
    if False:
        return 10
    '\n    Subtract ``c*x**i`` from ``f`` in ``K[x]``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys import ring, ZZ\n    >>> R, x = ring("x", ZZ)\n\n    >>> R.dup_sub_term(2*x**4 + x**2 - 1, ZZ(2), 4)\n    x**2 - 1\n\n    '
    if not c:
        return f
    n = len(f)
    m = n - i - 1
    if i == n - 1:
        return dup_strip([f[0] - c] + f[1:])
    elif i >= n:
        return [-c] + [K.zero] * (i - n) + f
    else:
        return f[:m] + [f[m] - c] + f[m + 1:]

def dmp_sub_term(f, c, i, u, K):
    if False:
        print('Hello World!')
    '\n    Subtract ``c(x_2..x_u)*x_0**i`` from ``f`` in ``K[X]``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys import ring, ZZ\n    >>> R, x,y = ring("x,y", ZZ)\n\n    >>> R.dmp_sub_term(2*x**2 + x*y + 1, 2, 2)\n    x*y + 1\n\n    '
    if not u:
        return dup_add_term(f, -c, i, K)
    v = u - 1
    if dmp_zero_p(c, v):
        return f
    n = len(f)
    m = n - i - 1
    if i == n - 1:
        return dmp_strip([dmp_sub(f[0], c, v, K)] + f[1:], u)
    elif i >= n:
        return [dmp_neg(c, v, K)] + dmp_zeros(i - n, v, K) + f
    else:
        return f[:m] + [dmp_sub(f[m], c, v, K)] + f[m + 1:]

def dup_mul_term(f, c, i, K):
    if False:
        while True:
            i = 10
    '\n    Multiply ``f`` by ``c*x**i`` in ``K[x]``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys import ring, ZZ\n    >>> R, x = ring("x", ZZ)\n\n    >>> R.dup_mul_term(x**2 - 1, ZZ(3), 2)\n    3*x**4 - 3*x**2\n\n    '
    if not c or not f:
        return []
    else:
        return [cf * c for cf in f] + [K.zero] * i

def dmp_mul_term(f, c, i, u, K):
    if False:
        return 10
    '\n    Multiply ``f`` by ``c(x_2..x_u)*x_0**i`` in ``K[X]``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys import ring, ZZ\n    >>> R, x,y = ring("x,y", ZZ)\n\n    >>> R.dmp_mul_term(x**2*y + x, 3*y, 2)\n    3*x**4*y**2 + 3*x**3*y\n\n    '
    if not u:
        return dup_mul_term(f, c, i, K)
    v = u - 1
    if dmp_zero_p(f, u):
        return f
    if dmp_zero_p(c, v):
        return dmp_zero(u)
    else:
        return [dmp_mul(cf, c, v, K) for cf in f] + dmp_zeros(i, v, K)

def dup_add_ground(f, c, K):
    if False:
        print('Hello World!')
    '\n    Add an element of the ground domain to ``f``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys import ring, ZZ\n    >>> R, x = ring("x", ZZ)\n\n    >>> R.dup_add_ground(x**3 + 2*x**2 + 3*x + 4, ZZ(4))\n    x**3 + 2*x**2 + 3*x + 8\n\n    '
    return dup_add_term(f, c, 0, K)

def dmp_add_ground(f, c, u, K):
    if False:
        for i in range(10):
            print('nop')
    '\n    Add an element of the ground domain to ``f``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys import ring, ZZ\n    >>> R, x,y = ring("x,y", ZZ)\n\n    >>> R.dmp_add_ground(x**3 + 2*x**2 + 3*x + 4, ZZ(4))\n    x**3 + 2*x**2 + 3*x + 8\n\n    '
    return dmp_add_term(f, dmp_ground(c, u - 1), 0, u, K)

def dup_sub_ground(f, c, K):
    if False:
        print('Hello World!')
    '\n    Subtract an element of the ground domain from ``f``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys import ring, ZZ\n    >>> R, x = ring("x", ZZ)\n\n    >>> R.dup_sub_ground(x**3 + 2*x**2 + 3*x + 4, ZZ(4))\n    x**3 + 2*x**2 + 3*x\n\n    '
    return dup_sub_term(f, c, 0, K)

def dmp_sub_ground(f, c, u, K):
    if False:
        print('Hello World!')
    '\n    Subtract an element of the ground domain from ``f``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys import ring, ZZ\n    >>> R, x,y = ring("x,y", ZZ)\n\n    >>> R.dmp_sub_ground(x**3 + 2*x**2 + 3*x + 4, ZZ(4))\n    x**3 + 2*x**2 + 3*x\n\n    '
    return dmp_sub_term(f, dmp_ground(c, u - 1), 0, u, K)

def dup_mul_ground(f, c, K):
    if False:
        print('Hello World!')
    '\n    Multiply ``f`` by a constant value in ``K[x]``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys import ring, ZZ\n    >>> R, x = ring("x", ZZ)\n\n    >>> R.dup_mul_ground(x**2 + 2*x - 1, ZZ(3))\n    3*x**2 + 6*x - 3\n\n    '
    if not c or not f:
        return []
    else:
        return [cf * c for cf in f]

def dmp_mul_ground(f, c, u, K):
    if False:
        i = 10
        return i + 15
    '\n    Multiply ``f`` by a constant value in ``K[X]``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys import ring, ZZ\n    >>> R, x,y = ring("x,y", ZZ)\n\n    >>> R.dmp_mul_ground(2*x + 2*y, ZZ(3))\n    6*x + 6*y\n\n    '
    if not u:
        return dup_mul_ground(f, c, K)
    v = u - 1
    return [dmp_mul_ground(cf, c, v, K) for cf in f]

def dup_quo_ground(f, c, K):
    if False:
        print('Hello World!')
    '\n    Quotient by a constant in ``K[x]``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys import ring, ZZ, QQ\n\n    >>> R, x = ring("x", ZZ)\n    >>> R.dup_quo_ground(3*x**2 + 2, ZZ(2))\n    x**2 + 1\n\n    >>> R, x = ring("x", QQ)\n    >>> R.dup_quo_ground(3*x**2 + 2, QQ(2))\n    3/2*x**2 + 1\n\n    '
    if not c:
        raise ZeroDivisionError('polynomial division')
    if not f:
        return f
    if K.is_Field:
        return [K.quo(cf, c) for cf in f]
    else:
        return [cf // c for cf in f]

def dmp_quo_ground(f, c, u, K):
    if False:
        i = 10
        return i + 15
    '\n    Quotient by a constant in ``K[X]``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys import ring, ZZ, QQ\n\n    >>> R, x,y = ring("x,y", ZZ)\n    >>> R.dmp_quo_ground(2*x**2*y + 3*x, ZZ(2))\n    x**2*y + x\n\n    >>> R, x,y = ring("x,y", QQ)\n    >>> R.dmp_quo_ground(2*x**2*y + 3*x, QQ(2))\n    x**2*y + 3/2*x\n\n    '
    if not u:
        return dup_quo_ground(f, c, K)
    v = u - 1
    return [dmp_quo_ground(cf, c, v, K) for cf in f]

def dup_exquo_ground(f, c, K):
    if False:
        for i in range(10):
            print('nop')
    '\n    Exact quotient by a constant in ``K[x]``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys import ring, QQ\n    >>> R, x = ring("x", QQ)\n\n    >>> R.dup_exquo_ground(x**2 + 2, QQ(2))\n    1/2*x**2 + 1\n\n    '
    if not c:
        raise ZeroDivisionError('polynomial division')
    if not f:
        return f
    return [K.exquo(cf, c) for cf in f]

def dmp_exquo_ground(f, c, u, K):
    if False:
        while True:
            i = 10
    '\n    Exact quotient by a constant in ``K[X]``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys import ring, QQ\n    >>> R, x,y = ring("x,y", QQ)\n\n    >>> R.dmp_exquo_ground(x**2*y + 2*x, QQ(2))\n    1/2*x**2*y + x\n\n    '
    if not u:
        return dup_exquo_ground(f, c, K)
    v = u - 1
    return [dmp_exquo_ground(cf, c, v, K) for cf in f]

def dup_lshift(f, n, K):
    if False:
        while True:
            i = 10
    '\n    Efficiently multiply ``f`` by ``x**n`` in ``K[x]``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys import ring, ZZ\n    >>> R, x = ring("x", ZZ)\n\n    >>> R.dup_lshift(x**2 + 1, 2)\n    x**4 + x**2\n\n    '
    if not f:
        return f
    else:
        return f + [K.zero] * n

def dup_rshift(f, n, K):
    if False:
        print('Hello World!')
    '\n    Efficiently divide ``f`` by ``x**n`` in ``K[x]``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys import ring, ZZ\n    >>> R, x = ring("x", ZZ)\n\n    >>> R.dup_rshift(x**4 + x**2, 2)\n    x**2 + 1\n    >>> R.dup_rshift(x**4 + x**2 + 2, 2)\n    x**2 + 1\n\n    '
    return f[:-n]

def dup_abs(f, K):
    if False:
        print('Hello World!')
    '\n    Make all coefficients positive in ``K[x]``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys import ring, ZZ\n    >>> R, x = ring("x", ZZ)\n\n    >>> R.dup_abs(x**2 - 1)\n    x**2 + 1\n\n    '
    return [K.abs(coeff) for coeff in f]

def dmp_abs(f, u, K):
    if False:
        return 10
    '\n    Make all coefficients positive in ``K[X]``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys import ring, ZZ\n    >>> R, x,y = ring("x,y", ZZ)\n\n    >>> R.dmp_abs(x**2*y - x)\n    x**2*y + x\n\n    '
    if not u:
        return dup_abs(f, K)
    v = u - 1
    return [dmp_abs(cf, v, K) for cf in f]

def dup_neg(f, K):
    if False:
        while True:
            i = 10
    '\n    Negate a polynomial in ``K[x]``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys import ring, ZZ\n    >>> R, x = ring("x", ZZ)\n\n    >>> R.dup_neg(x**2 - 1)\n    -x**2 + 1\n\n    '
    return [-coeff for coeff in f]

def dmp_neg(f, u, K):
    if False:
        print('Hello World!')
    '\n    Negate a polynomial in ``K[X]``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys import ring, ZZ\n    >>> R, x,y = ring("x,y", ZZ)\n\n    >>> R.dmp_neg(x**2*y - x)\n    -x**2*y + x\n\n    '
    if not u:
        return dup_neg(f, K)
    v = u - 1
    return [dmp_neg(cf, v, K) for cf in f]

def dup_add(f, g, K):
    if False:
        i = 10
        return i + 15
    '\n    Add dense polynomials in ``K[x]``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys import ring, ZZ\n    >>> R, x = ring("x", ZZ)\n\n    >>> R.dup_add(x**2 - 1, x - 2)\n    x**2 + x - 3\n\n    '
    if not f:
        return g
    if not g:
        return f
    df = dup_degree(f)
    dg = dup_degree(g)
    if df == dg:
        return dup_strip([a + b for (a, b) in zip(f, g)])
    else:
        k = abs(df - dg)
        if df > dg:
            (h, f) = (f[:k], f[k:])
        else:
            (h, g) = (g[:k], g[k:])
        return h + [a + b for (a, b) in zip(f, g)]

def dmp_add(f, g, u, K):
    if False:
        for i in range(10):
            print('nop')
    '\n    Add dense polynomials in ``K[X]``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys import ring, ZZ\n    >>> R, x,y = ring("x,y", ZZ)\n\n    >>> R.dmp_add(x**2 + y, x**2*y + x)\n    x**2*y + x**2 + x + y\n\n    '
    if not u:
        return dup_add(f, g, K)
    df = dmp_degree(f, u)
    if df < 0:
        return g
    dg = dmp_degree(g, u)
    if dg < 0:
        return f
    v = u - 1
    if df == dg:
        return dmp_strip([dmp_add(a, b, v, K) for (a, b) in zip(f, g)], u)
    else:
        k = abs(df - dg)
        if df > dg:
            (h, f) = (f[:k], f[k:])
        else:
            (h, g) = (g[:k], g[k:])
        return h + [dmp_add(a, b, v, K) for (a, b) in zip(f, g)]

def dup_sub(f, g, K):
    if False:
        return 10
    '\n    Subtract dense polynomials in ``K[x]``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys import ring, ZZ\n    >>> R, x = ring("x", ZZ)\n\n    >>> R.dup_sub(x**2 - 1, x - 2)\n    x**2 - x + 1\n\n    '
    if not f:
        return dup_neg(g, K)
    if not g:
        return f
    df = dup_degree(f)
    dg = dup_degree(g)
    if df == dg:
        return dup_strip([a - b for (a, b) in zip(f, g)])
    else:
        k = abs(df - dg)
        if df > dg:
            (h, f) = (f[:k], f[k:])
        else:
            (h, g) = (dup_neg(g[:k], K), g[k:])
        return h + [a - b for (a, b) in zip(f, g)]

def dmp_sub(f, g, u, K):
    if False:
        i = 10
        return i + 15
    '\n    Subtract dense polynomials in ``K[X]``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys import ring, ZZ\n    >>> R, x,y = ring("x,y", ZZ)\n\n    >>> R.dmp_sub(x**2 + y, x**2*y + x)\n    -x**2*y + x**2 - x + y\n\n    '
    if not u:
        return dup_sub(f, g, K)
    df = dmp_degree(f, u)
    if df < 0:
        return dmp_neg(g, u, K)
    dg = dmp_degree(g, u)
    if dg < 0:
        return f
    v = u - 1
    if df == dg:
        return dmp_strip([dmp_sub(a, b, v, K) for (a, b) in zip(f, g)], u)
    else:
        k = abs(df - dg)
        if df > dg:
            (h, f) = (f[:k], f[k:])
        else:
            (h, g) = (dmp_neg(g[:k], u, K), g[k:])
        return h + [dmp_sub(a, b, v, K) for (a, b) in zip(f, g)]

def dup_add_mul(f, g, h, K):
    if False:
        return 10
    '\n    Returns ``f + g*h`` where ``f, g, h`` are in ``K[x]``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys import ring, ZZ\n    >>> R, x = ring("x", ZZ)\n\n    >>> R.dup_add_mul(x**2 - 1, x - 2, x + 2)\n    2*x**2 - 5\n\n    '
    return dup_add(f, dup_mul(g, h, K), K)

def dmp_add_mul(f, g, h, u, K):
    if False:
        return 10
    '\n    Returns ``f + g*h`` where ``f, g, h`` are in ``K[X]``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys import ring, ZZ\n    >>> R, x,y = ring("x,y", ZZ)\n\n    >>> R.dmp_add_mul(x**2 + y, x, x + 2)\n    2*x**2 + 2*x + y\n\n    '
    return dmp_add(f, dmp_mul(g, h, u, K), u, K)

def dup_sub_mul(f, g, h, K):
    if False:
        return 10
    '\n    Returns ``f - g*h`` where ``f, g, h`` are in ``K[x]``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys import ring, ZZ\n    >>> R, x = ring("x", ZZ)\n\n    >>> R.dup_sub_mul(x**2 - 1, x - 2, x + 2)\n    3\n\n    '
    return dup_sub(f, dup_mul(g, h, K), K)

def dmp_sub_mul(f, g, h, u, K):
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns ``f - g*h`` where ``f, g, h`` are in ``K[X]``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys import ring, ZZ\n    >>> R, x,y = ring("x,y", ZZ)\n\n    >>> R.dmp_sub_mul(x**2 + y, x, x + 2)\n    -2*x + y\n\n    '
    return dmp_sub(f, dmp_mul(g, h, u, K), u, K)

def dup_mul(f, g, K):
    if False:
        print('Hello World!')
    '\n    Multiply dense polynomials in ``K[x]``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys import ring, ZZ\n    >>> R, x = ring("x", ZZ)\n\n    >>> R.dup_mul(x - 2, x + 2)\n    x**2 - 4\n\n    '
    if f == g:
        return dup_sqr(f, K)
    if not (f and g):
        return []
    df = dup_degree(f)
    dg = dup_degree(g)
    n = max(df, dg) + 1
    if n < 100:
        h = []
        for i in range(0, df + dg + 1):
            coeff = K.zero
            for j in range(max(0, i - dg), min(df, i) + 1):
                coeff += f[j] * g[i - j]
            h.append(coeff)
        return dup_strip(h)
    else:
        n2 = n // 2
        (fl, gl) = (dup_slice(f, 0, n2, K), dup_slice(g, 0, n2, K))
        fh = dup_rshift(dup_slice(f, n2, n, K), n2, K)
        gh = dup_rshift(dup_slice(g, n2, n, K), n2, K)
        (lo, hi) = (dup_mul(fl, gl, K), dup_mul(fh, gh, K))
        mid = dup_mul(dup_add(fl, fh, K), dup_add(gl, gh, K), K)
        mid = dup_sub(mid, dup_add(lo, hi, K), K)
        return dup_add(dup_add(lo, dup_lshift(mid, n2, K), K), dup_lshift(hi, 2 * n2, K), K)

def dmp_mul(f, g, u, K):
    if False:
        i = 10
        return i + 15
    '\n    Multiply dense polynomials in ``K[X]``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys import ring, ZZ\n    >>> R, x,y = ring("x,y", ZZ)\n\n    >>> R.dmp_mul(x*y + 1, x)\n    x**2*y + x\n\n    '
    if not u:
        return dup_mul(f, g, K)
    if f == g:
        return dmp_sqr(f, u, K)
    df = dmp_degree(f, u)
    if df < 0:
        return f
    dg = dmp_degree(g, u)
    if dg < 0:
        return g
    (h, v) = ([], u - 1)
    for i in range(0, df + dg + 1):
        coeff = dmp_zero(v)
        for j in range(max(0, i - dg), min(df, i) + 1):
            coeff = dmp_add(coeff, dmp_mul(f[j], g[i - j], v, K), v, K)
        h.append(coeff)
    return dmp_strip(h, u)

def dup_sqr(f, K):
    if False:
        print('Hello World!')
    '\n    Square dense polynomials in ``K[x]``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys import ring, ZZ\n    >>> R, x = ring("x", ZZ)\n\n    >>> R.dup_sqr(x**2 + 1)\n    x**4 + 2*x**2 + 1\n\n    '
    (df, h) = (len(f) - 1, [])
    for i in range(0, 2 * df + 1):
        c = K.zero
        jmin = max(0, i - df)
        jmax = min(i, df)
        n = jmax - jmin + 1
        jmax = jmin + n // 2 - 1
        for j in range(jmin, jmax + 1):
            c += f[j] * f[i - j]
        c += c
        if n & 1:
            elem = f[jmax + 1]
            c += elem ** 2
        h.append(c)
    return dup_strip(h)

def dmp_sqr(f, u, K):
    if False:
        while True:
            i = 10
    '\n    Square dense polynomials in ``K[X]``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys import ring, ZZ\n    >>> R, x,y = ring("x,y", ZZ)\n\n    >>> R.dmp_sqr(x**2 + x*y + y**2)\n    x**4 + 2*x**3*y + 3*x**2*y**2 + 2*x*y**3 + y**4\n\n    '
    if not u:
        return dup_sqr(f, K)
    df = dmp_degree(f, u)
    if df < 0:
        return f
    (h, v) = ([], u - 1)
    for i in range(0, 2 * df + 1):
        c = dmp_zero(v)
        jmin = max(0, i - df)
        jmax = min(i, df)
        n = jmax - jmin + 1
        jmax = jmin + n // 2 - 1
        for j in range(jmin, jmax + 1):
            c = dmp_add(c, dmp_mul(f[j], f[i - j], v, K), v, K)
        c = dmp_mul_ground(c, K(2), v, K)
        if n & 1:
            elem = dmp_sqr(f[jmax + 1], v, K)
            c = dmp_add(c, elem, v, K)
        h.append(c)
    return dmp_strip(h, u)

def dup_pow(f, n, K):
    if False:
        i = 10
        return i + 15
    '\n    Raise ``f`` to the ``n``-th power in ``K[x]``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys import ring, ZZ\n    >>> R, x = ring("x", ZZ)\n\n    >>> R.dup_pow(x - 2, 3)\n    x**3 - 6*x**2 + 12*x - 8\n\n    '
    if not n:
        return [K.one]
    if n < 0:
        raise ValueError('Cannot raise polynomial to a negative power')
    if n == 1 or not f or f == [K.one]:
        return f
    g = [K.one]
    while True:
        (n, m) = (n // 2, n)
        if m % 2:
            g = dup_mul(g, f, K)
            if not n:
                break
        f = dup_sqr(f, K)
    return g

def dmp_pow(f, n, u, K):
    if False:
        i = 10
        return i + 15
    '\n    Raise ``f`` to the ``n``-th power in ``K[X]``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys import ring, ZZ\n    >>> R, x,y = ring("x,y", ZZ)\n\n    >>> R.dmp_pow(x*y + 1, 3)\n    x**3*y**3 + 3*x**2*y**2 + 3*x*y + 1\n\n    '
    if not u:
        return dup_pow(f, n, K)
    if not n:
        return dmp_one(u, K)
    if n < 0:
        raise ValueError('Cannot raise polynomial to a negative power')
    if n == 1 or dmp_zero_p(f, u) or dmp_one_p(f, u, K):
        return f
    g = dmp_one(u, K)
    while True:
        (n, m) = (n // 2, n)
        if m & 1:
            g = dmp_mul(g, f, u, K)
            if not n:
                break
        f = dmp_sqr(f, u, K)
    return g

def dup_pdiv(f, g, K):
    if False:
        i = 10
        return i + 15
    '\n    Polynomial pseudo-division in ``K[x]``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys import ring, ZZ\n    >>> R, x = ring("x", ZZ)\n\n    >>> R.dup_pdiv(x**2 + 1, 2*x - 4)\n    (2*x + 4, 20)\n\n    '
    df = dup_degree(f)
    dg = dup_degree(g)
    (q, r, dr) = ([], f, df)
    if not g:
        raise ZeroDivisionError('polynomial division')
    elif df < dg:
        return (q, r)
    N = df - dg + 1
    lc_g = dup_LC(g, K)
    while True:
        lc_r = dup_LC(r, K)
        (j, N) = (dr - dg, N - 1)
        Q = dup_mul_ground(q, lc_g, K)
        q = dup_add_term(Q, lc_r, j, K)
        R = dup_mul_ground(r, lc_g, K)
        G = dup_mul_term(g, lc_r, j, K)
        r = dup_sub(R, G, K)
        (_dr, dr) = (dr, dup_degree(r))
        if dr < dg:
            break
        elif not dr < _dr:
            raise PolynomialDivisionFailed(f, g, K)
    c = lc_g ** N
    q = dup_mul_ground(q, c, K)
    r = dup_mul_ground(r, c, K)
    return (q, r)

def dup_prem(f, g, K):
    if False:
        return 10
    '\n    Polynomial pseudo-remainder in ``K[x]``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys import ring, ZZ\n    >>> R, x = ring("x", ZZ)\n\n    >>> R.dup_prem(x**2 + 1, 2*x - 4)\n    20\n\n    '
    df = dup_degree(f)
    dg = dup_degree(g)
    (r, dr) = (f, df)
    if not g:
        raise ZeroDivisionError('polynomial division')
    elif df < dg:
        return r
    N = df - dg + 1
    lc_g = dup_LC(g, K)
    while True:
        lc_r = dup_LC(r, K)
        (j, N) = (dr - dg, N - 1)
        R = dup_mul_ground(r, lc_g, K)
        G = dup_mul_term(g, lc_r, j, K)
        r = dup_sub(R, G, K)
        (_dr, dr) = (dr, dup_degree(r))
        if dr < dg:
            break
        elif not dr < _dr:
            raise PolynomialDivisionFailed(f, g, K)
    return dup_mul_ground(r, lc_g ** N, K)

def dup_pquo(f, g, K):
    if False:
        while True:
            i = 10
    '\n    Polynomial exact pseudo-quotient in ``K[X]``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys import ring, ZZ\n    >>> R, x = ring("x", ZZ)\n\n    >>> R.dup_pquo(x**2 - 1, 2*x - 2)\n    2*x + 2\n\n    >>> R.dup_pquo(x**2 + 1, 2*x - 4)\n    2*x + 4\n\n    '
    return dup_pdiv(f, g, K)[0]

def dup_pexquo(f, g, K):
    if False:
        return 10
    '\n    Polynomial pseudo-quotient in ``K[x]``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys import ring, ZZ\n    >>> R, x = ring("x", ZZ)\n\n    >>> R.dup_pexquo(x**2 - 1, 2*x - 2)\n    2*x + 2\n\n    >>> R.dup_pexquo(x**2 + 1, 2*x - 4)\n    Traceback (most recent call last):\n    ...\n    ExactQuotientFailed: [2, -4] does not divide [1, 0, 1]\n\n    '
    (q, r) = dup_pdiv(f, g, K)
    if not r:
        return q
    else:
        raise ExactQuotientFailed(f, g)

def dmp_pdiv(f, g, u, K):
    if False:
        print('Hello World!')
    '\n    Polynomial pseudo-division in ``K[X]``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys import ring, ZZ\n    >>> R, x,y = ring("x,y", ZZ)\n\n    >>> R.dmp_pdiv(x**2 + x*y, 2*x + 2)\n    (2*x + 2*y - 2, -4*y + 4)\n\n    '
    if not u:
        return dup_pdiv(f, g, K)
    df = dmp_degree(f, u)
    dg = dmp_degree(g, u)
    if dg < 0:
        raise ZeroDivisionError('polynomial division')
    (q, r, dr) = (dmp_zero(u), f, df)
    if df < dg:
        return (q, r)
    N = df - dg + 1
    lc_g = dmp_LC(g, K)
    while True:
        lc_r = dmp_LC(r, K)
        (j, N) = (dr - dg, N - 1)
        Q = dmp_mul_term(q, lc_g, 0, u, K)
        q = dmp_add_term(Q, lc_r, j, u, K)
        R = dmp_mul_term(r, lc_g, 0, u, K)
        G = dmp_mul_term(g, lc_r, j, u, K)
        r = dmp_sub(R, G, u, K)
        (_dr, dr) = (dr, dmp_degree(r, u))
        if dr < dg:
            break
        elif not dr < _dr:
            raise PolynomialDivisionFailed(f, g, K)
    c = dmp_pow(lc_g, N, u - 1, K)
    q = dmp_mul_term(q, c, 0, u, K)
    r = dmp_mul_term(r, c, 0, u, K)
    return (q, r)

def dmp_prem(f, g, u, K):
    if False:
        i = 10
        return i + 15
    '\n    Polynomial pseudo-remainder in ``K[X]``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys import ring, ZZ\n    >>> R, x,y = ring("x,y", ZZ)\n\n    >>> R.dmp_prem(x**2 + x*y, 2*x + 2)\n    -4*y + 4\n\n    '
    if not u:
        return dup_prem(f, g, K)
    df = dmp_degree(f, u)
    dg = dmp_degree(g, u)
    if dg < 0:
        raise ZeroDivisionError('polynomial division')
    (r, dr) = (f, df)
    if df < dg:
        return r
    N = df - dg + 1
    lc_g = dmp_LC(g, K)
    while True:
        lc_r = dmp_LC(r, K)
        (j, N) = (dr - dg, N - 1)
        R = dmp_mul_term(r, lc_g, 0, u, K)
        G = dmp_mul_term(g, lc_r, j, u, K)
        r = dmp_sub(R, G, u, K)
        (_dr, dr) = (dr, dmp_degree(r, u))
        if dr < dg:
            break
        elif not dr < _dr:
            raise PolynomialDivisionFailed(f, g, K)
    c = dmp_pow(lc_g, N, u - 1, K)
    return dmp_mul_term(r, c, 0, u, K)

def dmp_pquo(f, g, u, K):
    if False:
        for i in range(10):
            print('nop')
    '\n    Polynomial exact pseudo-quotient in ``K[X]``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys import ring, ZZ\n    >>> R, x,y = ring("x,y", ZZ)\n\n    >>> f = x**2 + x*y\n    >>> g = 2*x + 2*y\n    >>> h = 2*x + 2\n\n    >>> R.dmp_pquo(f, g)\n    2*x\n\n    >>> R.dmp_pquo(f, h)\n    2*x + 2*y - 2\n\n    '
    return dmp_pdiv(f, g, u, K)[0]

def dmp_pexquo(f, g, u, K):
    if False:
        print('Hello World!')
    '\n    Polynomial pseudo-quotient in ``K[X]``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys import ring, ZZ\n    >>> R, x,y = ring("x,y", ZZ)\n\n    >>> f = x**2 + x*y\n    >>> g = 2*x + 2*y\n    >>> h = 2*x + 2\n\n    >>> R.dmp_pexquo(f, g)\n    2*x\n\n    >>> R.dmp_pexquo(f, h)\n    Traceback (most recent call last):\n    ...\n    ExactQuotientFailed: [[2], [2]] does not divide [[1], [1, 0], []]\n\n    '
    (q, r) = dmp_pdiv(f, g, u, K)
    if dmp_zero_p(r, u):
        return q
    else:
        raise ExactQuotientFailed(f, g)

def dup_rr_div(f, g, K):
    if False:
        print('Hello World!')
    '\n    Univariate division with remainder over a ring.\n\n    Examples\n    ========\n\n    >>> from sympy.polys import ring, ZZ\n    >>> R, x = ring("x", ZZ)\n\n    >>> R.dup_rr_div(x**2 + 1, 2*x - 4)\n    (0, x**2 + 1)\n\n    '
    df = dup_degree(f)
    dg = dup_degree(g)
    (q, r, dr) = ([], f, df)
    if not g:
        raise ZeroDivisionError('polynomial division')
    elif df < dg:
        return (q, r)
    lc_g = dup_LC(g, K)
    while True:
        lc_r = dup_LC(r, K)
        if lc_r % lc_g:
            break
        c = K.exquo(lc_r, lc_g)
        j = dr - dg
        q = dup_add_term(q, c, j, K)
        h = dup_mul_term(g, c, j, K)
        r = dup_sub(r, h, K)
        (_dr, dr) = (dr, dup_degree(r))
        if dr < dg:
            break
        elif not dr < _dr:
            raise PolynomialDivisionFailed(f, g, K)
    return (q, r)

def dmp_rr_div(f, g, u, K):
    if False:
        while True:
            i = 10
    '\n    Multivariate division with remainder over a ring.\n\n    Examples\n    ========\n\n    >>> from sympy.polys import ring, ZZ\n    >>> R, x,y = ring("x,y", ZZ)\n\n    >>> R.dmp_rr_div(x**2 + x*y, 2*x + 2)\n    (0, x**2 + x*y)\n\n    '
    if not u:
        return dup_rr_div(f, g, K)
    df = dmp_degree(f, u)
    dg = dmp_degree(g, u)
    if dg < 0:
        raise ZeroDivisionError('polynomial division')
    (q, r, dr) = (dmp_zero(u), f, df)
    if df < dg:
        return (q, r)
    (lc_g, v) = (dmp_LC(g, K), u - 1)
    while True:
        lc_r = dmp_LC(r, K)
        (c, R) = dmp_rr_div(lc_r, lc_g, v, K)
        if not dmp_zero_p(R, v):
            break
        j = dr - dg
        q = dmp_add_term(q, c, j, u, K)
        h = dmp_mul_term(g, c, j, u, K)
        r = dmp_sub(r, h, u, K)
        (_dr, dr) = (dr, dmp_degree(r, u))
        if dr < dg:
            break
        elif not dr < _dr:
            raise PolynomialDivisionFailed(f, g, K)
    return (q, r)

def dup_ff_div(f, g, K):
    if False:
        i = 10
        return i + 15
    '\n    Polynomial division with remainder over a field.\n\n    Examples\n    ========\n\n    >>> from sympy.polys import ring, QQ\n    >>> R, x = ring("x", QQ)\n\n    >>> R.dup_ff_div(x**2 + 1, 2*x - 4)\n    (1/2*x + 1, 5)\n\n    '
    df = dup_degree(f)
    dg = dup_degree(g)
    (q, r, dr) = ([], f, df)
    if not g:
        raise ZeroDivisionError('polynomial division')
    elif df < dg:
        return (q, r)
    lc_g = dup_LC(g, K)
    while True:
        lc_r = dup_LC(r, K)
        c = K.exquo(lc_r, lc_g)
        j = dr - dg
        q = dup_add_term(q, c, j, K)
        h = dup_mul_term(g, c, j, K)
        r = dup_sub(r, h, K)
        (_dr, dr) = (dr, dup_degree(r))
        if dr < dg:
            break
        elif dr == _dr and (not K.is_Exact):
            r = dup_strip(r[1:])
            dr = dup_degree(r)
            if dr < dg:
                break
        elif not dr < _dr:
            raise PolynomialDivisionFailed(f, g, K)
    return (q, r)

def dmp_ff_div(f, g, u, K):
    if False:
        while True:
            i = 10
    '\n    Polynomial division with remainder over a field.\n\n    Examples\n    ========\n\n    >>> from sympy.polys import ring, QQ\n    >>> R, x,y = ring("x,y", QQ)\n\n    >>> R.dmp_ff_div(x**2 + x*y, 2*x + 2)\n    (1/2*x + 1/2*y - 1/2, -y + 1)\n\n    '
    if not u:
        return dup_ff_div(f, g, K)
    df = dmp_degree(f, u)
    dg = dmp_degree(g, u)
    if dg < 0:
        raise ZeroDivisionError('polynomial division')
    (q, r, dr) = (dmp_zero(u), f, df)
    if df < dg:
        return (q, r)
    (lc_g, v) = (dmp_LC(g, K), u - 1)
    while True:
        lc_r = dmp_LC(r, K)
        (c, R) = dmp_ff_div(lc_r, lc_g, v, K)
        if not dmp_zero_p(R, v):
            break
        j = dr - dg
        q = dmp_add_term(q, c, j, u, K)
        h = dmp_mul_term(g, c, j, u, K)
        r = dmp_sub(r, h, u, K)
        (_dr, dr) = (dr, dmp_degree(r, u))
        if dr < dg:
            break
        elif not dr < _dr:
            raise PolynomialDivisionFailed(f, g, K)
    return (q, r)

def dup_div(f, g, K):
    if False:
        return 10
    '\n    Polynomial division with remainder in ``K[x]``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys import ring, ZZ, QQ\n\n    >>> R, x = ring("x", ZZ)\n    >>> R.dup_div(x**2 + 1, 2*x - 4)\n    (0, x**2 + 1)\n\n    >>> R, x = ring("x", QQ)\n    >>> R.dup_div(x**2 + 1, 2*x - 4)\n    (1/2*x + 1, 5)\n\n    '
    if K.is_Field:
        return dup_ff_div(f, g, K)
    else:
        return dup_rr_div(f, g, K)

def dup_rem(f, g, K):
    if False:
        print('Hello World!')
    '\n    Returns polynomial remainder in ``K[x]``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys import ring, ZZ, QQ\n\n    >>> R, x = ring("x", ZZ)\n    >>> R.dup_rem(x**2 + 1, 2*x - 4)\n    x**2 + 1\n\n    >>> R, x = ring("x", QQ)\n    >>> R.dup_rem(x**2 + 1, 2*x - 4)\n    5\n\n    '
    return dup_div(f, g, K)[1]

def dup_quo(f, g, K):
    if False:
        i = 10
        return i + 15
    '\n    Returns exact polynomial quotient in ``K[x]``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys import ring, ZZ, QQ\n\n    >>> R, x = ring("x", ZZ)\n    >>> R.dup_quo(x**2 + 1, 2*x - 4)\n    0\n\n    >>> R, x = ring("x", QQ)\n    >>> R.dup_quo(x**2 + 1, 2*x - 4)\n    1/2*x + 1\n\n    '
    return dup_div(f, g, K)[0]

def dup_exquo(f, g, K):
    if False:
        i = 10
        return i + 15
    '\n    Returns polynomial quotient in ``K[x]``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys import ring, ZZ\n    >>> R, x = ring("x", ZZ)\n\n    >>> R.dup_exquo(x**2 - 1, x - 1)\n    x + 1\n\n    >>> R.dup_exquo(x**2 + 1, 2*x - 4)\n    Traceback (most recent call last):\n    ...\n    ExactQuotientFailed: [2, -4] does not divide [1, 0, 1]\n\n    '
    (q, r) = dup_div(f, g, K)
    if not r:
        return q
    else:
        raise ExactQuotientFailed(f, g)

def dmp_div(f, g, u, K):
    if False:
        return 10
    '\n    Polynomial division with remainder in ``K[X]``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys import ring, ZZ, QQ\n\n    >>> R, x,y = ring("x,y", ZZ)\n    >>> R.dmp_div(x**2 + x*y, 2*x + 2)\n    (0, x**2 + x*y)\n\n    >>> R, x,y = ring("x,y", QQ)\n    >>> R.dmp_div(x**2 + x*y, 2*x + 2)\n    (1/2*x + 1/2*y - 1/2, -y + 1)\n\n    '
    if K.is_Field:
        return dmp_ff_div(f, g, u, K)
    else:
        return dmp_rr_div(f, g, u, K)

def dmp_rem(f, g, u, K):
    if False:
        i = 10
        return i + 15
    '\n    Returns polynomial remainder in ``K[X]``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys import ring, ZZ, QQ\n\n    >>> R, x,y = ring("x,y", ZZ)\n    >>> R.dmp_rem(x**2 + x*y, 2*x + 2)\n    x**2 + x*y\n\n    >>> R, x,y = ring("x,y", QQ)\n    >>> R.dmp_rem(x**2 + x*y, 2*x + 2)\n    -y + 1\n\n    '
    return dmp_div(f, g, u, K)[1]

def dmp_quo(f, g, u, K):
    if False:
        print('Hello World!')
    '\n    Returns exact polynomial quotient in ``K[X]``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys import ring, ZZ, QQ\n\n    >>> R, x,y = ring("x,y", ZZ)\n    >>> R.dmp_quo(x**2 + x*y, 2*x + 2)\n    0\n\n    >>> R, x,y = ring("x,y", QQ)\n    >>> R.dmp_quo(x**2 + x*y, 2*x + 2)\n    1/2*x + 1/2*y - 1/2\n\n    '
    return dmp_div(f, g, u, K)[0]

def dmp_exquo(f, g, u, K):
    if False:
        print('Hello World!')
    '\n    Returns polynomial quotient in ``K[X]``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys import ring, ZZ\n    >>> R, x,y = ring("x,y", ZZ)\n\n    >>> f = x**2 + x*y\n    >>> g = x + y\n    >>> h = 2*x + 2\n\n    >>> R.dmp_exquo(f, g)\n    x\n\n    >>> R.dmp_exquo(f, h)\n    Traceback (most recent call last):\n    ...\n    ExactQuotientFailed: [[2], [2]] does not divide [[1], [1, 0], []]\n\n    '
    (q, r) = dmp_div(f, g, u, K)
    if dmp_zero_p(r, u):
        return q
    else:
        raise ExactQuotientFailed(f, g)

def dup_max_norm(f, K):
    if False:
        print('Hello World!')
    '\n    Returns maximum norm of a polynomial in ``K[x]``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys import ring, ZZ\n    >>> R, x = ring("x", ZZ)\n\n    >>> R.dup_max_norm(-x**2 + 2*x - 3)\n    3\n\n    '
    if not f:
        return K.zero
    else:
        return max(dup_abs(f, K))

def dmp_max_norm(f, u, K):
    if False:
        while True:
            i = 10
    '\n    Returns maximum norm of a polynomial in ``K[X]``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys import ring, ZZ\n    >>> R, x,y = ring("x,y", ZZ)\n\n    >>> R.dmp_max_norm(2*x*y - x - 3)\n    3\n\n    '
    if not u:
        return dup_max_norm(f, K)
    v = u - 1
    return max([dmp_max_norm(c, v, K) for c in f])

def dup_l1_norm(f, K):
    if False:
        print('Hello World!')
    '\n    Returns l1 norm of a polynomial in ``K[x]``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys import ring, ZZ\n    >>> R, x = ring("x", ZZ)\n\n    >>> R.dup_l1_norm(2*x**3 - 3*x**2 + 1)\n    6\n\n    '
    if not f:
        return K.zero
    else:
        return sum(dup_abs(f, K))

def dmp_l1_norm(f, u, K):
    if False:
        print('Hello World!')
    '\n    Returns l1 norm of a polynomial in ``K[X]``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys import ring, ZZ\n    >>> R, x,y = ring("x,y", ZZ)\n\n    >>> R.dmp_l1_norm(2*x*y - x - 3)\n    6\n\n    '
    if not u:
        return dup_l1_norm(f, K)
    v = u - 1
    return sum([dmp_l1_norm(c, v, K) for c in f])

def dup_l2_norm_squared(f, K):
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns squared l2 norm of a polynomial in ``K[x]``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys import ring, ZZ\n    >>> R, x = ring("x", ZZ)\n\n    >>> R.dup_l2_norm_squared(2*x**3 - 3*x**2 + 1)\n    14\n\n    '
    return sum([coeff ** 2 for coeff in f], K.zero)

def dmp_l2_norm_squared(f, u, K):
    if False:
        while True:
            i = 10
    '\n    Returns squared l2 norm of a polynomial in ``K[X]``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys import ring, ZZ\n    >>> R, x,y = ring("x,y", ZZ)\n\n    >>> R.dmp_l2_norm_squared(2*x*y - x - 3)\n    14\n\n    '
    if not u:
        return dup_l2_norm_squared(f, K)
    v = u - 1
    return sum([dmp_l2_norm_squared(c, v, K) for c in f])

def dup_expand(polys, K):
    if False:
        return 10
    '\n    Multiply together several polynomials in ``K[x]``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys import ring, ZZ\n    >>> R, x = ring("x", ZZ)\n\n    >>> R.dup_expand([x**2 - 1, x, 2])\n    2*x**3 - 2*x\n\n    '
    if not polys:
        return [K.one]
    f = polys[0]
    for g in polys[1:]:
        f = dup_mul(f, g, K)
    return f

def dmp_expand(polys, u, K):
    if False:
        print('Hello World!')
    '\n    Multiply together several polynomials in ``K[X]``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys import ring, ZZ\n    >>> R, x,y = ring("x,y", ZZ)\n\n    >>> R.dmp_expand([x**2 + y**2, x + 1])\n    x**3 + x**2 + x*y**2 + y**2\n\n    '
    if not polys:
        return dmp_one(u, K)
    f = polys[0]
    for g in polys[1:]:
        f = dmp_mul(f, g, u, K)
    return f