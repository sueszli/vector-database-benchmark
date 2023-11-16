"""
Algorithms for solving the Risch differential equation.

Given a differential field K of characteristic 0 that is a simple
monomial extension of a base field k and f, g in K, the Risch
Differential Equation problem is to decide if there exist y in K such
that Dy + f*y == g and to find one if there are some.  If t is a
monomial over k and the coefficients of f and g are in k(t), then y is
in k(t), and the outline of the algorithm here is given as:

1. Compute the normal part n of the denominator of y.  The problem is
then reduced to finding y' in k<t>, where y == y'/n.
2. Compute the special part s of the denominator of y.   The problem is
then reduced to finding y'' in k[t], where y == y''/(n*s)
3. Bound the degree of y''.
4. Reduce the equation Dy + f*y == g to a similar equation with f, g in
k[t].
5. Find the solutions in k[t] of bounded degree of the reduced equation.

See Chapter 6 of "Symbolic Integration I: Transcendental Functions" by
Manuel Bronstein.  See also the docstring of risch.py.
"""
from operator import mul
from functools import reduce
from sympy.core import oo
from sympy.core.symbol import Dummy
from sympy.polys import Poly, gcd, ZZ, cancel
from sympy.functions.elementary.complexes import im, re
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.integrals.risch import gcdex_diophantine, frac_in, derivation, splitfactor, NonElementaryIntegralException, DecrementLevel, recognize_log_derivative

def order_at(a, p, t):
    if False:
        for i in range(10):
            print('nop')
    '\n    Computes the order of a at p, with respect to t.\n\n    Explanation\n    ===========\n\n    For a, p in k[t], the order of a at p is defined as nu_p(a) = max({n\n    in Z+ such that p**n|a}), where a != 0.  If a == 0, nu_p(a) = +oo.\n\n    To compute the order at a rational function, a/b, use the fact that\n    nu_p(a/b) == nu_p(a) - nu_p(b).\n    '
    if a.is_zero:
        return oo
    if p == Poly(t, t):
        return a.as_poly(t).ET()[0][0]
    power_list = []
    p1 = p
    r = a.rem(p1)
    tracks_power = 1
    while r.is_zero:
        power_list.append((p1, tracks_power))
        p1 = p1 * p1
        tracks_power *= 2
        r = a.rem(p1)
    n = 0
    product = Poly(1, t)
    while len(power_list) != 0:
        final = power_list.pop()
        productf = product * final[0]
        r = a.rem(productf)
        if r.is_zero:
            n += final[1]
            product = productf
    return n

def order_at_oo(a, d, t):
    if False:
        return 10
    '\n    Computes the order of a/d at oo (infinity), with respect to t.\n\n    For f in k(t), the order or f at oo is defined as deg(d) - deg(a), where\n    f == a/d.\n    '
    if a.is_zero:
        return oo
    return d.degree(t) - a.degree(t)

def weak_normalizer(a, d, DE, z=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Weak normalization.\n\n    Explanation\n    ===========\n\n    Given a derivation D on k[t] and f == a/d in k(t), return q in k[t]\n    such that f - Dq/q is weakly normalized with respect to t.\n\n    f in k(t) is said to be "weakly normalized" with respect to t if\n    residue_p(f) is not a positive integer for any normal irreducible p\n    in k[t] such that f is in R_p (Definition 6.1.1).  If f has an\n    elementary integral, this is equivalent to no logarithm of\n    integral(f) whose argument depends on t has a positive integer\n    coefficient, where the arguments of the logarithms not in k(t) are\n    in k[t].\n\n    Returns (q, f - Dq/q)\n    '
    z = z or Dummy('z')
    (dn, ds) = splitfactor(d, DE)
    g = gcd(dn, dn.diff(DE.t))
    d_sqf_part = dn.quo(g)
    d1 = d_sqf_part.quo(gcd(d_sqf_part, g))
    (a1, b) = gcdex_diophantine(d.quo(d1).as_poly(DE.t), d1.as_poly(DE.t), a.as_poly(DE.t))
    r = (a - Poly(z, DE.t) * derivation(d1, DE)).as_poly(DE.t).resultant(d1.as_poly(DE.t))
    r = Poly(r, z)
    if not r.expr.has(z):
        return (Poly(1, DE.t), (a, d))
    N = [i for i in r.real_roots() if i in ZZ and i > 0]
    q = reduce(mul, [gcd(a - Poly(n, DE.t) * derivation(d1, DE), d1) for n in N], Poly(1, DE.t))
    dq = derivation(q, DE)
    sn = q * a - d * dq
    sd = q * d
    (sn, sd) = sn.cancel(sd, include=True)
    return (q, (sn, sd))

def normal_denom(fa, fd, ga, gd, DE):
    if False:
        print('Hello World!')
    '\n    Normal part of the denominator.\n\n    Explanation\n    ===========\n\n    Given a derivation D on k[t] and f, g in k(t) with f weakly\n    normalized with respect to t, either raise NonElementaryIntegralException,\n    in which case the equation Dy + f*y == g has no solution in k(t), or the\n    quadruplet (a, b, c, h) such that a, h in k[t], b, c in k<t>, and for any\n    solution y in k(t) of Dy + f*y == g, q = y*h in k<t> satisfies\n    a*Dq + b*q == c.\n\n    This constitutes step 1 in the outline given in the rde.py docstring.\n    '
    (dn, ds) = splitfactor(fd, DE)
    (en, es) = splitfactor(gd, DE)
    p = dn.gcd(en)
    h = en.gcd(en.diff(DE.t)).quo(p.gcd(p.diff(DE.t)))
    a = dn * h
    c = a * h
    if c.div(en)[1]:
        raise NonElementaryIntegralException
    ca = c * ga
    (ca, cd) = ca.cancel(gd, include=True)
    ba = a * fa - dn * derivation(h, DE) * fd
    (ba, bd) = ba.cancel(fd, include=True)
    return (a, (ba, bd), (ca, cd), h)

def special_denom(a, ba, bd, ca, cd, DE, case='auto'):
    if False:
        i = 10
        return i + 15
    "\n    Special part of the denominator.\n\n    Explanation\n    ===========\n\n    case is one of {'exp', 'tan', 'primitive'} for the hyperexponential,\n    hypertangent, and primitive cases, respectively.  For the\n    hyperexponential (resp. hypertangent) case, given a derivation D on\n    k[t] and a in k[t], b, c, in k<t> with Dt/t in k (resp. Dt/(t**2 + 1) in\n    k, sqrt(-1) not in k), a != 0, and gcd(a, t) == 1 (resp.\n    gcd(a, t**2 + 1) == 1), return the quadruplet (A, B, C, 1/h) such that\n    A, B, C, h in k[t] and for any solution q in k<t> of a*Dq + b*q == c,\n    r = qh in k[t] satisfies A*Dr + B*r == C.\n\n    For ``case == 'primitive'``, k<t> == k[t], so it returns (a, b, c, 1) in\n    this case.\n\n    This constitutes step 2 of the outline given in the rde.py docstring.\n    "
    if case == 'auto':
        case = DE.case
    if case == 'exp':
        p = Poly(DE.t, DE.t)
    elif case == 'tan':
        p = Poly(DE.t ** 2 + 1, DE.t)
    elif case in ('primitive', 'base'):
        B = ba.to_field().quo(bd)
        C = ca.to_field().quo(cd)
        return (a, B, C, Poly(1, DE.t))
    else:
        raise ValueError("case must be one of {'exp', 'tan', 'primitive', 'base'}, not %s." % case)
    nb = order_at(ba, p, DE.t) - order_at(bd, p, DE.t)
    nc = order_at(ca, p, DE.t) - order_at(cd, p, DE.t)
    n = min(0, nc - min(0, nb))
    if not nb:
        from .prde import parametric_log_deriv
        if case == 'exp':
            dcoeff = DE.d.quo(Poly(DE.t, DE.t))
            with DecrementLevel(DE):
                (alphaa, alphad) = frac_in(-ba.eval(0) / bd.eval(0) / a.eval(0), DE.t)
                (etaa, etad) = frac_in(dcoeff, DE.t)
                A = parametric_log_deriv(alphaa, alphad, etaa, etad, DE)
                if A is not None:
                    (Q, m, z) = A
                    if Q == 1:
                        n = min(n, m)
        elif case == 'tan':
            dcoeff = DE.d.quo(Poly(DE.t ** 2 + 1, DE.t))
            with DecrementLevel(DE):
                (alphaa, alphad) = frac_in(im(-ba.eval(sqrt(-1)) / bd.eval(sqrt(-1)) / a.eval(sqrt(-1))), DE.t)
                (betaa, betad) = frac_in(re(-ba.eval(sqrt(-1)) / bd.eval(sqrt(-1)) / a.eval(sqrt(-1))), DE.t)
                (etaa, etad) = frac_in(dcoeff, DE.t)
                if recognize_log_derivative(Poly(2, DE.t) * betaa, betad, DE):
                    A = parametric_log_deriv(alphaa * Poly(sqrt(-1), DE.t) * betad + alphad * betaa, alphad * betad, etaa, etad, DE)
                    if A is not None:
                        (Q, m, z) = A
                        if Q == 1:
                            n = min(n, m)
    N = max(0, -nb, n - nc)
    pN = p ** N
    pn = p ** (-n)
    A = a * pN
    B = ba * pN.quo(bd) + Poly(n, DE.t) * a * derivation(p, DE).quo(p) * pN
    C = (ca * pN * pn).quo(cd)
    h = pn
    return (A, B, C, h)

def bound_degree(a, b, cQ, DE, case='auto', parametric=False):
    if False:
        i = 10
        return i + 15
    '\n    Bound on polynomial solutions.\n\n    Explanation\n    ===========\n\n    Given a derivation D on k[t] and ``a``, ``b``, ``c`` in k[t] with ``a != 0``, return\n    n in ZZ such that deg(q) <= n for any solution q in k[t] of\n    a*Dq + b*q == c, when parametric=False, or deg(q) <= n for any solution\n    c1, ..., cm in Const(k) and q in k[t] of a*Dq + b*q == Sum(ci*gi, (i, 1, m))\n    when parametric=True.\n\n    For ``parametric=False``, ``cQ`` is ``c``, a ``Poly``; for ``parametric=True``, ``cQ`` is Q ==\n    [q1, ..., qm], a list of Polys.\n\n    This constitutes step 3 of the outline given in the rde.py docstring.\n    '
    if case == 'auto':
        case = DE.case
    da = a.degree(DE.t)
    db = b.degree(DE.t)
    if parametric:
        dc = max([i.degree(DE.t) for i in cQ])
    else:
        dc = cQ.degree(DE.t)
    alpha = cancel(-b.as_poly(DE.t).LC().as_expr() / a.as_poly(DE.t).LC().as_expr())
    if case == 'base':
        n = max(0, dc - max(db, da - 1))
        if db == da - 1 and alpha.is_Integer:
            n = max(0, alpha, dc - db)
    elif case == 'primitive':
        if db > da:
            n = max(0, dc - db)
        else:
            n = max(0, dc - da + 1)
        (etaa, etad) = frac_in(DE.d, DE.T[DE.level - 1])
        t1 = DE.t
        with DecrementLevel(DE):
            (alphaa, alphad) = frac_in(alpha, DE.t)
            if db == da - 1:
                from .prde import limited_integrate
                try:
                    ((za, zd), m) = limited_integrate(alphaa, alphad, [(etaa, etad)], DE)
                except NonElementaryIntegralException:
                    pass
                else:
                    if len(m) != 1:
                        raise ValueError('Length of m should be 1')
                    n = max(n, m[0])
            elif db == da:
                from .prde import is_log_deriv_k_t_radical_in_field
                A = is_log_deriv_k_t_radical_in_field(alphaa, alphad, DE)
                if A is not None:
                    (aa, z) = A
                    if aa == 1:
                        beta = -(a * derivation(z, DE).as_poly(t1) + b * z.as_poly(t1)).LC() / (z.as_expr() * a.LC())
                        (betaa, betad) = frac_in(beta, DE.t)
                        from .prde import limited_integrate
                        try:
                            ((za, zd), m) = limited_integrate(betaa, betad, [(etaa, etad)], DE)
                        except NonElementaryIntegralException:
                            pass
                        else:
                            if len(m) != 1:
                                raise ValueError('Length of m should be 1')
                            n = max(n, m[0].as_expr())
    elif case == 'exp':
        from .prde import parametric_log_deriv
        n = max(0, dc - max(db, da))
        if da == db:
            (etaa, etad) = frac_in(DE.d.quo(Poly(DE.t, DE.t)), DE.T[DE.level - 1])
            with DecrementLevel(DE):
                (alphaa, alphad) = frac_in(alpha, DE.t)
                A = parametric_log_deriv(alphaa, alphad, etaa, etad, DE)
                if A is not None:
                    (a, m, z) = A
                    if a == 1:
                        n = max(n, m)
    elif case in ('tan', 'other_nonlinear'):
        delta = DE.d.degree(DE.t)
        lam = DE.d.LC()
        alpha = cancel(alpha / lam)
        n = max(0, dc - max(da + delta - 1, db))
        if db == da + delta - 1 and alpha.is_Integer:
            n = max(0, alpha, dc - db)
    else:
        raise ValueError("case must be one of {'exp', 'tan', 'primitive', 'other_nonlinear', 'base'}, not %s." % case)
    return n

def spde(a, b, c, n, DE):
    if False:
        i = 10
        return i + 15
    "\n    Rothstein's Special Polynomial Differential Equation algorithm.\n\n    Explanation\n    ===========\n\n    Given a derivation D on k[t], an integer n and ``a``,``b``,``c`` in k[t] with\n    ``a != 0``, either raise NonElementaryIntegralException, in which case the\n    equation a*Dq + b*q == c has no solution of degree at most ``n`` in\n    k[t], or return the tuple (B, C, m, alpha, beta) such that B, C,\n    alpha, beta in k[t], m in ZZ, and any solution q in k[t] of degree\n    at most n of a*Dq + b*q == c must be of the form\n    q == alpha*h + beta, where h in k[t], deg(h) <= m, and Dh + B*h == C.\n\n    This constitutes step 4 of the outline given in the rde.py docstring.\n    "
    zero = Poly(0, DE.t)
    alpha = Poly(1, DE.t)
    beta = Poly(0, DE.t)
    while True:
        if c.is_zero:
            return (zero, zero, 0, zero, beta)
        if (n < 0) is True:
            raise NonElementaryIntegralException
        g = a.gcd(b)
        if not c.rem(g).is_zero:
            raise NonElementaryIntegralException
        (a, b, c) = (a.quo(g), b.quo(g), c.quo(g))
        if a.degree(DE.t) == 0:
            b = b.to_field().quo(a)
            c = c.to_field().quo(a)
            return (b, c, n, alpha, beta)
        (r, z) = gcdex_diophantine(b, a, c)
        b += derivation(a, DE)
        c = z - derivation(r, DE)
        n -= a.degree(DE.t)
        beta += alpha * r
        alpha *= a

def no_cancel_b_large(b, c, n, DE):
    if False:
        i = 10
        return i + 15
    '\n    Poly Risch Differential Equation - No cancellation: deg(b) large enough.\n\n    Explanation\n    ===========\n\n    Given a derivation D on k[t], ``n`` either an integer or +oo, and ``b``,``c``\n    in k[t] with ``b != 0`` and either D == d/dt or\n    deg(b) > max(0, deg(D) - 1), either raise NonElementaryIntegralException, in\n    which case the equation ``Dq + b*q == c`` has no solution of degree at\n    most n in k[t], or a solution q in k[t] of this equation with\n    ``deg(q) < n``.\n    '
    q = Poly(0, DE.t)
    while not c.is_zero:
        m = c.degree(DE.t) - b.degree(DE.t)
        if not 0 <= m <= n:
            raise NonElementaryIntegralException
        p = Poly(c.as_poly(DE.t).LC() / b.as_poly(DE.t).LC() * DE.t ** m, DE.t, expand=False)
        q = q + p
        n = m - 1
        c = c - derivation(p, DE) - b * p
    return q

def no_cancel_b_small(b, c, n, DE):
    if False:
        while True:
            i = 10
    '\n    Poly Risch Differential Equation - No cancellation: deg(b) small enough.\n\n    Explanation\n    ===========\n\n    Given a derivation D on k[t], ``n`` either an integer or +oo, and ``b``,``c``\n    in k[t] with deg(b) < deg(D) - 1 and either D == d/dt or\n    deg(D) >= 2, either raise NonElementaryIntegralException, in which case the\n    equation Dq + b*q == c has no solution of degree at most n in k[t],\n    or a solution q in k[t] of this equation with deg(q) <= n, or the\n    tuple (h, b0, c0) such that h in k[t], b0, c0, in k, and for any\n    solution q in k[t] of degree at most n of Dq + bq == c, y == q - h\n    is a solution in k of Dy + b0*y == c0.\n    '
    q = Poly(0, DE.t)
    while not c.is_zero:
        if n == 0:
            m = 0
        else:
            m = c.degree(DE.t) - DE.d.degree(DE.t) + 1
        if not 0 <= m <= n:
            raise NonElementaryIntegralException
        if m > 0:
            p = Poly(c.as_poly(DE.t).LC() / (m * DE.d.as_poly(DE.t).LC()) * DE.t ** m, DE.t, expand=False)
        else:
            if b.degree(DE.t) != c.degree(DE.t):
                raise NonElementaryIntegralException
            if b.degree(DE.t) == 0:
                return (q, b.as_poly(DE.T[DE.level - 1]), c.as_poly(DE.T[DE.level - 1]))
            p = Poly(c.as_poly(DE.t).LC() / b.as_poly(DE.t).LC(), DE.t, expand=False)
        q = q + p
        n = m - 1
        c = c - derivation(p, DE) - b * p
    return q

def no_cancel_equal(b, c, n, DE):
    if False:
        print('Hello World!')
    '\n    Poly Risch Differential Equation - No cancellation: deg(b) == deg(D) - 1\n\n    Explanation\n    ===========\n\n    Given a derivation D on k[t] with deg(D) >= 2, n either an integer\n    or +oo, and b, c in k[t] with deg(b) == deg(D) - 1, either raise\n    NonElementaryIntegralException, in which case the equation Dq + b*q == c has\n    no solution of degree at most n in k[t], or a solution q in k[t] of\n    this equation with deg(q) <= n, or the tuple (h, m, C) such that h\n    in k[t], m in ZZ, and C in k[t], and for any solution q in k[t] of\n    degree at most n of Dq + b*q == c, y == q - h is a solution in k[t]\n    of degree at most m of Dy + b*y == C.\n    '
    q = Poly(0, DE.t)
    lc = cancel(-b.as_poly(DE.t).LC() / DE.d.as_poly(DE.t).LC())
    if lc.is_Integer and lc.is_positive:
        M = lc
    else:
        M = -1
    while not c.is_zero:
        m = max(M, c.degree(DE.t) - DE.d.degree(DE.t) + 1)
        if not 0 <= m <= n:
            raise NonElementaryIntegralException
        u = cancel(m * DE.d.as_poly(DE.t).LC() + b.as_poly(DE.t).LC())
        if u.is_zero:
            return (q, m, c)
        if m > 0:
            p = Poly(c.as_poly(DE.t).LC() / u * DE.t ** m, DE.t, expand=False)
        elif c.degree(DE.t) != DE.d.degree(DE.t) - 1:
            raise NonElementaryIntegralException
        else:
            p = c.as_poly(DE.t).LC() / b.as_poly(DE.t).LC()
        q = q + p
        n = m - 1
        c = c - derivation(p, DE) - b * p
    return q

def cancel_primitive(b, c, n, DE):
    if False:
        print('Hello World!')
    '\n    Poly Risch Differential Equation - Cancellation: Primitive case.\n\n    Explanation\n    ===========\n\n    Given a derivation D on k[t], n either an integer or +oo, ``b`` in k, and\n    ``c`` in k[t] with Dt in k and ``b != 0``, either raise\n    NonElementaryIntegralException, in which case the equation Dq + b*q == c\n    has no solution of degree at most n in k[t], or a solution q in k[t] of\n    this equation with deg(q) <= n.\n    '
    from .prde import is_log_deriv_k_t_radical_in_field
    with DecrementLevel(DE):
        (ba, bd) = frac_in(b, DE.t)
        A = is_log_deriv_k_t_radical_in_field(ba, bd, DE)
        if A is not None:
            (n, z) = A
            if n == 1:
                raise NotImplementedError('is_deriv_in_field() is required to  solve this problem.')
    if c.is_zero:
        return c
    if n < c.degree(DE.t):
        raise NonElementaryIntegralException
    q = Poly(0, DE.t)
    while not c.is_zero:
        m = c.degree(DE.t)
        if n < m:
            raise NonElementaryIntegralException
        with DecrementLevel(DE):
            (a2a, a2d) = frac_in(c.LC(), DE.t)
            (sa, sd) = rischDE(ba, bd, a2a, a2d, DE)
        stm = Poly(sa.as_expr() / sd.as_expr() * DE.t ** m, DE.t, expand=False)
        q += stm
        n = m - 1
        c -= b * stm + derivation(stm, DE)
    return q

def cancel_exp(b, c, n, DE):
    if False:
        return 10
    '\n    Poly Risch Differential Equation - Cancellation: Hyperexponential case.\n\n    Explanation\n    ===========\n\n    Given a derivation D on k[t], n either an integer or +oo, ``b`` in k, and\n    ``c`` in k[t] with Dt/t in k and ``b != 0``, either raise\n    NonElementaryIntegralException, in which case the equation Dq + b*q == c\n    has no solution of degree at most n in k[t], or a solution q in k[t] of\n    this equation with deg(q) <= n.\n    '
    from .prde import parametric_log_deriv
    eta = DE.d.quo(Poly(DE.t, DE.t)).as_expr()
    with DecrementLevel(DE):
        (etaa, etad) = frac_in(eta, DE.t)
        (ba, bd) = frac_in(b, DE.t)
        A = parametric_log_deriv(ba, bd, etaa, etad, DE)
        if A is not None:
            (a, m, z) = A
            if a == 1:
                raise NotImplementedError('is_deriv_in_field() is required to solve this problem.')
    if c.is_zero:
        return c
    if n < c.degree(DE.t):
        raise NonElementaryIntegralException
    q = Poly(0, DE.t)
    while not c.is_zero:
        m = c.degree(DE.t)
        if n < m:
            raise NonElementaryIntegralException
        a1 = b.as_expr()
        with DecrementLevel(DE):
            (a1a, a1d) = frac_in(a1, DE.t)
            a1a = a1a * etad + etaa * a1d * Poly(m, DE.t)
            a1d = a1d * etad
            (a2a, a2d) = frac_in(c.LC(), DE.t)
            (sa, sd) = rischDE(a1a, a1d, a2a, a2d, DE)
        stm = Poly(sa.as_expr() / sd.as_expr() * DE.t ** m, DE.t, expand=False)
        q += stm
        n = m - 1
        c -= b * stm + derivation(stm, DE)
    return q

def solve_poly_rde(b, cQ, n, DE, parametric=False):
    if False:
        i = 10
        return i + 15
    '\n    Solve a Polynomial Risch Differential Equation with degree bound ``n``.\n\n    This constitutes step 4 of the outline given in the rde.py docstring.\n\n    For parametric=False, cQ is c, a Poly; for parametric=True, cQ is Q ==\n    [q1, ..., qm], a list of Polys.\n    '
    if not b.is_zero and (DE.case == 'base' or b.degree(DE.t) > max(0, DE.d.degree(DE.t) - 1)):
        if parametric:
            from .prde import prde_no_cancel_b_large
            return prde_no_cancel_b_large(b, cQ, n, DE)
        return no_cancel_b_large(b, cQ, n, DE)
    elif (b.is_zero or b.degree(DE.t) < DE.d.degree(DE.t) - 1) and (DE.case == 'base' or DE.d.degree(DE.t) >= 2):
        if parametric:
            from .prde import prde_no_cancel_b_small
            return prde_no_cancel_b_small(b, cQ, n, DE)
        R = no_cancel_b_small(b, cQ, n, DE)
        if isinstance(R, Poly):
            return R
        else:
            (h, b0, c0) = R
            with DecrementLevel(DE):
                (b0, c0) = (b0.as_poly(DE.t), c0.as_poly(DE.t))
                if b0 is None:
                    raise ValueError('b0 should be a non-Null value')
                if c0 is None:
                    raise ValueError('c0 should be a non-Null value')
                y = solve_poly_rde(b0, c0, n, DE).as_poly(DE.t)
            return h + y
    elif DE.d.degree(DE.t) >= 2 and b.degree(DE.t) == DE.d.degree(DE.t) - 1 and (n > -b.as_poly(DE.t).LC() / DE.d.as_poly(DE.t).LC()):
        if not b.as_poly(DE.t).LC().is_number:
            raise TypeError('Result should be a number')
        if parametric:
            raise NotImplementedError('prde_no_cancel_b_equal() is not yet implemented.')
        R = no_cancel_equal(b, cQ, n, DE)
        if isinstance(R, Poly):
            return R
        else:
            (h, m, C) = R
            y = solve_poly_rde(b, C, m, DE)
            return h + y
    else:
        if b.is_zero:
            raise NotImplementedError('Remaining cases for Poly (P)RDE are not yet implemented (is_deriv_in_field() required).')
        elif DE.case == 'exp':
            if parametric:
                raise NotImplementedError('Parametric RDE cancellation hyperexponential case is not yet implemented.')
            return cancel_exp(b, cQ, n, DE)
        elif DE.case == 'primitive':
            if parametric:
                raise NotImplementedError('Parametric RDE cancellation primitive case is not yet implemented.')
            return cancel_primitive(b, cQ, n, DE)
        else:
            raise NotImplementedError('Other Poly (P)RDE cancellation cases are not yet implemented (%s).' % DE.case)
        if parametric:
            raise NotImplementedError('Remaining cases for Poly PRDE not yet implemented.')
        raise NotImplementedError('Remaining cases for Poly RDE not yet implemented.')

def rischDE(fa, fd, ga, gd, DE):
    if False:
        while True:
            i = 10
    '\n    Solve a Risch Differential Equation: Dy + f*y == g.\n\n    Explanation\n    ===========\n\n    See the outline in the docstring of rde.py for more information\n    about the procedure used.  Either raise NonElementaryIntegralException, in\n    which case there is no solution y in the given differential field,\n    or return y in k(t) satisfying Dy + f*y == g, or raise\n    NotImplementedError, in which case, the algorithms necessary to\n    solve the given Risch Differential Equation have not yet been\n    implemented.\n    '
    (_, (fa, fd)) = weak_normalizer(fa, fd, DE)
    (a, (ba, bd), (ca, cd), hn) = normal_denom(fa, fd, ga, gd, DE)
    (A, B, C, hs) = special_denom(a, ba, bd, ca, cd, DE)
    try:
        n = bound_degree(A, B, C, DE)
    except NotImplementedError:
        n = oo
    (B, C, m, alpha, beta) = spde(A, B, C, n, DE)
    if C.is_zero:
        y = C
    else:
        y = solve_poly_rde(B, C, m, DE)
    return (alpha * y + beta, hn * hs)