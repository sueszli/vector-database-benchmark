"""This module implements tools for integrating rational functions. """
from sympy.core.function import Lambda
from sympy.core.numbers import I
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol, symbols
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.trigonometric import atan
from sympy.polys.polyroots import roots
from sympy.polys.polytools import cancel
from sympy.polys.rootoftools import RootSum
from sympy.polys import Poly, resultant, ZZ

def ratint(f, x, **flags):
    if False:
        i = 10
        return i + 15
    "\n    Performs indefinite integration of rational functions.\n\n    Explanation\n    ===========\n\n    Given a field :math:`K` and a rational function :math:`f = p/q`,\n    where :math:`p` and :math:`q` are polynomials in :math:`K[x]`,\n    returns a function :math:`g` such that :math:`f = g'`.\n\n    Examples\n    ========\n\n    >>> from sympy.integrals.rationaltools import ratint\n    >>> from sympy.abc import x\n\n    >>> ratint(36/(x**5 - 2*x**4 - 2*x**3 + 4*x**2 + x - 2), x)\n    (12*x + 6)/(x**2 - 1) + 4*log(x - 2) - 4*log(x + 1)\n\n    References\n    ==========\n\n    .. [1] M. Bronstein, Symbolic Integration I: Transcendental\n       Functions, Second Edition, Springer-Verlag, 2005, pp. 35-70\n\n    See Also\n    ========\n\n    sympy.integrals.integrals.Integral.doit\n    sympy.integrals.rationaltools.ratint_logpart\n    sympy.integrals.rationaltools.ratint_ratpart\n\n    "
    if isinstance(f, tuple):
        (p, q) = f
    else:
        (p, q) = f.as_numer_denom()
    (p, q) = (Poly(p, x, composite=False, field=True), Poly(q, x, composite=False, field=True))
    (coeff, p, q) = p.cancel(q)
    (poly, p) = p.div(q)
    result = poly.integrate(x).as_expr()
    if p.is_zero:
        return coeff * result
    (g, h) = ratint_ratpart(p, q, x)
    (P, Q) = h.as_numer_denom()
    P = Poly(P, x)
    Q = Poly(Q, x)
    (q, r) = P.div(Q)
    result += g + q.integrate(x).as_expr()
    if not r.is_zero:
        symbol = flags.get('symbol', 't')
        if not isinstance(symbol, Symbol):
            t = Dummy(symbol)
        else:
            t = symbol.as_dummy()
        L = ratint_logpart(r, Q, x, t)
        real = flags.get('real')
        if real is None:
            if isinstance(f, tuple):
                (p, q) = f
                atoms = p.atoms() | q.atoms()
            else:
                atoms = f.atoms()
            for elt in atoms - {x}:
                if not elt.is_extended_real:
                    real = False
                    break
            else:
                real = True
        eps = S.Zero
        if not real:
            for (h, q) in L:
                (_, h) = h.primitive()
                eps += RootSum(q, Lambda(t, t * log(h.as_expr())), quadratic=True)
        else:
            for (h, q) in L:
                (_, h) = h.primitive()
                R = log_to_real(h, q, x, t)
                if R is not None:
                    eps += R
                else:
                    eps += RootSum(q, Lambda(t, t * log(h.as_expr())), quadratic=True)
        result += eps
    return coeff * result

def ratint_ratpart(f, g, x):
    if False:
        while True:
            i = 10
    "\n    Horowitz-Ostrogradsky algorithm.\n\n    Explanation\n    ===========\n\n    Given a field K and polynomials f and g in K[x], such that f and g\n    are coprime and deg(f) < deg(g), returns fractions A and B in K(x),\n    such that f/g = A' + B and B has square-free denominator.\n\n    Examples\n    ========\n\n        >>> from sympy.integrals.rationaltools import ratint_ratpart\n        >>> from sympy.abc import x, y\n        >>> from sympy import Poly\n        >>> ratint_ratpart(Poly(1, x, domain='ZZ'),\n        ... Poly(x + 1, x, domain='ZZ'), x)\n        (0, 1/(x + 1))\n        >>> ratint_ratpart(Poly(1, x, domain='EX'),\n        ... Poly(x**2 + y**2, x, domain='EX'), x)\n        (0, 1/(x**2 + y**2))\n        >>> ratint_ratpart(Poly(36, x, domain='ZZ'),\n        ... Poly(x**5 - 2*x**4 - 2*x**3 + 4*x**2 + x - 2, x, domain='ZZ'), x)\n        ((12*x + 6)/(x**2 - 1), 12/(x**2 - x - 2))\n\n    See Also\n    ========\n\n    ratint, ratint_logpart\n    "
    from sympy.solvers.solvers import solve
    f = Poly(f, x)
    g = Poly(g, x)
    (u, v, _) = g.cofactors(g.diff())
    n = u.degree()
    m = v.degree()
    A_coeffs = [Dummy('a' + str(n - i)) for i in range(0, n)]
    B_coeffs = [Dummy('b' + str(m - i)) for i in range(0, m)]
    C_coeffs = A_coeffs + B_coeffs
    A = Poly(A_coeffs, x, domain=ZZ[C_coeffs])
    B = Poly(B_coeffs, x, domain=ZZ[C_coeffs])
    H = f - A.diff() * v + A * (u.diff() * v).quo(u) - B * u
    result = solve(H.coeffs(), C_coeffs)
    A = A.as_expr().subs(result)
    B = B.as_expr().subs(result)
    rat_part = cancel(A / u.as_expr(), x)
    log_part = cancel(B / v.as_expr(), x)
    return (rat_part, log_part)

def ratint_logpart(f, g, x, t=None):
    if False:
        print('Hello World!')
    "\n    Lazard-Rioboo-Trager algorithm.\n\n    Explanation\n    ===========\n\n    Given a field K and polynomials f and g in K[x], such that f and g\n    are coprime, deg(f) < deg(g) and g is square-free, returns a list\n    of tuples (s_i, q_i) of polynomials, for i = 1..n, such that s_i\n    in K[t, x] and q_i in K[t], and::\n\n                           ___    ___\n                 d  f   d  \\  `   \\  `\n                 -- - = --  )      )   a log(s_i(a, x))\n                 dx g   dx /__,   /__,\n                          i=1..n a | q_i(a) = 0\n\n    Examples\n    ========\n\n    >>> from sympy.integrals.rationaltools import ratint_logpart\n    >>> from sympy.abc import x\n    >>> from sympy import Poly\n    >>> ratint_logpart(Poly(1, x, domain='ZZ'),\n    ... Poly(x**2 + x + 1, x, domain='ZZ'), x)\n    [(Poly(x + 3*_t/2 + 1/2, x, domain='QQ[_t]'),\n    ...Poly(3*_t**2 + 1, _t, domain='ZZ'))]\n    >>> ratint_logpart(Poly(12, x, domain='ZZ'),\n    ... Poly(x**2 - x - 2, x, domain='ZZ'), x)\n    [(Poly(x - 3*_t/8 - 1/2, x, domain='QQ[_t]'),\n    ...Poly(-_t**2 + 16, _t, domain='ZZ'))]\n\n    See Also\n    ========\n\n    ratint, ratint_ratpart\n    "
    (f, g) = (Poly(f, x), Poly(g, x))
    t = t or Dummy('t')
    (a, b) = (g, f - g.diff() * Poly(t, x))
    (res, R) = resultant(a, b, includePRS=True)
    res = Poly(res, t, composite=False)
    assert res, 'BUG: resultant(%s, %s) cannot be zero' % (a, b)
    (R_map, H) = ({}, [])
    for r in R:
        R_map[r.degree()] = r

    def _include_sign(c, sqf):
        if False:
            print('Hello World!')
        if c.is_extended_real and (c < 0) == True:
            (h, k) = sqf[0]
            c_poly = c.as_poly(h.gens)
            sqf[0] = (h * c_poly, k)
    (C, res_sqf) = res.sqf_list()
    _include_sign(C, res_sqf)
    for (q, i) in res_sqf:
        (_, q) = q.primitive()
        if g.degree() == i:
            H.append((g, q))
        else:
            h = R_map[i]
            h_lc = Poly(h.LC(), t, field=True)
            (c, h_lc_sqf) = h_lc.sqf_list(all=True)
            _include_sign(c, h_lc_sqf)
            for (a, j) in h_lc_sqf:
                h = h.quo(Poly(a.gcd(q) ** j, x))
            (inv, coeffs) = (h_lc.invert(q), [S.One])
            for coeff in h.coeffs()[1:]:
                coeff = coeff.as_poly(inv.gens)
                T = (inv * coeff).rem(q)
                coeffs.append(T.as_expr())
            h = Poly(dict(list(zip(h.monoms(), coeffs))), x)
            H.append((h, q))
    return H

def log_to_atan(f, g):
    if False:
        print('Hello World!')
    "\n    Convert complex logarithms to real arctangents.\n\n    Explanation\n    ===========\n\n    Given a real field K and polynomials f and g in K[x], with g != 0,\n    returns a sum h of arctangents of polynomials in K[x], such that:\n\n                   dh   d         f + I g\n                   -- = -- I log( ------- )\n                   dx   dx        f - I g\n\n    Examples\n    ========\n\n        >>> from sympy.integrals.rationaltools import log_to_atan\n        >>> from sympy.abc import x\n        >>> from sympy import Poly, sqrt, S\n        >>> log_to_atan(Poly(x, x, domain='ZZ'), Poly(1, x, domain='ZZ'))\n        2*atan(x)\n        >>> log_to_atan(Poly(x + S(1)/2, x, domain='QQ'),\n        ... Poly(sqrt(3)/2, x, domain='EX'))\n        2*atan(2*sqrt(3)*x/3 + sqrt(3)/3)\n\n    See Also\n    ========\n\n    log_to_real\n    "
    if f.degree() < g.degree():
        (f, g) = (-g, f)
    f = f.to_field()
    g = g.to_field()
    (p, q) = f.div(g)
    if q.is_zero:
        return 2 * atan(p.as_expr())
    else:
        (s, t, h) = g.gcdex(-f)
        u = (f * s + g * t).quo(h)
        A = 2 * atan(u.as_expr())
        return A + log_to_atan(s, t)

def log_to_real(h, q, x, t):
    if False:
        return 10
    "\n    Convert complex logarithms to real functions.\n\n    Explanation\n    ===========\n\n    Given real field K and polynomials h in K[t,x] and q in K[t],\n    returns real function f such that:\n                          ___\n                  df   d  \\  `\n                  -- = --  )  a log(h(a, x))\n                  dx   dx /__,\n                         a | q(a) = 0\n\n    Examples\n    ========\n\n        >>> from sympy.integrals.rationaltools import log_to_real\n        >>> from sympy.abc import x, y\n        >>> from sympy import Poly, S\n        >>> log_to_real(Poly(x + 3*y/2 + S(1)/2, x, domain='QQ[y]'),\n        ... Poly(3*y**2 + 1, y, domain='ZZ'), x, y)\n        2*sqrt(3)*atan(2*sqrt(3)*x/3 + sqrt(3)/3)/3\n        >>> log_to_real(Poly(x**2 - 1, x, domain='ZZ'),\n        ... Poly(-2*y + 1, y, domain='ZZ'), x, y)\n        log(x**2 - 1)/2\n\n    See Also\n    ========\n\n    log_to_atan\n    "
    from sympy.simplify.radsimp import collect
    (u, v) = symbols('u,v', cls=Dummy)
    H = h.as_expr().xreplace({t: u + I * v}).expand()
    Q = q.as_expr().xreplace({t: u + I * v}).expand()
    H_map = collect(H, I, evaluate=False)
    Q_map = collect(Q, I, evaluate=False)
    (a, b) = (H_map.get(S.One, S.Zero), H_map.get(I, S.Zero))
    (c, d) = (Q_map.get(S.One, S.Zero), Q_map.get(I, S.Zero))
    R = Poly(resultant(c, d, v), u)
    R_u = roots(R, filter='R')
    if len(R_u) != R.count_roots():
        return None
    result = S.Zero
    for r_u in R_u.keys():
        C = Poly(c.xreplace({u: r_u}), v)
        if not C:
            C = Poly(d.xreplace({u: r_u}), v)
            d = S.Zero
        R_v = roots(C, filter='R')
        if len(R_v) != C.count_roots():
            return None
        R_v_paired = []
        for r_v in R_v:
            if r_v not in R_v_paired and -r_v not in R_v_paired:
                if r_v.is_negative or r_v.could_extract_minus_sign():
                    R_v_paired.append(-r_v)
                elif not r_v.is_zero:
                    R_v_paired.append(r_v)
        for r_v in R_v_paired:
            D = d.xreplace({u: r_u, v: r_v})
            if D.evalf(chop=True) != 0:
                continue
            A = Poly(a.xreplace({u: r_u, v: r_v}), x)
            B = Poly(b.xreplace({u: r_u, v: r_v}), x)
            AB = (A ** 2 + B ** 2).as_expr()
            result += r_u * log(AB) + r_v * log_to_atan(A, B)
    R_q = roots(q, filter='R')
    if len(R_q) != q.count_roots():
        return None
    for r in R_q.keys():
        result += r * log(h.as_expr().subs(t, r))
    return result