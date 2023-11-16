from __future__ import annotations
from itertools import permutations
from functools import reduce
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.mul import Mul
from sympy.core.symbol import Wild, Dummy, Symbol
from sympy.core.basic import sympify
from sympy.core.numbers import Rational, pi, I
from sympy.core.relational import Eq, Ne
from sympy.core.singleton import S
from sympy.core.sorting import ordered
from sympy.core.traversal import iterfreeargs
from sympy.functions import exp, sin, cos, tan, cot, asin, atan
from sympy.functions import log, sinh, cosh, tanh, coth, asinh
from sympy.functions import sqrt, erf, erfi, li, Ei
from sympy.functions import besselj, bessely, besseli, besselk
from sympy.functions import hankel1, hankel2, jn, yn
from sympy.functions.elementary.complexes import Abs, re, im, sign, arg
from sympy.functions.elementary.exponential import LambertW
from sympy.functions.elementary.integers import floor, ceiling
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.special.delta_functions import Heaviside, DiracDelta
from sympy.simplify.radsimp import collect
from sympy.logic.boolalg import And, Or
from sympy.utilities.iterables import uniq
from sympy.polys import quo, gcd, lcm, factor_list, cancel, PolynomialError
from sympy.polys.monomials import itermonomials
from sympy.polys.polyroots import root_factors
from sympy.polys.rings import PolyRing
from sympy.polys.solvers import solve_lin_sys
from sympy.polys.constructor import construct_domain
from sympy.integrals.integrals import integrate

def components(f, x):
    if False:
        i = 10
        return i + 15
    '\n    Returns a set of all functional components of the given expression\n    which includes symbols, function applications and compositions and\n    non-integer powers. Fractional powers are collected with\n    minimal, positive exponents.\n\n    Examples\n    ========\n\n    >>> from sympy import cos, sin\n    >>> from sympy.abc import x\n    >>> from sympy.integrals.heurisch import components\n\n    >>> components(sin(x)*cos(x)**2, x)\n    {x, sin(x), cos(x)}\n\n    See Also\n    ========\n\n    heurisch\n    '
    result = set()
    if f.has_free(x):
        if f.is_symbol and f.is_commutative:
            result.add(f)
        elif f.is_Function or f.is_Derivative:
            for g in f.args:
                result |= components(g, x)
            result.add(f)
        elif f.is_Pow:
            result |= components(f.base, x)
            if not f.exp.is_Integer:
                if f.exp.is_Rational:
                    result.add(f.base ** Rational(1, f.exp.q))
                else:
                    result |= components(f.exp, x) | {f}
        else:
            for g in f.args:
                result |= components(g, x)
    return result
_symbols_cache: dict[str, list[Dummy]] = {}

def _symbols(name, n):
    if False:
        return 10
    'get vector of symbols local to this module'
    try:
        lsyms = _symbols_cache[name]
    except KeyError:
        lsyms = []
        _symbols_cache[name] = lsyms
    while len(lsyms) < n:
        lsyms.append(Dummy('%s%i' % (name, len(lsyms))))
    return lsyms[:n]

def heurisch_wrapper(f, x, rewrite=False, hints=None, mappings=None, retries=3, degree_offset=0, unnecessary_permutations=None, _try_heurisch=None):
    if False:
        i = 10
        return i + 15
    "\n    A wrapper around the heurisch integration algorithm.\n\n    Explanation\n    ===========\n\n    This method takes the result from heurisch and checks for poles in the\n    denominator. For each of these poles, the integral is reevaluated, and\n    the final integration result is given in terms of a Piecewise.\n\n    Examples\n    ========\n\n    >>> from sympy import cos, symbols\n    >>> from sympy.integrals.heurisch import heurisch, heurisch_wrapper\n    >>> n, x = symbols('n x')\n    >>> heurisch(cos(n*x), x)\n    sin(n*x)/n\n    >>> heurisch_wrapper(cos(n*x), x)\n    Piecewise((sin(n*x)/n, Ne(n, 0)), (x, True))\n\n    See Also\n    ========\n\n    heurisch\n    "
    from sympy.solvers.solvers import solve, denoms
    f = sympify(f)
    if not f.has_free(x):
        return f * x
    res = heurisch(f, x, rewrite, hints, mappings, retries, degree_offset, unnecessary_permutations, _try_heurisch)
    if not isinstance(res, Basic):
        return res
    slns = []
    for d in ordered(denoms(res)):
        try:
            slns += solve([d], dict=True, exclude=(x,))
        except NotImplementedError:
            pass
    if not slns:
        return res
    slns = list(uniq(slns))
    slns0 = []
    for d in denoms(f):
        try:
            slns0 += solve([d], dict=True, exclude=(x,))
        except NotImplementedError:
            pass
    slns = [s for s in slns if s not in slns0]
    if not slns:
        return res
    if len(slns) > 1:
        eqs = []
        for sub_dict in slns:
            eqs.extend([Eq(key, value) for (key, value) in sub_dict.items()])
        slns = solve(eqs, dict=True, exclude=(x,)) + slns
    pairs = []
    for sub_dict in slns:
        expr = heurisch(f.subs(sub_dict), x, rewrite, hints, mappings, retries, degree_offset, unnecessary_permutations, _try_heurisch)
        cond = And(*[Eq(key, value) for (key, value) in sub_dict.items()])
        generic = Or(*[Ne(key, value) for (key, value) in sub_dict.items()])
        if expr is None:
            expr = integrate(f.subs(sub_dict), x)
        pairs.append((expr, cond))
    if len(pairs) == 1:
        pairs = [(heurisch(f, x, rewrite, hints, mappings, retries, degree_offset, unnecessary_permutations, _try_heurisch), generic), (pairs[0][0], True)]
    else:
        pairs.append((heurisch(f, x, rewrite, hints, mappings, retries, degree_offset, unnecessary_permutations, _try_heurisch), True))
    return Piecewise(*pairs)

class BesselTable:
    """
    Derivatives of Bessel functions of orders n and n-1
    in terms of each other.

    See the docstring of DiffCache.
    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.table = {}
        self.n = Dummy('n')
        self.z = Dummy('z')
        self._create_table()

    def _create_table(t):
        if False:
            while True:
                i = 10
        (table, n, z) = (t.table, t.n, t.z)
        for f in (besselj, bessely, hankel1, hankel2):
            table[f] = (f(n - 1, z) - n * f(n, z) / z, (n - 1) * f(n - 1, z) / z - f(n, z))
        f = besseli
        table[f] = (f(n - 1, z) - n * f(n, z) / z, (n - 1) * f(n - 1, z) / z + f(n, z))
        f = besselk
        table[f] = (-f(n - 1, z) - n * f(n, z) / z, (n - 1) * f(n - 1, z) / z - f(n, z))
        for f in (jn, yn):
            table[f] = (f(n - 1, z) - (n + 1) * f(n, z) / z, (n - 1) * f(n - 1, z) / z - f(n, z))

    def diffs(t, f, n, z):
        if False:
            return 10
        if f in t.table:
            (diff0, diff1) = t.table[f]
            repl = [(t.n, n), (t.z, z)]
            return (diff0.subs(repl), diff1.subs(repl))

    def has(t, f):
        if False:
            return 10
        return f in t.table
_bessel_table = None

class DiffCache:
    """
    Store for derivatives of expressions.

    Explanation
    ===========

    The standard form of the derivative of a Bessel function of order n
    contains two Bessel functions of orders n-1 and n+1, respectively.
    Such forms cannot be used in parallel Risch algorithm, because
    there is a linear recurrence relation between the three functions
    while the algorithm expects that functions and derivatives are
    represented in terms of algebraically independent transcendentals.

    The solution is to take two of the functions, e.g., those of orders
    n and n-1, and to express the derivatives in terms of the pair.
    To guarantee that the proper form is used the two derivatives are
    cached as soon as one is encountered.

    Derivatives of other functions are also cached at no extra cost.
    All derivatives are with respect to the same variable `x`.
    """

    def __init__(self, x):
        if False:
            for i in range(10):
                print('nop')
        self.cache = {}
        self.x = x
        global _bessel_table
        if not _bessel_table:
            _bessel_table = BesselTable()

    def get_diff(self, f):
        if False:
            for i in range(10):
                print('nop')
        cache = self.cache
        if f in cache:
            pass
        elif not hasattr(f, 'func') or not _bessel_table.has(f.func):
            cache[f] = cancel(f.diff(self.x))
        else:
            (n, z) = f.args
            (d0, d1) = _bessel_table.diffs(f.func, n, z)
            dz = self.get_diff(z)
            cache[f] = d0 * dz
            cache[f.func(n - 1, z)] = d1 * dz
        return cache[f]

def heurisch(f, x, rewrite=False, hints=None, mappings=None, retries=3, degree_offset=0, unnecessary_permutations=None, _try_heurisch=None):
    if False:
        print('Hello World!')
    '\n    Compute indefinite integral using heuristic Risch algorithm.\n\n    Explanation\n    ===========\n\n    This is a heuristic approach to indefinite integration in finite\n    terms using the extended heuristic (parallel) Risch algorithm, based\n    on Manuel Bronstein\'s "Poor Man\'s Integrator".\n\n    The algorithm supports various classes of functions including\n    transcendental elementary or special functions like Airy,\n    Bessel, Whittaker and Lambert.\n\n    Note that this algorithm is not a decision procedure. If it isn\'t\n    able to compute the antiderivative for a given function, then this is\n    not a proof that such a functions does not exist.  One should use\n    recursive Risch algorithm in such case.  It\'s an open question if\n    this algorithm can be made a full decision procedure.\n\n    This is an internal integrator procedure. You should use top level\n    \'integrate\' function in most cases, as this procedure needs some\n    preprocessing steps and otherwise may fail.\n\n    Specification\n    =============\n\n     heurisch(f, x, rewrite=False, hints=None)\n\n       where\n         f : expression\n         x : symbol\n\n         rewrite -> force rewrite \'f\' in terms of \'tan\' and \'tanh\'\n         hints   -> a list of functions that may appear in anti-derivate\n\n          - hints = None          --> no suggestions at all\n          - hints = [ ]           --> try to figure out\n          - hints = [f1, ..., fn] --> we know better\n\n    Examples\n    ========\n\n    >>> from sympy import tan\n    >>> from sympy.integrals.heurisch import heurisch\n    >>> from sympy.abc import x, y\n\n    >>> heurisch(y*tan(x), x)\n    y*log(tan(x)**2 + 1)/2\n\n    See Manuel Bronstein\'s "Poor Man\'s Integrator":\n\n    References\n    ==========\n\n    .. [1] https://www-sop.inria.fr/cafe/Manuel.Bronstein/pmint/index.html\n\n    For more information on the implemented algorithm refer to:\n\n    .. [2] K. Geddes, L. Stefanus, On the Risch-Norman Integration\n       Method and its Implementation in Maple, Proceedings of\n       ISSAC\'89, ACM Press, 212-217.\n\n    .. [3] J. H. Davenport, On the Parallel Risch Algorithm (I),\n       Proceedings of EUROCAM\'82, LNCS 144, Springer, 144-157.\n\n    .. [4] J. H. Davenport, On the Parallel Risch Algorithm (III):\n       Use of Tangents, SIGSAM Bulletin 16 (1982), 3-6.\n\n    .. [5] J. H. Davenport, B. M. Trager, On the Parallel Risch\n       Algorithm (II), ACM Transactions on Mathematical\n       Software 11 (1985), 356-362.\n\n    See Also\n    ========\n\n    sympy.integrals.integrals.Integral.doit\n    sympy.integrals.integrals.Integral\n    sympy.integrals.heurisch.components\n    '
    f = sympify(f)
    if _try_heurisch is not True:
        if f.has(Abs, re, im, sign, Heaviside, DiracDelta, floor, ceiling, arg):
            return
    if not f.has_free(x):
        return f * x
    if not f.is_Add:
        (indep, f) = f.as_independent(x)
    else:
        indep = S.One
    rewritables = {(sin, cos, cot): tan, (sinh, cosh, coth): tanh}
    if rewrite:
        for (candidates, rule) in rewritables.items():
            f = f.rewrite(candidates, rule)
    else:
        for candidates in rewritables.keys():
            if f.has(*candidates):
                break
        else:
            rewrite = True
    terms = components(f, x)
    dcache = DiffCache(x)
    if hints is not None:
        if not hints:
            a = Wild('a', exclude=[x])
            b = Wild('b', exclude=[x])
            c = Wild('c', exclude=[x])
            for g in set(terms):
                if g.is_Function:
                    if isinstance(g, li):
                        M = g.args[0].match(a * x ** b)
                        if M is not None:
                            terms.add(x * (li(M[a] * x ** M[b]) - (M[a] * x ** M[b]) ** (-1 / M[b]) * Ei((M[b] + 1) * log(M[a] * x ** M[b]) / M[b])))
                    elif isinstance(g, exp):
                        M = g.args[0].match(a * x ** 2)
                        if M is not None:
                            if M[a].is_positive:
                                terms.add(erfi(sqrt(M[a]) * x))
                            else:
                                terms.add(erf(sqrt(-M[a]) * x))
                        M = g.args[0].match(a * x ** 2 + b * x + c)
                        if M is not None:
                            if M[a].is_positive:
                                terms.add(sqrt(pi / 4 * -M[a]) * exp(M[c] - M[b] ** 2 / (4 * M[a])) * erfi(sqrt(M[a]) * x + M[b] / (2 * sqrt(M[a]))))
                            elif M[a].is_negative:
                                terms.add(sqrt(pi / 4 * -M[a]) * exp(M[c] - M[b] ** 2 / (4 * M[a])) * erf(sqrt(-M[a]) * x - M[b] / (2 * sqrt(-M[a]))))
                        M = g.args[0].match(a * log(x) ** 2)
                        if M is not None:
                            if M[a].is_positive:
                                terms.add(erfi(sqrt(M[a]) * log(x) + 1 / (2 * sqrt(M[a]))))
                            if M[a].is_negative:
                                terms.add(erf(sqrt(-M[a]) * log(x) - 1 / (2 * sqrt(-M[a]))))
                elif g.is_Pow:
                    if g.exp.is_Rational and g.exp.q == 2:
                        M = g.base.match(a * x ** 2 + b)
                        if M is not None and M[b].is_positive:
                            if M[a].is_positive:
                                terms.add(asinh(sqrt(M[a] / M[b]) * x))
                            elif M[a].is_negative:
                                terms.add(asin(sqrt(-M[a] / M[b]) * x))
                        M = g.base.match(a * x ** 2 - b)
                        if M is not None and M[b].is_positive:
                            if M[a].is_positive:
                                dF = 1 / sqrt(M[a] * x ** 2 - M[b])
                                F = log(2 * sqrt(M[a]) * sqrt(M[a] * x ** 2 - M[b]) + 2 * M[a] * x) / sqrt(M[a])
                                dcache.cache[F] = dF
                                terms.add(F)
                            elif M[a].is_negative:
                                terms.add(-M[b] / 2 * sqrt(-M[a]) * atan(sqrt(-M[a]) * x / sqrt(M[a] * x ** 2 - M[b])))
        else:
            terms |= set(hints)
    for g in set(terms):
        terms |= components(dcache.get_diff(g), x)
    V = _symbols('x', len(terms))
    mapping = list(reversed(list(zip(*ordered([(a[0].as_independent(x)[1], a) for a in zip(terms, V)])))[1]))
    rev_mapping = {v: k for (k, v) in mapping}
    if mappings is None:
        assert mapping[-1][0] == x
        unnecessary_permutations = [mapping.pop(-1)]
        mappings = permutations(mapping)
    else:
        unnecessary_permutations = unnecessary_permutations or []

    def _substitute(expr):
        if False:
            i = 10
            return i + 15
        return expr.subs(mapping)
    for mapping in mappings:
        mapping = list(mapping)
        mapping = mapping + unnecessary_permutations
        diffs = [_substitute(dcache.get_diff(g)) for g in terms]
        denoms = [g.as_numer_denom()[1] for g in diffs]
        if all((h.is_polynomial(*V) for h in denoms)) and _substitute(f).is_rational_function(*V):
            denom = reduce(lambda p, q: lcm(p, q, *V), denoms)
            break
    else:
        if not rewrite:
            result = heurisch(f, x, rewrite=True, hints=hints, unnecessary_permutations=unnecessary_permutations)
            if result is not None:
                return indep * result
        return None
    numers = [cancel(denom * g) for g in diffs]

    def _derivation(h):
        if False:
            while True:
                i = 10
        return Add(*[d * h.diff(v) for (d, v) in zip(numers, V)])

    def _deflation(p):
        if False:
            return 10
        for y in V:
            if not p.has(y):
                continue
            if _derivation(p) is not S.Zero:
                (c, q) = p.as_poly(y).primitive()
                return _deflation(c) * gcd(q, q.diff(y)).as_expr()
        return p

    def _splitter(p):
        if False:
            while True:
                i = 10
        for y in V:
            if not p.has(y):
                continue
            if _derivation(y) is not S.Zero:
                (c, q) = p.as_poly(y).primitive()
                q = q.as_expr()
                h = gcd(q, _derivation(q), y)
                s = quo(h, gcd(q, q.diff(y), y), y)
                c_split = _splitter(c)
                if s.as_poly(y).degree() == 0:
                    return (c_split[0], q * c_split[1])
                q_split = _splitter(cancel(q / s))
                return (c_split[0] * q_split[0] * s, c_split[1] * q_split[1])
        return (S.One, p)
    special = {}
    for term in terms:
        if term.is_Function:
            if isinstance(term, tan):
                special[1 + _substitute(term) ** 2] = False
            elif isinstance(term, tanh):
                special[1 + _substitute(term)] = False
                special[1 - _substitute(term)] = False
            elif isinstance(term, LambertW):
                special[_substitute(term)] = True
    F = _substitute(f)
    (P, Q) = F.as_numer_denom()
    u_split = _splitter(denom)
    v_split = _splitter(Q)
    polys = set(list(v_split) + [u_split[0]] + list(special.keys()))
    s = u_split[0] * Mul(*[k for (k, v) in special.items() if v])
    polified = [p.as_poly(*V) for p in [s, P, Q]]
    if None in polified:
        return None
    (a, b, c) = [p.total_degree() for p in polified]
    poly_denom = (s * v_split[0] * _deflation(v_split[1])).as_expr()

    def _exponent(g):
        if False:
            return 10
        if g.is_Pow:
            if g.exp.is_Rational and g.exp.q != 1:
                if g.exp.p > 0:
                    return g.exp.p + g.exp.q - 1
                else:
                    return abs(g.exp.p + g.exp.q)
            else:
                return 1
        elif not g.is_Atom and g.args:
            return max([_exponent(h) for h in g.args])
        else:
            return 1
    (A, B) = (_exponent(f), a + max(b, c))
    if A > 1 and B > 1:
        monoms = tuple(ordered(itermonomials(V, A + B - 1 + degree_offset)))
    else:
        monoms = tuple(ordered(itermonomials(V, A + B + degree_offset)))
    poly_coeffs = _symbols('A', len(monoms))
    poly_part = Add(*[poly_coeffs[i] * monomial for (i, monomial) in enumerate(monoms)])
    reducibles = set()
    for poly in ordered(polys):
        (coeff, factors) = factor_list(poly, *V)
        reducibles.add(coeff)
        for (fact, mul) in factors:
            reducibles.add(fact)

    def _integrate(field=None):
        if False:
            i = 10
            return i + 15
        atans = set()
        pairs = set()
        if field == 'Q':
            irreducibles = set(reducibles)
        else:
            setV = set(V)
            irreducibles = set()
            for poly in ordered(reducibles):
                zV = setV & set(iterfreeargs(poly))
                for z in ordered(zV):
                    s = set(root_factors(poly, z, filter=field))
                    irreducibles |= s
                    break
        (log_part, atan_part) = ([], [])
        for poly in ordered(irreducibles):
            m = collect(poly, I, evaluate=False)
            y = m.get(I, S.Zero)
            if y:
                x = m.get(S.One, S.Zero)
                if x.has(I) or y.has(I):
                    continue
                pairs.add((x, y))
                irreducibles.remove(poly)
        while pairs:
            (x, y) = pairs.pop()
            if (x, -y) in pairs:
                pairs.remove((x, -y))
                if y.could_extract_minus_sign():
                    y = -y
                irreducibles.add(x * x + y * y)
                atans.add(atan(x / y))
            else:
                irreducibles.add(x + I * y)
        B = _symbols('B', len(irreducibles))
        C = _symbols('C', len(atans))
        for (poly, b) in reversed(list(zip(ordered(irreducibles), B))):
            if poly.has(*V):
                poly_coeffs.append(b)
                log_part.append(b * log(poly))
        for (poly, c) in reversed(list(zip(ordered(atans), C))):
            if poly.has(*V):
                poly_coeffs.append(c)
                atan_part.append(c * poly)
        candidate = poly_part / poly_denom + Add(*log_part) + Add(*atan_part)
        h = F - _derivation(candidate) / denom
        raw_numer = h.as_numer_denom()[0]
        syms = set(poly_coeffs) | set(V)
        non_syms = set()

        def find_non_syms(expr):
            if False:
                for i in range(10):
                    print('nop')
            if expr.is_Integer or expr.is_Rational:
                pass
            elif expr in syms:
                pass
            elif not expr.has_free(*syms):
                non_syms.add(expr)
            elif expr.is_Add or expr.is_Mul or expr.is_Pow:
                list(map(find_non_syms, expr.args))
            else:
                raise PolynomialError
        try:
            find_non_syms(raw_numer)
        except PolynomialError:
            return None
        else:
            (ground, _) = construct_domain(non_syms, field=True)
        coeff_ring = PolyRing(poly_coeffs, ground)
        ring = PolyRing(V, coeff_ring)
        try:
            numer = ring.from_expr(raw_numer)
        except ValueError:
            raise PolynomialError
        solution = solve_lin_sys(numer.coeffs(), coeff_ring, _raw=False)
        if solution is None:
            return None
        else:
            return candidate.xreplace(solution).xreplace(dict(zip(poly_coeffs, [S.Zero] * len(poly_coeffs))))
    if all((isinstance(_, Symbol) for _ in V)):
        more_free = F.free_symbols - set(V)
    else:
        Fd = F.as_dummy()
        more_free = Fd.xreplace(dict(zip(V, (Dummy() for _ in V)))).free_symbols & Fd.free_symbols
    if not more_free:
        solution = _integrate('Q')
        if solution is None:
            solution = _integrate()
    else:
        solution = _integrate()
    if solution is not None:
        antideriv = solution.subs(rev_mapping)
        antideriv = cancel(antideriv).expand()
        if antideriv.is_Add:
            antideriv = antideriv.as_independent(x)[1]
        return indep * antideriv
    else:
        if retries >= 0:
            result = heurisch(f, x, mappings=mappings, rewrite=rewrite, hints=hints, retries=retries - 1, unnecessary_permutations=unnecessary_permutations)
            if result is not None:
                return indep * result
        return None