"""Formal Power Series"""
from collections import defaultdict
from sympy.core.numbers import nan, oo, zoo
from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.function import Derivative, Function, expand
from sympy.core.mul import Mul
from sympy.core.numbers import Rational
from sympy.core.relational import Eq
from sympy.sets.sets import Interval
from sympy.core.singleton import S
from sympy.core.symbol import Wild, Dummy, symbols, Symbol
from sympy.core.sympify import sympify
from sympy.discrete.convolutions import convolution
from sympy.functions.combinatorial.factorials import binomial, factorial, rf
from sympy.functions.combinatorial.numbers import bell
from sympy.functions.elementary.integers import floor, frac, ceiling
from sympy.functions.elementary.miscellaneous import Min, Max
from sympy.functions.elementary.piecewise import Piecewise
from sympy.series.limits import Limit
from sympy.series.order import Order
from sympy.series.sequences import sequence
from sympy.series.series_class import SeriesBase
from sympy.utilities.iterables import iterable

def rational_algorithm(f, x, k, order=4, full=False):
    if False:
        i = 10
        return i + 15
    "\n    Rational algorithm for computing\n    formula of coefficients of Formal Power Series\n    of a function.\n\n    Explanation\n    ===========\n\n    Applicable when f(x) or some derivative of f(x)\n    is a rational function in x.\n\n    :func:`rational_algorithm` uses :func:`~.apart` function for partial fraction\n    decomposition. :func:`~.apart` by default uses 'undetermined coefficients\n    method'. By setting ``full=True``, 'Bronstein's algorithm' can be used\n    instead.\n\n    Looks for derivative of a function up to 4'th order (by default).\n    This can be overridden using order option.\n\n    Parameters\n    ==========\n\n    x : Symbol\n    order : int, optional\n        Order of the derivative of ``f``, Default is 4.\n    full : bool\n\n    Returns\n    =======\n\n    formula : Expr\n    ind : Expr\n        Independent terms.\n    order : int\n    full : bool\n\n    Examples\n    ========\n\n    >>> from sympy import log, atan\n    >>> from sympy.series.formal import rational_algorithm as ra\n    >>> from sympy.abc import x, k\n\n    >>> ra(1 / (1 - x), x, k)\n    (1, 0, 0)\n    >>> ra(log(1 + x), x, k)\n    (-1/((-1)**k*k), 0, 1)\n\n    >>> ra(atan(x), x, k, full=True)\n    ((-I/(2*(-I)**k) + I/(2*I**k))/k, 0, 1)\n\n    Notes\n    =====\n\n    By setting ``full=True``, range of admissible functions to be solved using\n    ``rational_algorithm`` can be increased. This option should be used\n    carefully as it can significantly slow down the computation as ``doit`` is\n    performed on the :class:`~.RootSum` object returned by the :func:`~.apart`\n    function. Use ``full=False`` whenever possible.\n\n    See Also\n    ========\n\n    sympy.polys.partfrac.apart\n\n    References\n    ==========\n\n    .. [1] Formal Power Series - Dominik Gruntz, Wolfram Koepf\n    .. [2] Power Series in Computer Algebra - Wolfram Koepf\n\n    "
    from sympy.polys import RootSum, apart
    from sympy.integrals import integrate
    diff = f
    ds = []
    for i in range(order + 1):
        if i:
            diff = diff.diff(x)
        if diff.is_rational_function(x):
            (coeff, sep) = (S.Zero, S.Zero)
            terms = apart(diff, x, full=full)
            if terms.has(RootSum):
                terms = terms.doit()
            for t in Add.make_args(terms):
                (num, den) = t.as_numer_denom()
                if not den.has(x):
                    sep += t
                else:
                    if isinstance(den, Mul):
                        ind = den.as_independent(x)
                        den = ind[1]
                        num /= ind[0]
                    (den, j) = den.as_base_exp()
                    (a, xterm) = den.as_coeff_add(x)
                    if not a:
                        sep += t
                        continue
                    xc = xterm[0].coeff(x)
                    a /= -xc
                    num /= xc ** j
                    ak = (-1) ** j * num * binomial(j + k - 1, k).rewrite(factorial) / a ** (j + k)
                    coeff += ak
            if coeff.is_zero:
                return None
            if coeff.has(x) or coeff.has(zoo) or coeff.has(oo) or coeff.has(nan):
                return None
            for j in range(i):
                coeff = coeff / (k + j + 1)
                sep = integrate(sep, x)
                sep += (ds.pop() - sep).limit(x, 0)
            return (coeff.subs(k, k - i), sep, i)
        else:
            ds.append(diff)
    return None

def rational_independent(terms, x):
    if False:
        i = 10
        return i + 15
    '\n    Returns a list of all the rationally independent terms.\n\n    Examples\n    ========\n\n    >>> from sympy import sin, cos\n    >>> from sympy.series.formal import rational_independent\n    >>> from sympy.abc import x\n\n    >>> rational_independent([cos(x), sin(x)], x)\n    [cos(x), sin(x)]\n    >>> rational_independent([x**2, sin(x), x*sin(x), x**3], x)\n    [x**3 + x**2, x*sin(x) + sin(x)]\n    '
    if not terms:
        return []
    ind = terms[0:1]
    for t in terms[1:]:
        n = t.as_independent(x)[1]
        for (i, term) in enumerate(ind):
            d = term.as_independent(x)[1]
            q = (n / d).cancel()
            if q.is_rational_function(x):
                ind[i] += t
                break
        else:
            ind.append(t)
    return ind

def simpleDE(f, x, g, order=4):
    if False:
        i = 10
        return i + 15
    "\n    Generates simple DE.\n\n    Explanation\n    ===========\n\n    DE is of the form\n\n    .. math::\n        f^k(x) + \\sum\\limits_{j=0}^{k-1} A_j f^j(x) = 0\n\n    where :math:`A_j` should be rational function in x.\n\n    Generates DE's upto order 4 (default). DE's can also have free parameters.\n\n    By increasing order, higher order DE's can be found.\n\n    Yields a tuple of (DE, order).\n    "
    from sympy.solvers.solveset import linsolve
    a = symbols('a:%d' % order)

    def _makeDE(k):
        if False:
            print('Hello World!')
        eq = f.diff(x, k) + Add(*[a[i] * f.diff(x, i) for i in range(0, k)])
        DE = g(x).diff(x, k) + Add(*[a[i] * g(x).diff(x, i) for i in range(0, k)])
        return (eq, DE)
    found = False
    for k in range(1, order + 1):
        (eq, DE) = _makeDE(k)
        eq = eq.expand()
        terms = eq.as_ordered_terms()
        ind = rational_independent(terms, x)
        if found or len(ind) == k:
            sol = dict(zip(a, (i for s in linsolve(ind, a[:k]) for i in s)))
            if sol:
                found = True
                DE = DE.subs(sol)
            DE = DE.as_numer_denom()[0]
            DE = DE.factor().as_coeff_mul(Derivative)[1][0]
            yield (DE.collect(Derivative(g(x))), k)

def exp_re(DE, r, k):
    if False:
        print('Hello World!')
    "Converts a DE with constant coefficients (explike) into a RE.\n\n    Explanation\n    ===========\n\n    Performs the substitution:\n\n    .. math::\n        f^j(x) \\to r(k + j)\n\n    Normalises the terms so that lowest order of a term is always r(k).\n\n    Examples\n    ========\n\n    >>> from sympy import Function, Derivative\n    >>> from sympy.series.formal import exp_re\n    >>> from sympy.abc import x, k\n    >>> f, r = Function('f'), Function('r')\n\n    >>> exp_re(-f(x) + Derivative(f(x)), r, k)\n    -r(k) + r(k + 1)\n    >>> exp_re(Derivative(f(x), x) + Derivative(f(x), (x, 2)), r, k)\n    r(k) + r(k + 1)\n\n    See Also\n    ========\n\n    sympy.series.formal.hyper_re\n    "
    RE = S.Zero
    g = DE.atoms(Function).pop()
    mini = None
    for t in Add.make_args(DE):
        (coeff, d) = t.as_independent(g)
        if isinstance(d, Derivative):
            j = d.derivative_count
        else:
            j = 0
        if mini is None or j < mini:
            mini = j
        RE += coeff * r(k + j)
    if mini:
        RE = RE.subs(k, k - mini)
    return RE

def hyper_re(DE, r, k):
    if False:
        print('Hello World!')
    "\n    Converts a DE into a RE.\n\n    Explanation\n    ===========\n\n    Performs the substitution:\n\n    .. math::\n        x^l f^j(x) \\to (k + 1 - l)_j . a_{k + j - l}\n\n    Normalises the terms so that lowest order of a term is always r(k).\n\n    Examples\n    ========\n\n    >>> from sympy import Function, Derivative\n    >>> from sympy.series.formal import hyper_re\n    >>> from sympy.abc import x, k\n    >>> f, r = Function('f'), Function('r')\n\n    >>> hyper_re(-f(x) + Derivative(f(x)), r, k)\n    (k + 1)*r(k + 1) - r(k)\n    >>> hyper_re(-x*f(x) + Derivative(f(x), (x, 2)), r, k)\n    (k + 2)*(k + 3)*r(k + 3) - r(k)\n\n    See Also\n    ========\n\n    sympy.series.formal.exp_re\n    "
    RE = S.Zero
    g = DE.atoms(Function).pop()
    x = g.atoms(Symbol).pop()
    mini = None
    for t in Add.make_args(DE.expand()):
        (coeff, d) = t.as_independent(g)
        (c, v) = coeff.as_independent(x)
        l = v.as_coeff_exponent(x)[1]
        if isinstance(d, Derivative):
            j = d.derivative_count
        else:
            j = 0
        RE += c * rf(k + 1 - l, j) * r(k + j - l)
        if mini is None or j - l < mini:
            mini = j - l
    RE = RE.subs(k, k - mini)
    m = Wild('m')
    return RE.collect(r(k + m))

def _transformation_a(f, x, P, Q, k, m, shift):
    if False:
        while True:
            i = 10
    f *= x ** (-shift)
    P = P.subs(k, k + shift)
    Q = Q.subs(k, k + shift)
    return (f, P, Q, m)

def _transformation_c(f, x, P, Q, k, m, scale):
    if False:
        while True:
            i = 10
    f = f.subs(x, x ** scale)
    P = P.subs(k, k / scale)
    Q = Q.subs(k, k / scale)
    m *= scale
    return (f, P, Q, m)

def _transformation_e(f, x, P, Q, k, m):
    if False:
        i = 10
        return i + 15
    f = f.diff(x)
    P = P.subs(k, k + 1) * (k + m + 1)
    Q = Q.subs(k, k + 1) * (k + 1)
    return (f, P, Q, m)

def _apply_shift(sol, shift):
    if False:
        print('Hello World!')
    return [(res, cond + shift) for (res, cond) in sol]

def _apply_scale(sol, scale):
    if False:
        print('Hello World!')
    return [(res, cond / scale) for (res, cond) in sol]

def _apply_integrate(sol, x, k):
    if False:
        while True:
            i = 10
    return [(res / ((cond + 1) * cond.as_coeff_Add()[1].coeff(k)), cond + 1) for (res, cond) in sol]

def _compute_formula(f, x, P, Q, k, m, k_max):
    if False:
        return 10
    'Computes the formula for f.'
    from sympy.polys import roots
    sol = []
    for i in range(k_max + 1, k_max + m + 1):
        if (i < 0) == True:
            continue
        r = f.diff(x, i).limit(x, 0) / factorial(i)
        if r.is_zero:
            continue
        kterm = m * k + i
        res = r
        p = P.subs(k, kterm)
        q = Q.subs(k, kterm)
        c1 = p.subs(k, 1 / k).leadterm(k)[0]
        c2 = q.subs(k, 1 / k).leadterm(k)[0]
        res *= (-c1 / c2) ** k
        res *= Mul(*[rf(-r, k) ** mul for (r, mul) in roots(p, k).items()])
        res /= Mul(*[rf(-r, k) ** mul for (r, mul) in roots(q, k).items()])
        sol.append((res, kterm))
    return sol

def _rsolve_hypergeometric(f, x, P, Q, k, m):
    if False:
        return 10
    '\n    Recursive wrapper to rsolve_hypergeometric.\n\n    Explanation\n    ===========\n\n    Returns a Tuple of (formula, series independent terms,\n    maximum power of x in independent terms) if successful\n    otherwise ``None``.\n\n    See :func:`rsolve_hypergeometric` for details.\n    '
    from sympy.polys import lcm, roots
    from sympy.integrals import integrate
    (proots, qroots) = (roots(P, k), roots(Q, k))
    all_roots = dict(proots)
    all_roots.update(qroots)
    scale = lcm([r.as_numer_denom()[1] for (r, t) in all_roots.items() if r.is_rational])
    (f, P, Q, m) = _transformation_c(f, x, P, Q, k, m, scale)
    qroots = roots(Q, k)
    if qroots:
        k_min = Min(*qroots.keys())
    else:
        k_min = S.Zero
    shift = k_min + m
    (f, P, Q, m) = _transformation_a(f, x, P, Q, k, m, shift)
    l = (x * f).limit(x, 0)
    if not isinstance(l, Limit) and l != 0:
        return None
    qroots = roots(Q, k)
    if qroots:
        k_max = Max(*qroots.keys())
    else:
        k_max = S.Zero
    (ind, mp) = (S.Zero, -oo)
    for i in range(k_max + m + 1):
        r = f.diff(x, i).limit(x, 0) / factorial(i)
        if r.is_finite is False:
            old_f = f
            (f, P, Q, m) = _transformation_a(f, x, P, Q, k, m, i)
            (f, P, Q, m) = _transformation_e(f, x, P, Q, k, m)
            (sol, ind, mp) = _rsolve_hypergeometric(f, x, P, Q, k, m)
            sol = _apply_integrate(sol, x, k)
            sol = _apply_shift(sol, i)
            ind = integrate(ind, x)
            ind += (old_f - ind).limit(x, 0)
            mp += 1
            return (sol, ind, mp)
        elif r:
            ind += r * x ** (i + shift)
            pow_x = Rational(i + shift, scale)
            if pow_x > mp:
                mp = pow_x
    ind = ind.subs(x, x ** (1 / scale))
    sol = _compute_formula(f, x, P, Q, k, m, k_max)
    sol = _apply_shift(sol, shift)
    sol = _apply_scale(sol, scale)
    return (sol, ind, mp)

def rsolve_hypergeometric(f, x, P, Q, k, m):
    if False:
        while True:
            i = 10
    "\n    Solves RE of hypergeometric type.\n\n    Explanation\n    ===========\n\n    Attempts to solve RE of the form\n\n    Q(k)*a(k + m) - P(k)*a(k)\n\n    Transformations that preserve Hypergeometric type:\n\n        a. x**n*f(x): b(k + m) = R(k - n)*b(k)\n        b. f(A*x): b(k + m) = A**m*R(k)*b(k)\n        c. f(x**n): b(k + n*m) = R(k/n)*b(k)\n        d. f(x**(1/m)): b(k + 1) = R(k*m)*b(k)\n        e. f'(x): b(k + m) = ((k + m + 1)/(k + 1))*R(k + 1)*b(k)\n\n    Some of these transformations have been used to solve the RE.\n\n    Returns\n    =======\n\n    formula : Expr\n    ind : Expr\n        Independent terms.\n    order : int\n\n    Examples\n    ========\n\n    >>> from sympy import exp, ln, S\n    >>> from sympy.series.formal import rsolve_hypergeometric as rh\n    >>> from sympy.abc import x, k\n\n    >>> rh(exp(x), x, -S.One, (k + 1), k, 1)\n    (Piecewise((1/factorial(k), Eq(Mod(k, 1), 0)), (0, True)), 1, 1)\n\n    >>> rh(ln(1 + x), x, k**2, k*(k + 1), k, 1)\n    (Piecewise(((-1)**(k - 1)*factorial(k - 1)/RisingFactorial(2, k - 1),\n     Eq(Mod(k, 1), 0)), (0, True)), x, 2)\n\n    References\n    ==========\n\n    .. [1] Formal Power Series - Dominik Gruntz, Wolfram Koepf\n    .. [2] Power Series in Computer Algebra - Wolfram Koepf\n    "
    result = _rsolve_hypergeometric(f, x, P, Q, k, m)
    if result is None:
        return None
    (sol_list, ind, mp) = result
    sol_dict = defaultdict(lambda : S.Zero)
    for (res, cond) in sol_list:
        (j, mk) = cond.as_coeff_Add()
        c = mk.coeff(k)
        if j.is_integer is False:
            res *= x ** frac(j)
            j = floor(j)
        res = res.subs(k, (k - j) / c)
        cond = Eq(k % c, j % c)
        sol_dict[cond] += res
    sol = []
    for (cond, res) in sol_dict.items():
        sol.append((res, cond))
    sol.append((S.Zero, True))
    sol = Piecewise(*sol)
    if mp is -oo:
        s = S.Zero
    elif mp.is_integer is False:
        s = ceiling(mp)
    else:
        s = mp + 1
    if s < 0:
        ind += sum(sequence(sol * x ** k, (k, s, -1)))
        s = S.Zero
    return (sol, ind, s)

def _solve_hyper_RE(f, x, RE, g, k):
    if False:
        i = 10
        return i + 15
    'See docstring of :func:`rsolve_hypergeometric` for details.'
    terms = Add.make_args(RE)
    if len(terms) == 2:
        gs = list(RE.atoms(Function))
        (P, Q) = map(RE.coeff, gs)
        m = gs[1].args[0] - gs[0].args[0]
        if m < 0:
            (P, Q) = (Q, P)
            m = abs(m)
        return rsolve_hypergeometric(f, x, P, Q, k, m)

def _solve_explike_DE(f, x, DE, g, k):
    if False:
        print('Hello World!')
    'Solves DE with constant coefficients.'
    from sympy.solvers import rsolve
    for t in Add.make_args(DE):
        (coeff, d) = t.as_independent(g)
        if coeff.free_symbols:
            return
    RE = exp_re(DE, g, k)
    init = {}
    for i in range(len(Add.make_args(RE))):
        if i:
            f = f.diff(x)
        init[g(k).subs(k, i)] = f.limit(x, 0)
    sol = rsolve(RE, g(k), init)
    if sol:
        return (sol / factorial(k), S.Zero, S.Zero)

def _solve_simple(f, x, DE, g, k):
    if False:
        print('Hello World!')
    'Converts DE into RE and solves using :func:`rsolve`.'
    from sympy.solvers import rsolve
    RE = hyper_re(DE, g, k)
    init = {}
    for i in range(len(Add.make_args(RE))):
        if i:
            f = f.diff(x)
        init[g(k).subs(k, i)] = f.limit(x, 0) / factorial(i)
    sol = rsolve(RE, g(k), init)
    if sol:
        return (sol, S.Zero, S.Zero)

def _transform_explike_DE(DE, g, x, order, syms):
    if False:
        print('Hello World!')
    'Converts DE with free parameters into DE with constant coefficients.'
    from sympy.solvers.solveset import linsolve
    eq = []
    highest_coeff = DE.coeff(Derivative(g(x), x, order))
    for i in range(order):
        coeff = DE.coeff(Derivative(g(x), x, i))
        coeff = (coeff / highest_coeff).expand().collect(x)
        for t in Add.make_args(coeff):
            eq.append(t)
    temp = []
    for e in eq:
        if e.has(x):
            break
        elif e.has(Symbol):
            temp.append(e)
    else:
        eq = temp
    if eq:
        sol = dict(zip(syms, (i for s in linsolve(eq, list(syms)) for i in s)))
        if sol:
            DE = DE.subs(sol)
            DE = DE.factor().as_coeff_mul(Derivative)[1][0]
            DE = DE.collect(Derivative(g(x)))
    return DE

def _transform_DE_RE(DE, g, k, order, syms):
    if False:
        print('Hello World!')
    'Converts DE with free parameters into RE of hypergeometric type.'
    from sympy.solvers.solveset import linsolve
    RE = hyper_re(DE, g, k)
    eq = []
    for i in range(1, order):
        coeff = RE.coeff(g(k + i))
        eq.append(coeff)
    sol = dict(zip(syms, (i for s in linsolve(eq, list(syms)) for i in s)))
    if sol:
        m = Wild('m')
        RE = RE.subs(sol)
        RE = RE.factor().as_numer_denom()[0].collect(g(k + m))
        RE = RE.as_coeff_mul(g)[1][0]
        for i in range(order):
            if RE.coeff(g(k + i)) and i:
                RE = RE.subs(k, k - i)
                break
    return RE

def solve_de(f, x, DE, order, g, k):
    if False:
        return 10
    "\n    Solves the DE.\n\n    Explanation\n    ===========\n\n    Tries to solve DE by either converting into a RE containing two terms or\n    converting into a DE having constant coefficients.\n\n    Returns\n    =======\n\n    formula : Expr\n    ind : Expr\n        Independent terms.\n    order : int\n\n    Examples\n    ========\n\n    >>> from sympy import Derivative as D, Function\n    >>> from sympy import exp, ln\n    >>> from sympy.series.formal import solve_de\n    >>> from sympy.abc import x, k\n    >>> f = Function('f')\n\n    >>> solve_de(exp(x), x, D(f(x), x) - f(x), 1, f, k)\n    (Piecewise((1/factorial(k), Eq(Mod(k, 1), 0)), (0, True)), 1, 1)\n\n    >>> solve_de(ln(1 + x), x, (x + 1)*D(f(x), x, 2) + D(f(x)), 2, f, k)\n    (Piecewise(((-1)**(k - 1)*factorial(k - 1)/RisingFactorial(2, k - 1),\n     Eq(Mod(k, 1), 0)), (0, True)), x, 2)\n    "
    sol = None
    syms = DE.free_symbols.difference({g, x})
    if syms:
        RE = _transform_DE_RE(DE, g, k, order, syms)
    else:
        RE = hyper_re(DE, g, k)
    if not RE.free_symbols.difference({k}):
        sol = _solve_hyper_RE(f, x, RE, g, k)
    if sol:
        return sol
    if syms:
        DE = _transform_explike_DE(DE, g, x, order, syms)
    if not DE.free_symbols.difference({x}):
        sol = _solve_explike_DE(f, x, DE, g, k)
    if sol:
        return sol

def hyper_algorithm(f, x, k, order=4):
    if False:
        while True:
            i = 10
    '\n    Hypergeometric algorithm for computing Formal Power Series.\n\n    Explanation\n    ===========\n\n    Steps:\n        * Generates DE\n        * Convert the DE into RE\n        * Solves the RE\n\n    Examples\n    ========\n\n    >>> from sympy import exp, ln\n    >>> from sympy.series.formal import hyper_algorithm\n\n    >>> from sympy.abc import x, k\n\n    >>> hyper_algorithm(exp(x), x, k)\n    (Piecewise((1/factorial(k), Eq(Mod(k, 1), 0)), (0, True)), 1, 1)\n\n    >>> hyper_algorithm(ln(1 + x), x, k)\n    (Piecewise(((-1)**(k - 1)*factorial(k - 1)/RisingFactorial(2, k - 1),\n     Eq(Mod(k, 1), 0)), (0, True)), x, 2)\n\n    See Also\n    ========\n\n    sympy.series.formal.simpleDE\n    sympy.series.formal.solve_de\n    '
    g = Function('g')
    des = []
    sol = None
    for (DE, i) in simpleDE(f, x, g, order):
        if DE is not None:
            sol = solve_de(f, x, DE, i, g, k)
        if sol:
            return sol
        if not DE.free_symbols.difference({x}):
            des.append(DE)
    for DE in des:
        sol = _solve_simple(f, x, DE, g, k)
        if sol:
            return sol

def _compute_fps(f, x, x0, dir, hyper, order, rational, full):
    if False:
        print('Hello World!')
    'Recursive wrapper to compute fps.\n\n    See :func:`compute_fps` for details.\n    '
    if x0 in [S.Infinity, S.NegativeInfinity]:
        dir = S.One if x0 is S.Infinity else -S.One
        temp = f.subs(x, 1 / x)
        result = _compute_fps(temp, x, 0, dir, hyper, order, rational, full)
        if result is None:
            return None
        return (result[0], result[1].subs(x, 1 / x), result[2].subs(x, 1 / x))
    elif x0 or dir == -S.One:
        if dir == -S.One:
            rep = -x + x0
            rep2 = -x
            rep2b = x0
        else:
            rep = x + x0
            rep2 = x
            rep2b = -x0
        temp = f.subs(x, rep)
        result = _compute_fps(temp, x, 0, S.One, hyper, order, rational, full)
        if result is None:
            return None
        return (result[0], result[1].subs(x, rep2 + rep2b), result[2].subs(x, rep2 + rep2b))
    if f.is_polynomial(x):
        k = Dummy('k')
        ak = sequence(Coeff(f, x, k), (k, 1, oo))
        xk = sequence(x ** k, (k, 0, oo))
        ind = f.coeff(x, 0)
        return (ak, xk, ind)
    if isinstance(f, Add):
        result = False
        ak = sequence(S.Zero, (0, oo))
        (ind, xk) = (S.Zero, None)
        for t in Add.make_args(f):
            res = _compute_fps(t, x, 0, S.One, hyper, order, rational, full)
            if res:
                if not result:
                    result = True
                    xk = res[1]
                if res[0].start > ak.start:
                    seq = ak
                    (s, f) = (ak.start, res[0].start)
                else:
                    seq = res[0]
                    (s, f) = (res[0].start, ak.start)
                save = Add(*[z[0] * z[1] for z in zip(seq[0:f - s], xk[s:f])])
                ak += res[0]
                ind += res[2] + save
            else:
                ind += t
        if result:
            return (ak, xk, ind)
        return None
    syms = f.free_symbols.difference({x})
    (f, symb) = expand(f).as_independent(*syms)
    result = None
    k = Dummy('k')
    if rational:
        result = rational_algorithm(f, x, k, order, full)
    if result is None and hyper:
        result = hyper_algorithm(f, x, k, order)
    if result is None:
        return None
    from sympy.simplify.powsimp import powsimp
    if symb.is_zero:
        symb = S.One
    else:
        symb = powsimp(symb)
    ak = sequence(result[0], (k, result[2], oo))
    xk_formula = powsimp(x ** k * symb)
    xk = sequence(xk_formula, (k, 0, oo))
    ind = powsimp(result[1] * symb)
    return (ak, xk, ind)

def compute_fps(f, x, x0=0, dir=1, hyper=True, order=4, rational=True, full=False):
    if False:
        return 10
    "\n    Computes the formula for Formal Power Series of a function.\n\n    Explanation\n    ===========\n\n    Tries to compute the formula by applying the following techniques\n    (in order):\n\n    * rational_algorithm\n    * Hypergeometric algorithm\n\n    Parameters\n    ==========\n\n    x : Symbol\n    x0 : number, optional\n        Point to perform series expansion about. Default is 0.\n    dir : {1, -1, '+', '-'}, optional\n        If dir is 1 or '+' the series is calculated from the right and\n        for -1 or '-' the series is calculated from the left. For smooth\n        functions this flag will not alter the results. Default is 1.\n    hyper : {True, False}, optional\n        Set hyper to False to skip the hypergeometric algorithm.\n        By default it is set to False.\n    order : int, optional\n        Order of the derivative of ``f``, Default is 4.\n    rational : {True, False}, optional\n        Set rational to False to skip rational algorithm. By default it is set\n        to True.\n    full : {True, False}, optional\n        Set full to True to increase the range of rational algorithm.\n        See :func:`rational_algorithm` for details. By default it is set to\n        False.\n\n    Returns\n    =======\n\n    ak : sequence\n        Sequence of coefficients.\n    xk : sequence\n        Sequence of powers of x.\n    ind : Expr\n        Independent terms.\n    mul : Pow\n        Common terms.\n\n    See Also\n    ========\n\n    sympy.series.formal.rational_algorithm\n    sympy.series.formal.hyper_algorithm\n    "
    f = sympify(f)
    x = sympify(x)
    if not f.has(x):
        return None
    x0 = sympify(x0)
    if dir == '+':
        dir = S.One
    elif dir == '-':
        dir = -S.One
    elif dir not in [S.One, -S.One]:
        raise ValueError("Dir must be '+' or '-'")
    else:
        dir = sympify(dir)
    return _compute_fps(f, x, x0, dir, hyper, order, rational, full)

class Coeff(Function):
    """
    Coeff(p, x, n) represents the nth coefficient of the polynomial p in x
    """

    @classmethod
    def eval(cls, p, x, n):
        if False:
            i = 10
            return i + 15
        if p.is_polynomial(x) and n.is_integer:
            return p.coeff(x, n)

class FormalPowerSeries(SeriesBase):
    """
    Represents Formal Power Series of a function.

    Explanation
    ===========

    No computation is performed. This class should only to be used to represent
    a series. No checks are performed.

    For computing a series use :func:`fps`.

    See Also
    ========

    sympy.series.formal.fps
    """

    def __new__(cls, *args):
        if False:
            return 10
        args = map(sympify, args)
        return Expr.__new__(cls, *args)

    def __init__(self, *args):
        if False:
            while True:
                i = 10
        ak = args[4][0]
        k = ak.variables[0]
        self.ak_seq = sequence(ak.formula, (k, 1, oo))
        self.fact_seq = sequence(factorial(k), (k, 1, oo))
        self.bell_coeff_seq = self.ak_seq * self.fact_seq
        self.sign_seq = sequence((-1, 1), (k, 1, oo))

    @property
    def function(self):
        if False:
            for i in range(10):
                print('nop')
        return self.args[0]

    @property
    def x(self):
        if False:
            i = 10
            return i + 15
        return self.args[1]

    @property
    def x0(self):
        if False:
            i = 10
            return i + 15
        return self.args[2]

    @property
    def dir(self):
        if False:
            while True:
                i = 10
        return self.args[3]

    @property
    def ak(self):
        if False:
            while True:
                i = 10
        return self.args[4][0]

    @property
    def xk(self):
        if False:
            for i in range(10):
                print('nop')
        return self.args[4][1]

    @property
    def ind(self):
        if False:
            i = 10
            return i + 15
        return self.args[4][2]

    @property
    def interval(self):
        if False:
            i = 10
            return i + 15
        return Interval(0, oo)

    @property
    def start(self):
        if False:
            while True:
                i = 10
        return self.interval.inf

    @property
    def stop(self):
        if False:
            while True:
                i = 10
        return self.interval.sup

    @property
    def length(self):
        if False:
            for i in range(10):
                print('nop')
        return oo

    @property
    def infinite(self):
        if False:
            print('Hello World!')
        'Returns an infinite representation of the series'
        from sympy.concrete import Sum
        (ak, xk) = (self.ak, self.xk)
        k = ak.variables[0]
        inf_sum = Sum(ak.formula * xk.formula, (k, ak.start, ak.stop))
        return self.ind + inf_sum

    def _get_pow_x(self, term):
        if False:
            while True:
                i = 10
        'Returns the power of x in a term.'
        (xterm, pow_x) = term.as_independent(self.x)[1].as_base_exp()
        if not xterm.has(self.x):
            return S.Zero
        return pow_x

    def polynomial(self, n=6):
        if False:
            return 10
        '\n        Truncated series as polynomial.\n\n        Explanation\n        ===========\n\n        Returns series expansion of ``f`` upto order ``O(x**n)``\n        as a polynomial(without ``O`` term).\n        '
        terms = []
        sym = self.free_symbols
        for (i, t) in enumerate(self):
            xp = self._get_pow_x(t)
            if xp.has(*sym):
                xp = xp.as_coeff_add(*sym)[0]
            if xp >= n:
                break
            elif xp.is_integer is True and i == n + 1:
                break
            elif t is not S.Zero:
                terms.append(t)
        return Add(*terms)

    def truncate(self, n=6):
        if False:
            for i in range(10):
                print('nop')
        '\n        Truncated series.\n\n        Explanation\n        ===========\n\n        Returns truncated series expansion of f upto\n        order ``O(x**n)``.\n\n        If n is ``None``, returns an infinite iterator.\n        '
        if n is None:
            return iter(self)
        (x, x0) = (self.x, self.x0)
        pt_xk = self.xk.coeff(n)
        if x0 is S.NegativeInfinity:
            x0 = S.Infinity
        return self.polynomial(n) + Order(pt_xk, (x, x0))

    def zero_coeff(self):
        if False:
            return 10
        return self._eval_term(0)

    def _eval_term(self, pt):
        if False:
            for i in range(10):
                print('nop')
        try:
            pt_xk = self.xk.coeff(pt)
            pt_ak = self.ak.coeff(pt).simplify()
        except IndexError:
            term = S.Zero
        else:
            term = pt_ak * pt_xk
        if self.ind:
            ind = S.Zero
            sym = self.free_symbols
            for t in Add.make_args(self.ind):
                pow_x = self._get_pow_x(t)
                if pow_x.has(*sym):
                    pow_x = pow_x.as_coeff_add(*sym)[0]
                if pt == 0 and pow_x < 1:
                    ind += t
                elif pow_x >= pt and pow_x < pt + 1:
                    ind += t
            term += ind
        return term.collect(self.x)

    def _eval_subs(self, old, new):
        if False:
            print('Hello World!')
        x = self.x
        if old.has(x):
            return self

    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        if False:
            for i in range(10):
                print('nop')
        for t in self:
            if t is not S.Zero:
                return t

    def _eval_derivative(self, x):
        if False:
            print('Hello World!')
        f = self.function.diff(x)
        ind = self.ind.diff(x)
        pow_xk = self._get_pow_x(self.xk.formula)
        ak = self.ak
        k = ak.variables[0]
        if ak.formula.has(x):
            form = []
            for (e, c) in ak.formula.args:
                temp = S.Zero
                for t in Add.make_args(e):
                    pow_x = self._get_pow_x(t)
                    temp += t * (pow_xk + pow_x)
                form.append((temp, c))
            form = Piecewise(*form)
            ak = sequence(form.subs(k, k + 1), (k, ak.start - 1, ak.stop))
        else:
            ak = sequence((ak.formula * pow_xk).subs(k, k + 1), (k, ak.start - 1, ak.stop))
        return self.func(f, self.x, self.x0, self.dir, (ak, self.xk, ind))

    def integrate(self, x=None, **kwargs):
        if False:
            print('Hello World!')
        '\n        Integrate Formal Power Series.\n\n        Examples\n        ========\n\n        >>> from sympy import fps, sin, integrate\n        >>> from sympy.abc import x\n        >>> f = fps(sin(x))\n        >>> f.integrate(x).truncate()\n        -1 + x**2/2 - x**4/24 + O(x**6)\n        >>> integrate(f, (x, 0, 1))\n        1 - cos(1)\n        '
        from sympy.integrals import integrate
        if x is None:
            x = self.x
        elif iterable(x):
            return integrate(self.function, x)
        f = integrate(self.function, x)
        ind = integrate(self.ind, x)
        ind += (f - ind).limit(x, 0)
        pow_xk = self._get_pow_x(self.xk.formula)
        ak = self.ak
        k = ak.variables[0]
        if ak.formula.has(x):
            form = []
            for (e, c) in ak.formula.args:
                temp = S.Zero
                for t in Add.make_args(e):
                    pow_x = self._get_pow_x(t)
                    temp += t / (pow_xk + pow_x + 1)
                form.append((temp, c))
            form = Piecewise(*form)
            ak = sequence(form.subs(k, k - 1), (k, ak.start + 1, ak.stop))
        else:
            ak = sequence((ak.formula / (pow_xk + 1)).subs(k, k - 1), (k, ak.start + 1, ak.stop))
        return self.func(f, self.x, self.x0, self.dir, (ak, self.xk, ind))

    def product(self, other, x=None, n=6):
        if False:
            while True:
                i = 10
        '\n        Multiplies two Formal Power Series, using discrete convolution and\n        return the truncated terms upto specified order.\n\n        Parameters\n        ==========\n\n        n : Number, optional\n            Specifies the order of the term up to which the polynomial should\n            be truncated.\n\n        Examples\n        ========\n\n        >>> from sympy import fps, sin, exp\n        >>> from sympy.abc import x\n        >>> f1 = fps(sin(x))\n        >>> f2 = fps(exp(x))\n\n        >>> f1.product(f2, x).truncate(4)\n        x + x**2 + x**3/3 + O(x**4)\n\n        See Also\n        ========\n\n        sympy.discrete.convolutions\n        sympy.series.formal.FormalPowerSeriesProduct\n\n        '
        if n is None:
            return iter(self)
        other = sympify(other)
        if not isinstance(other, FormalPowerSeries):
            raise ValueError('Both series should be an instance of FormalPowerSeries class.')
        if self.dir != other.dir:
            raise ValueError('Both series should be calculated from the same direction.')
        elif self.x0 != other.x0:
            raise ValueError('Both series should be calculated about the same point.')
        elif self.x != other.x:
            raise ValueError('Both series should have the same symbol.')
        return FormalPowerSeriesProduct(self, other)

    def coeff_bell(self, n):
        if False:
            while True:
                i = 10
        '\n        self.coeff_bell(n) returns a sequence of Bell polynomials of the second kind.\n        Note that ``n`` should be a integer.\n\n        The second kind of Bell polynomials (are sometimes called "partial" Bell\n        polynomials or incomplete Bell polynomials) are defined as\n\n        .. math::\n            B_{n,k}(x_1, x_2,\\dotsc x_{n-k+1}) =\n                \\sum_{j_1+j_2+j_2+\\dotsb=k \\atop j_1+2j_2+3j_2+\\dotsb=n}\n                \\frac{n!}{j_1!j_2!\\dotsb j_{n-k+1}!}\n                \\left(\\frac{x_1}{1!} \\right)^{j_1}\n                \\left(\\frac{x_2}{2!} \\right)^{j_2} \\dotsb\n                \\left(\\frac{x_{n-k+1}}{(n-k+1)!} \\right) ^{j_{n-k+1}}.\n\n        * ``bell(n, k, (x1, x2, ...))`` gives Bell polynomials of the second kind,\n          `B_{n,k}(x_1, x_2, \\dotsc, x_{n-k+1})`.\n\n        See Also\n        ========\n\n        sympy.functions.combinatorial.numbers.bell\n\n        '
        inner_coeffs = [bell(n, j, tuple(self.bell_coeff_seq[:n - j + 1])) for j in range(1, n + 1)]
        k = Dummy('k')
        return sequence(tuple(inner_coeffs), (k, 1, oo))

    def compose(self, other, x=None, n=6):
        if False:
            i = 10
            return i + 15
        '\n        Returns the truncated terms of the formal power series of the composed function,\n        up to specified ``n``.\n\n        Explanation\n        ===========\n\n        If ``f`` and ``g`` are two formal power series of two different functions,\n        then the coefficient sequence ``ak`` of the composed formal power series `fp`\n        will be as follows.\n\n        .. math::\n            \\sum\\limits_{k=0}^{n} b_k B_{n,k}(x_1, x_2, \\dotsc, x_{n-k+1})\n\n        Parameters\n        ==========\n\n        n : Number, optional\n            Specifies the order of the term up to which the polynomial should\n            be truncated.\n\n        Examples\n        ========\n\n        >>> from sympy import fps, sin, exp\n        >>> from sympy.abc import x\n        >>> f1 = fps(exp(x))\n        >>> f2 = fps(sin(x))\n\n        >>> f1.compose(f2, x).truncate()\n        1 + x + x**2/2 - x**4/8 - x**5/15 + O(x**6)\n\n        >>> f1.compose(f2, x).truncate(8)\n        1 + x + x**2/2 - x**4/8 - x**5/15 - x**6/240 + x**7/90 + O(x**8)\n\n        See Also\n        ========\n\n        sympy.functions.combinatorial.numbers.bell\n        sympy.series.formal.FormalPowerSeriesCompose\n\n        References\n        ==========\n\n        .. [1] Comtet, Louis: Advanced combinatorics; the art of finite and infinite expansions. Reidel, 1974.\n\n        '
        if n is None:
            return iter(self)
        other = sympify(other)
        if not isinstance(other, FormalPowerSeries):
            raise ValueError('Both series should be an instance of FormalPowerSeries class.')
        if self.dir != other.dir:
            raise ValueError('Both series should be calculated from the same direction.')
        elif self.x0 != other.x0:
            raise ValueError('Both series should be calculated about the same point.')
        elif self.x != other.x:
            raise ValueError('Both series should have the same symbol.')
        if other._eval_term(0).as_coeff_mul(other.x)[0] is not S.Zero:
            raise ValueError('The formal power series of the inner function should not have any constant coefficient term.')
        return FormalPowerSeriesCompose(self, other)

    def inverse(self, x=None, n=6):
        if False:
            i = 10
            return i + 15
        '\n        Returns the truncated terms of the inverse of the formal power series,\n        up to specified ``n``.\n\n        Explanation\n        ===========\n\n        If ``f`` and ``g`` are two formal power series of two different functions,\n        then the coefficient sequence ``ak`` of the composed formal power series ``fp``\n        will be as follows.\n\n        .. math::\n            \\sum\\limits_{k=0}^{n} (-1)^{k} x_0^{-k-1} B_{n,k}(x_1, x_2, \\dotsc, x_{n-k+1})\n\n        Parameters\n        ==========\n\n        n : Number, optional\n            Specifies the order of the term up to which the polynomial should\n            be truncated.\n\n        Examples\n        ========\n\n        >>> from sympy import fps, exp, cos\n        >>> from sympy.abc import x\n        >>> f1 = fps(exp(x))\n        >>> f2 = fps(cos(x))\n\n        >>> f1.inverse(x).truncate()\n        1 - x + x**2/2 - x**3/6 + x**4/24 - x**5/120 + O(x**6)\n\n        >>> f2.inverse(x).truncate(8)\n        1 + x**2/2 + 5*x**4/24 + 61*x**6/720 + O(x**8)\n\n        See Also\n        ========\n\n        sympy.functions.combinatorial.numbers.bell\n        sympy.series.formal.FormalPowerSeriesInverse\n\n        References\n        ==========\n\n        .. [1] Comtet, Louis: Advanced combinatorics; the art of finite and infinite expansions. Reidel, 1974.\n\n        '
        if n is None:
            return iter(self)
        if self._eval_term(0).is_zero:
            raise ValueError('Constant coefficient should exist for an inverse of a formal power series to exist.')
        return FormalPowerSeriesInverse(self)

    def __add__(self, other):
        if False:
            while True:
                i = 10
        other = sympify(other)
        if isinstance(other, FormalPowerSeries):
            if self.dir != other.dir:
                raise ValueError('Both series should be calculated from the same direction.')
            elif self.x0 != other.x0:
                raise ValueError('Both series should be calculated about the same point.')
            (x, y) = (self.x, other.x)
            f = self.function + other.function.subs(y, x)
            if self.x not in f.free_symbols:
                return f
            ak = self.ak + other.ak
            if self.ak.start > other.ak.start:
                seq = other.ak
                (s, e) = (other.ak.start, self.ak.start)
            else:
                seq = self.ak
                (s, e) = (self.ak.start, other.ak.start)
            save = Add(*[z[0] * z[1] for z in zip(seq[0:e - s], self.xk[s:e])])
            ind = self.ind + other.ind + save
            return self.func(f, x, self.x0, self.dir, (ak, self.xk, ind))
        elif not other.has(self.x):
            f = self.function + other
            ind = self.ind + other
            return self.func(f, self.x, self.x0, self.dir, (self.ak, self.xk, ind))
        return Add(self, other)

    def __radd__(self, other):
        if False:
            print('Hello World!')
        return self.__add__(other)

    def __neg__(self):
        if False:
            while True:
                i = 10
        return self.func(-self.function, self.x, self.x0, self.dir, (-self.ak, self.xk, -self.ind))

    def __sub__(self, other):
        if False:
            return 10
        return self.__add__(-other)

    def __rsub__(self, other):
        if False:
            while True:
                i = 10
        return (-self).__add__(other)

    def __mul__(self, other):
        if False:
            for i in range(10):
                print('nop')
        other = sympify(other)
        if other.has(self.x):
            return Mul(self, other)
        f = self.function * other
        ak = self.ak.coeff_mul(other)
        ind = self.ind * other
        return self.func(f, self.x, self.x0, self.dir, (ak, self.xk, ind))

    def __rmul__(self, other):
        if False:
            print('Hello World!')
        return self.__mul__(other)

class FiniteFormalPowerSeries(FormalPowerSeries):
    """Base Class for Product, Compose and Inverse classes"""

    def __init__(self, *args):
        if False:
            i = 10
            return i + 15
        pass

    @property
    def ffps(self):
        if False:
            return 10
        return self.args[0]

    @property
    def gfps(self):
        if False:
            for i in range(10):
                print('nop')
        return self.args[1]

    @property
    def f(self):
        if False:
            print('Hello World!')
        return self.ffps.function

    @property
    def g(self):
        if False:
            while True:
                i = 10
        return self.gfps.function

    @property
    def infinite(self):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError('No infinite version for an object of FiniteFormalPowerSeries class.')

    def _eval_terms(self, n):
        if False:
            print('Hello World!')
        raise NotImplementedError('(%s)._eval_terms()' % self)

    def _eval_term(self, pt):
        if False:
            return 10
        raise NotImplementedError('By the current logic, one can get termsupto a certain order, instead of getting term by term.')

    def polynomial(self, n):
        if False:
            print('Hello World!')
        return self._eval_terms(n)

    def truncate(self, n=6):
        if False:
            i = 10
            return i + 15
        ffps = self.ffps
        pt_xk = ffps.xk.coeff(n)
        (x, x0) = (ffps.x, ffps.x0)
        return self.polynomial(n) + Order(pt_xk, (x, x0))

    def _eval_derivative(self, x):
        if False:
            return 10
        raise NotImplementedError

    def integrate(self, x):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

class FormalPowerSeriesProduct(FiniteFormalPowerSeries):
    """Represents the product of two formal power series of two functions.

    Explanation
    ===========

    No computation is performed. Terms are calculated using a term by term logic,
    instead of a point by point logic.

    There are two differences between a :obj:`FormalPowerSeries` object and a
    :obj:`FormalPowerSeriesProduct` object. The first argument contains the two
    functions involved in the product. Also, the coefficient sequence contains
    both the coefficient sequence of the formal power series of the involved functions.

    See Also
    ========

    sympy.series.formal.FormalPowerSeries
    sympy.series.formal.FiniteFormalPowerSeries

    """

    def __init__(self, *args):
        if False:
            for i in range(10):
                print('nop')
        (ffps, gfps) = (self.ffps, self.gfps)
        k = ffps.ak.variables[0]
        self.coeff1 = sequence(ffps.ak.formula, (k, 0, oo))
        k = gfps.ak.variables[0]
        self.coeff2 = sequence(gfps.ak.formula, (k, 0, oo))

    @property
    def function(self):
        if False:
            for i in range(10):
                print('nop')
        'Function of the product of two formal power series.'
        return self.f * self.g

    def _eval_terms(self, n):
        if False:
            print('Hello World!')
        '\n        Returns the first ``n`` terms of the product formal power series.\n        Term by term logic is implemented here.\n\n        Examples\n        ========\n\n        >>> from sympy import fps, sin, exp\n        >>> from sympy.abc import x\n        >>> f1 = fps(sin(x))\n        >>> f2 = fps(exp(x))\n        >>> fprod = f1.product(f2, x)\n\n        >>> fprod._eval_terms(4)\n        x**3/3 + x**2 + x\n\n        See Also\n        ========\n\n        sympy.series.formal.FormalPowerSeries.product\n\n        '
        (coeff1, coeff2) = (self.coeff1, self.coeff2)
        aks = convolution(coeff1[:n], coeff2[:n])
        terms = []
        for i in range(0, n):
            terms.append(aks[i] * self.ffps.xk.coeff(i))
        return Add(*terms)

class FormalPowerSeriesCompose(FiniteFormalPowerSeries):
    """
    Represents the composed formal power series of two functions.

    Explanation
    ===========

    No computation is performed. Terms are calculated using a term by term logic,
    instead of a point by point logic.

    There are two differences between a :obj:`FormalPowerSeries` object and a
    :obj:`FormalPowerSeriesCompose` object. The first argument contains the outer
    function and the inner function involved in the omposition. Also, the
    coefficient sequence contains the generic sequence which is to be multiplied
    by a custom ``bell_seq`` finite sequence. The finite terms will then be added up to
    get the final terms.

    See Also
    ========

    sympy.series.formal.FormalPowerSeries
    sympy.series.formal.FiniteFormalPowerSeries

    """

    @property
    def function(self):
        if False:
            return 10
        'Function for the composed formal power series.'
        (f, g, x) = (self.f, self.g, self.ffps.x)
        return f.subs(x, g)

    def _eval_terms(self, n):
        if False:
            i = 10
            return i + 15
        '\n        Returns the first `n` terms of the composed formal power series.\n        Term by term logic is implemented here.\n\n        Explanation\n        ===========\n\n        The coefficient sequence of the :obj:`FormalPowerSeriesCompose` object is the generic sequence.\n        It is multiplied by ``bell_seq`` to get a sequence, whose terms are added up to get\n        the final terms for the polynomial.\n\n        Examples\n        ========\n\n        >>> from sympy import fps, sin, exp\n        >>> from sympy.abc import x\n        >>> f1 = fps(exp(x))\n        >>> f2 = fps(sin(x))\n        >>> fcomp = f1.compose(f2, x)\n\n        >>> fcomp._eval_terms(6)\n        -x**5/15 - x**4/8 + x**2/2 + x + 1\n\n        >>> fcomp._eval_terms(8)\n        x**7/90 - x**6/240 - x**5/15 - x**4/8 + x**2/2 + x + 1\n\n        See Also\n        ========\n\n        sympy.series.formal.FormalPowerSeries.compose\n        sympy.series.formal.FormalPowerSeries.coeff_bell\n\n        '
        (ffps, gfps) = (self.ffps, self.gfps)
        terms = [ffps.zero_coeff()]
        for i in range(1, n):
            bell_seq = gfps.coeff_bell(i)
            seq = ffps.bell_coeff_seq * bell_seq
            terms.append(Add(*seq[:i]) / ffps.fact_seq[i - 1] * ffps.xk.coeff(i))
        return Add(*terms)

class FormalPowerSeriesInverse(FiniteFormalPowerSeries):
    """
    Represents the Inverse of a formal power series.

    Explanation
    ===========

    No computation is performed. Terms are calculated using a term by term logic,
    instead of a point by point logic.

    There is a single difference between a :obj:`FormalPowerSeries` object and a
    :obj:`FormalPowerSeriesInverse` object. The coefficient sequence contains the
    generic sequence which is to be multiplied by a custom ``bell_seq`` finite sequence.
    The finite terms will then be added up to get the final terms.

    See Also
    ========

    sympy.series.formal.FormalPowerSeries
    sympy.series.formal.FiniteFormalPowerSeries

    """

    def __init__(self, *args):
        if False:
            return 10
        ffps = self.ffps
        k = ffps.xk.variables[0]
        inv = ffps.zero_coeff()
        inv_seq = sequence(inv ** (-(k + 1)), (k, 1, oo))
        self.aux_seq = ffps.sign_seq * ffps.fact_seq * inv_seq

    @property
    def function(self):
        if False:
            print('Hello World!')
        'Function for the inverse of a formal power series.'
        f = self.f
        return 1 / f

    @property
    def g(self):
        if False:
            return 10
        raise ValueError('Only one function is considered while performinginverse of a formal power series.')

    @property
    def gfps(self):
        if False:
            for i in range(10):
                print('nop')
        raise ValueError('Only one function is considered while performinginverse of a formal power series.')

    def _eval_terms(self, n):
        if False:
            return 10
        '\n        Returns the first ``n`` terms of the composed formal power series.\n        Term by term logic is implemented here.\n\n        Explanation\n        ===========\n\n        The coefficient sequence of the `FormalPowerSeriesInverse` object is the generic sequence.\n        It is multiplied by ``bell_seq`` to get a sequence, whose terms are added up to get\n        the final terms for the polynomial.\n\n        Examples\n        ========\n\n        >>> from sympy import fps, exp, cos\n        >>> from sympy.abc import x\n        >>> f1 = fps(exp(x))\n        >>> f2 = fps(cos(x))\n        >>> finv1, finv2 = f1.inverse(), f2.inverse()\n\n        >>> finv1._eval_terms(6)\n        -x**5/120 + x**4/24 - x**3/6 + x**2/2 - x + 1\n\n        >>> finv2._eval_terms(8)\n        61*x**6/720 + 5*x**4/24 + x**2/2 + 1\n\n        See Also\n        ========\n\n        sympy.series.formal.FormalPowerSeries.inverse\n        sympy.series.formal.FormalPowerSeries.coeff_bell\n\n        '
        ffps = self.ffps
        terms = [ffps.zero_coeff()]
        for i in range(1, n):
            bell_seq = ffps.coeff_bell(i)
            seq = self.aux_seq * bell_seq
            terms.append(Add(*seq[:i]) / ffps.fact_seq[i - 1] * ffps.xk.coeff(i))
        return Add(*terms)

def fps(f, x=None, x0=0, dir=1, hyper=True, order=4, rational=True, full=False):
    if False:
        for i in range(10):
            print('nop')
    "\n    Generates Formal Power Series of ``f``.\n\n    Explanation\n    ===========\n\n    Returns the formal series expansion of ``f`` around ``x = x0``\n    with respect to ``x`` in the form of a ``FormalPowerSeries`` object.\n\n    Formal Power Series is represented using an explicit formula\n    computed using different algorithms.\n\n    See :func:`compute_fps` for the more details regarding the computation\n    of formula.\n\n    Parameters\n    ==========\n\n    x : Symbol, optional\n        If x is None and ``f`` is univariate, the univariate symbols will be\n        supplied, otherwise an error will be raised.\n    x0 : number, optional\n        Point to perform series expansion about. Default is 0.\n    dir : {1, -1, '+', '-'}, optional\n        If dir is 1 or '+' the series is calculated from the right and\n        for -1 or '-' the series is calculated from the left. For smooth\n        functions this flag will not alter the results. Default is 1.\n    hyper : {True, False}, optional\n        Set hyper to False to skip the hypergeometric algorithm.\n        By default it is set to False.\n    order : int, optional\n        Order of the derivative of ``f``, Default is 4.\n    rational : {True, False}, optional\n        Set rational to False to skip rational algorithm. By default it is set\n        to True.\n    full : {True, False}, optional\n        Set full to True to increase the range of rational algorithm.\n        See :func:`rational_algorithm` for details. By default it is set to\n        False.\n\n    Examples\n    ========\n\n    >>> from sympy import fps, ln, atan, sin\n    >>> from sympy.abc import x, n\n\n    Rational Functions\n\n    >>> fps(ln(1 + x)).truncate()\n    x - x**2/2 + x**3/3 - x**4/4 + x**5/5 + O(x**6)\n\n    >>> fps(atan(x), full=True).truncate()\n    x - x**3/3 + x**5/5 + O(x**6)\n\n    Symbolic Functions\n\n    >>> fps(x**n*sin(x**2), x).truncate(8)\n    -x**(n + 6)/6 + x**(n + 2) + O(x**(n + 8))\n\n    See Also\n    ========\n\n    sympy.series.formal.FormalPowerSeries\n    sympy.series.formal.compute_fps\n    "
    f = sympify(f)
    if x is None:
        free = f.free_symbols
        if len(free) == 1:
            x = free.pop()
        elif not free:
            return f
        else:
            raise NotImplementedError('multivariate formal power series')
    result = compute_fps(f, x, x0, dir, hyper, order, rational, full)
    if result is None:
        return f
    return FormalPowerSeries(f, x, x0, dir, result)