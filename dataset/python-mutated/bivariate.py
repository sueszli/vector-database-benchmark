from sympy.core.add import Add
from sympy.core.exprtools import factor_terms
from sympy.core.function import expand_log, _mexpand
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.sorting import ordered
from sympy.core.symbol import Dummy
from sympy.functions.elementary.exponential import LambertW, exp, log
from sympy.functions.elementary.miscellaneous import root
from sympy.polys.polyroots import roots
from sympy.polys.polytools import Poly, factor
from sympy.simplify.simplify import separatevars
from sympy.simplify.radsimp import collect
from sympy.simplify.simplify import powsimp
from sympy.solvers.solvers import solve, _invert
from sympy.utilities.iterables import uniq

def _filtered_gens(poly, symbol):
    if False:
        print('Hello World!')
    'process the generators of ``poly``, returning the set of generators that\n    have ``symbol``.  If there are two generators that are inverses of each other,\n    prefer the one that has no denominator.\n\n    Examples\n    ========\n\n    >>> from sympy.solvers.bivariate import _filtered_gens\n    >>> from sympy import Poly, exp\n    >>> from sympy.abc import x\n    >>> _filtered_gens(Poly(x + 1/x + exp(x)), x)\n    {x, exp(x)}\n\n    '
    gens = {g for g in poly.gens if symbol in g.free_symbols}
    for g in list(gens):
        ag = 1 / g
        if g in gens and ag in gens:
            if ag.as_numer_denom()[1] is not S.One:
                g = ag
            gens.remove(g)
    return gens

def _mostfunc(lhs, func, X=None):
    if False:
        for i in range(10):
            print('nop')
    'Returns the term in lhs which contains the most of the\n    func-type things e.g. log(log(x)) wins over log(x) if both terms appear.\n\n    ``func`` can be a function (exp, log, etc...) or any other SymPy object,\n    like Pow.\n\n    If ``X`` is not ``None``, then the function returns the term composed with the\n    most ``func`` having the specified variable.\n\n    Examples\n    ========\n\n    >>> from sympy.solvers.bivariate import _mostfunc\n    >>> from sympy import exp\n    >>> from sympy.abc import x, y\n    >>> _mostfunc(exp(x) + exp(exp(x) + 2), exp)\n    exp(exp(x) + 2)\n    >>> _mostfunc(exp(x) + exp(exp(y) + 2), exp)\n    exp(exp(y) + 2)\n    >>> _mostfunc(exp(x) + exp(exp(y) + 2), exp, x)\n    exp(x)\n    >>> _mostfunc(x, exp, x) is None\n    True\n    >>> _mostfunc(exp(x) + exp(x*y), exp, x)\n    exp(x)\n    '
    fterms = [tmp for tmp in lhs.atoms(func) if not X or (X.is_Symbol and X in tmp.free_symbols) or (not X.is_Symbol and tmp.has(X))]
    if len(fterms) == 1:
        return fterms[0]
    elif fterms:
        return max(list(ordered(fterms)), key=lambda x: x.count(func))
    return None

def _linab(arg, symbol):
    if False:
        for i in range(10):
            print('nop')
    'Return ``a, b, X`` assuming ``arg`` can be written as ``a*X + b``\n    where ``X`` is a symbol-dependent factor and ``a`` and ``b`` are\n    independent of ``symbol``.\n\n    Examples\n    ========\n\n    >>> from sympy.solvers.bivariate import _linab\n    >>> from sympy.abc import x, y\n    >>> from sympy import exp, S\n    >>> _linab(S(2), x)\n    (2, 0, 1)\n    >>> _linab(2*x, x)\n    (2, 0, x)\n    >>> _linab(y + y*x + 2*x, x)\n    (y + 2, y, x)\n    >>> _linab(3 + 2*exp(x), x)\n    (2, 3, exp(x))\n    '
    arg = factor_terms(arg.expand())
    (ind, dep) = arg.as_independent(symbol)
    if arg.is_Mul and dep.is_Add:
        (a, b, x) = _linab(dep, symbol)
        return (ind * a, ind * b, x)
    if not arg.is_Add:
        b = 0
        (a, x) = (ind, dep)
    else:
        b = ind
        (a, x) = separatevars(dep).as_independent(symbol, as_Add=False)
    if x.could_extract_minus_sign():
        a = -a
        x = -x
    return (a, b, x)

def _lambert(eq, x):
    if False:
        for i in range(10):
            print('nop')
    '\n    Given an expression assumed to be in the form\n        ``F(X, a..f) = a*log(b*X + c) + d*X + f = 0``\n    where X = g(x) and x = g^-1(X), return the Lambert solution,\n        ``x = g^-1(-c/b + (a/d)*W(d/(a*b)*exp(c*d/a/b)*exp(-f/a)))``.\n    '
    eq = _mexpand(expand_log(eq))
    mainlog = _mostfunc(eq, log, x)
    if not mainlog:
        return []
    other = eq.subs(mainlog, 0)
    if isinstance(-other, log):
        eq = (eq - other).subs(mainlog, mainlog.args[0])
        mainlog = mainlog.args[0]
        if not isinstance(mainlog, log):
            return []
        other = -(-other).args[0]
        eq += other
    if x not in other.free_symbols:
        return []
    (d, f, X2) = _linab(other, x)
    logterm = collect(eq - other, mainlog)
    a = logterm.as_coefficient(mainlog)
    if a is None or x in a.free_symbols:
        return []
    logarg = mainlog.args[0]
    (b, c, X1) = _linab(logarg, x)
    if X1 != X2:
        return []
    u = Dummy('rhs')
    xusolns = solve(X1 - u, x)
    lambert_real_branches = [-1, 0]
    sol = []
    (num, den) = ((c * d - b * f) / a / b).as_numer_denom()
    (p, den) = den.as_coeff_Mul()
    e = exp(num / den)
    t = Dummy('t')
    args = [d / (a * b) * t for t in roots(t ** p - e, t).keys()]
    for arg in args:
        for k in lambert_real_branches:
            w = LambertW(arg, k)
            if k and (not w.is_real):
                continue
            rhs = -c / b + a / d * w
            sol.extend((xu.subs(u, rhs) for xu in xusolns))
    return sol

def _solve_lambert(f, symbol, gens):
    if False:
        print('Hello World!')
    'Return solution to ``f`` if it is a Lambert-type expression\n    else raise NotImplementedError.\n\n    For ``f(X, a..f) = a*log(b*X + c) + d*X - f = 0`` the solution\n    for ``X`` is ``X = -c/b + (a/d)*W(d/(a*b)*exp(c*d/a/b)*exp(f/a))``.\n    There are a variety of forms for `f(X, a..f)` as enumerated below:\n\n    1a1)\n      if B**B = R for R not in [0, 1] (since those cases would already\n      be solved before getting here) then log of both sides gives\n      log(B) + log(log(B)) = log(log(R)) and\n      X = log(B), a = 1, b = 1, c = 0, d = 1, f = log(log(R))\n    1a2)\n      if B*(b*log(B) + c)**a = R then log of both sides gives\n      log(B) + a*log(b*log(B) + c) = log(R) and\n      X = log(B), d=1, f=log(R)\n    1b)\n      if a*log(b*B + c) + d*B = R and\n      X = B, f = R\n    2a)\n      if (b*B + c)*exp(d*B + g) = R then log of both sides gives\n      log(b*B + c) + d*B + g = log(R) and\n      X = B, a = 1, f = log(R) - g\n    2b)\n      if g*exp(d*B + h) - b*B = c then the log form is\n      log(g) + d*B + h - log(b*B + c) = 0 and\n      X = B, a = -1, f = -h - log(g)\n    3)\n      if d*p**(a*B + g) - b*B = c then the log form is\n      log(d) + (a*B + g)*log(p) - log(b*B + c) = 0 and\n      X = B, a = -1, d = a*log(p), f = -log(d) - g*log(p)\n    '

    def _solve_even_degree_expr(expr, t, symbol):
        if False:
            while True:
                i = 10
        'Return the unique solutions of equations derived from\n        ``expr`` by replacing ``t`` with ``+/- symbol``.\n\n        Parameters\n        ==========\n\n        expr : Expr\n            The expression which includes a dummy variable t to be\n            replaced with +symbol and -symbol.\n\n        symbol : Symbol\n            The symbol for which a solution is being sought.\n\n        Returns\n        =======\n\n        List of unique solution of the two equations generated by\n        replacing ``t`` with positive and negative ``symbol``.\n\n        Notes\n        =====\n\n        If ``expr = 2*log(t) + x/2` then solutions for\n        ``2*log(x) + x/2 = 0`` and ``2*log(-x) + x/2 = 0`` are\n        returned by this function. Though this may seem\n        counter-intuitive, one must note that the ``expr`` being\n        solved here has been derived from a different expression. For\n        an expression like ``eq = x**2*g(x) = 1``, if we take the\n        log of both sides we obtain ``log(x**2) + log(g(x)) = 0``. If\n        x is positive then this simplifies to\n        ``2*log(x) + log(g(x)) = 0``; the Lambert-solving routines will\n        return solutions for this, but we must also consider the\n        solutions for  ``2*log(-x) + log(g(x))`` since those must also\n        be a solution of ``eq`` which has the same value when the ``x``\n        in ``x**2`` is negated. If `g(x)` does not have even powers of\n        symbol then we do not want to replace the ``x`` there with\n        ``-x``. So the role of the ``t`` in the expression received by\n        this function is to mark where ``+/-x`` should be inserted\n        before obtaining the Lambert solutions.\n\n        '
        (nlhs, plhs) = [expr.xreplace({t: sgn * symbol}) for sgn in (-1, 1)]
        sols = _solve_lambert(nlhs, symbol, gens)
        if plhs != nlhs:
            sols.extend(_solve_lambert(plhs, symbol, gens))
        return list(uniq(sols))
    (nrhs, lhs) = f.as_independent(symbol, as_Add=True)
    rhs = -nrhs
    lamcheck = [tmp for tmp in gens if tmp.func in [exp, log] or (tmp.is_Pow and symbol in tmp.exp.free_symbols)]
    if not lamcheck:
        raise NotImplementedError()
    if lhs.is_Add or lhs.is_Mul:
        t = Dummy('t', **symbol.assumptions0)
        lhs = lhs.replace(lambda i: i.is_Pow and i.base == symbol and i.exp.is_even, lambda i: t ** i.exp)
        if lhs.is_Add and lhs.has(t):
            t_indep = lhs.subs(t, 0)
            t_term = lhs - t_indep
            _rhs = rhs - t_indep
            if not t_term.is_Add and _rhs and (not t_term.has(S.ComplexInfinity, S.NaN)):
                eq = expand_log(log(t_term) - log(_rhs))
                return _solve_even_degree_expr(eq, t, symbol)
        elif lhs.is_Mul and rhs:
            lhs = expand_log(log(lhs), force=True)
            rhs = log(rhs)
            if lhs.has(t) and lhs.is_Add:
                eq = lhs - rhs
                return _solve_even_degree_expr(eq, t, symbol)
        lhs = lhs.xreplace({t: symbol})
    lhs = powsimp(factor(lhs, deep=True))
    r = Dummy()
    (i, lhs) = _invert(lhs - r, symbol)
    rhs = i.xreplace({r: rhs})
    soln = []
    if not soln:
        mainlog = _mostfunc(lhs, log, symbol)
        if mainlog:
            if lhs.is_Mul and rhs != 0:
                soln = _lambert(log(lhs) - log(rhs), symbol)
            elif lhs.is_Add:
                other = lhs.subs(mainlog, 0)
                if other and (not other.is_Add) and [tmp for tmp in other.atoms(Pow) if symbol in tmp.free_symbols]:
                    if not rhs:
                        diff = log(other) - log(other - lhs)
                    else:
                        diff = log(lhs - other) - log(rhs - other)
                    soln = _lambert(expand_log(diff), symbol)
                else:
                    soln = _lambert(lhs - rhs, symbol)
    if not soln:
        mainexp = _mostfunc(lhs, exp, symbol)
        if mainexp:
            lhs = collect(lhs, mainexp)
            if lhs.is_Mul and rhs != 0:
                soln = _lambert(expand_log(log(lhs) - log(rhs)), symbol)
            elif lhs.is_Add:
                other = lhs.subs(mainexp, 0)
                mainterm = lhs - other
                rhs = rhs - other
                if mainterm.could_extract_minus_sign() and rhs.could_extract_minus_sign():
                    mainterm *= -1
                    rhs *= -1
                diff = log(mainterm) - log(rhs)
                soln = _lambert(expand_log(diff), symbol)
    if not soln:
        mainpow = _mostfunc(lhs, Pow, symbol)
        if mainpow and symbol in mainpow.exp.free_symbols:
            lhs = collect(lhs, mainpow)
            if lhs.is_Mul and rhs != 0:
                soln = _lambert(expand_log(log(lhs) - log(rhs)), symbol)
            elif lhs.is_Add:
                other = lhs.subs(mainpow, 0)
                mainterm = lhs - other
                rhs = rhs - other
                diff = log(mainterm) - log(rhs)
                soln = _lambert(expand_log(diff), symbol)
    if not soln:
        raise NotImplementedError('%s does not appear to have a solution in terms of LambertW' % f)
    return list(ordered(soln))

def bivariate_type(f, x, y, *, first=True):
    if False:
        return 10
    'Given an expression, f, 3 tests will be done to see what type\n    of composite bivariate it might be, options for u(x, y) are::\n\n        x*y\n        x+y\n        x*y+x\n        x*y+y\n\n    If it matches one of these types, ``u(x, y)``, ``P(u)`` and dummy\n    variable ``u`` will be returned. Solving ``P(u)`` for ``u`` and\n    equating the solutions to ``u(x, y)`` and then solving for ``x`` or\n    ``y`` is equivalent to solving the original expression for ``x`` or\n    ``y``. If ``x`` and ``y`` represent two functions in the same\n    variable, e.g. ``x = g(t)`` and ``y = h(t)``, then if ``u(x, y) - p``\n    can be solved for ``t`` then these represent the solutions to\n    ``P(u) = 0`` when ``p`` are the solutions of ``P(u) = 0``.\n\n    Only positive values of ``u`` are considered.\n\n    Examples\n    ========\n\n    >>> from sympy import solve\n    >>> from sympy.solvers.bivariate import bivariate_type\n    >>> from sympy.abc import x, y\n    >>> eq = (x**2 - 3).subs(x, x + y)\n    >>> bivariate_type(eq, x, y)\n    (x + y, _u**2 - 3, _u)\n    >>> uxy, pu, u = _\n    >>> usol = solve(pu, u); usol\n    [sqrt(3)]\n    >>> [solve(uxy - s) for s in solve(pu, u)]\n    [[{x: -y + sqrt(3)}]]\n    >>> all(eq.subs(s).equals(0) for sol in _ for s in sol)\n    True\n\n    '
    u = Dummy('u', positive=True)
    if first:
        p = Poly(f, x, y)
        f = p.as_expr()
        _x = Dummy()
        _y = Dummy()
        rv = bivariate_type(Poly(f.subs({x: _x, y: _y}), _x, _y), _x, _y, first=False)
        if rv:
            reps = {_x: x, _y: y}
            return (rv[0].xreplace(reps), rv[1].xreplace(reps), rv[2])
        return
    p = f
    f = p.as_expr()
    args = Add.make_args(p.as_expr())
    new = []
    for a in args:
        a = _mexpand(a.subs(x, u / y))
        free = a.free_symbols
        if x in free or y in free:
            break
        new.append(a)
    else:
        return (x * y, Add(*new), u)

    def ok(f, v, c):
        if False:
            i = 10
            return i + 15
        new = _mexpand(f.subs(v, c))
        free = new.free_symbols
        return None if x in free or y in free else new
    new = []
    d = p.degree(x)
    if p.degree(y) == d:
        a = root(p.coeff_monomial(x ** d), d)
        b = root(p.coeff_monomial(y ** d), d)
        new = ok(f, x, (u - b * y) / a)
        if new is not None:
            return (a * x + b * y, new, u)
    new = []
    d = p.degree(x)
    if p.degree(y) == d:
        for itry in range(2):
            a = root(p.coeff_monomial(x ** d * y ** d), d)
            b = root(p.coeff_monomial(y ** d), d)
            new = ok(f, x, (u - b * y) / a / y)
            if new is not None:
                return (a * x * y + b * y, new, u)
            (x, y) = (y, x)