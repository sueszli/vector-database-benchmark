"""
This module contains the implementation of the internal helper functions for the lie_group hint for
dsolve. These helper functions apply different heuristics on the given equation
and return the solution. These functions are used by :py:meth:`sympy.solvers.ode.single.LieGroup`

References
=========

- `abaco1_simple`, `function_sum` and `chi`  are referenced from E.S Cheb-Terrab, L.G.S Duarte
and L.A,C.P da Mota, Computer Algebra Solving of First Order ODEs Using
Symmetry Methods, pp. 7 - pp. 8

- `abaco1_product`, `abaco2_similar`, `abaco2_unique_unknown`, `linear`  and `abaco2_unique_general`
are referenced from E.S. Cheb-Terrab, A.D. Roche, Symmetries and First Order
ODE Patterns, pp. 7 - pp. 12

- `bivariate` from Lie Groups and Differential Equations pp. 327 - pp. 329

"""
from itertools import islice
from sympy.core import Add, S, Mul, Pow
from sympy.core.exprtools import factor_terms
from sympy.core.function import Function, AppliedUndef, expand
from sympy.core.relational import Equality, Eq
from sympy.core.symbol import Symbol, Wild, Dummy, symbols
from sympy.functions import exp, log
from sympy.integrals.integrals import integrate
from sympy.polys import Poly
from sympy.polys.polytools import cancel, div
from sympy.simplify import collect, powsimp, separatevars, simplify
from sympy.solvers import solve
from sympy.solvers.pde import pdsolve
from sympy.utilities import numbered_symbols
from sympy.solvers.deutils import _preprocess, ode_order
from .ode import checkinfsol
lie_heuristics = ('abaco1_simple', 'abaco1_product', 'abaco2_similar', 'abaco2_unique_unknown', 'abaco2_unique_general', 'linear', 'function_sum', 'bivariate', 'chi')

def _ode_lie_group_try_heuristic(eq, heuristic, func, match, inf):
    if False:
        return 10
    xi = Function('xi')
    eta = Function('eta')
    f = func.func
    x = func.args[0]
    y = match['y']
    h = match['h']
    tempsol = []
    if not inf:
        try:
            inf = infinitesimals(eq, hint=heuristic, func=func, order=1, match=match)
        except ValueError:
            return None
    for infsim in inf:
        xiinf = infsim[xi(x, func)].subs(func, y)
        etainf = infsim[eta(x, func)].subs(func, y)
        if simplify(etainf / xiinf) == h:
            continue
        rpde = f(x, y).diff(x) * xiinf + f(x, y).diff(y) * etainf
        r = pdsolve(rpde, func=f(x, y)).rhs
        s = pdsolve(rpde - 1, func=f(x, y)).rhs
        newcoord = [_lie_group_remove(coord) for coord in [r, s]]
        r = Dummy('r')
        s = Dummy('s')
        C1 = Symbol('C1')
        rcoord = newcoord[0]
        scoord = newcoord[-1]
        try:
            sol = solve([r - rcoord, s - scoord], x, y, dict=True)
            if sol == []:
                continue
        except NotImplementedError:
            continue
        else:
            sol = sol[0]
            xsub = sol[x]
            ysub = sol[y]
            num = simplify(scoord.diff(x) + scoord.diff(y) * h)
            denom = simplify(rcoord.diff(x) + rcoord.diff(y) * h)
            if num and denom:
                diffeq = simplify((num / denom).subs([(x, xsub), (y, ysub)]))
                sep = separatevars(diffeq, symbols=[r, s], dict=True)
                if sep:
                    deq = integrate(1 / sep[s], s) + C1 - integrate(sep['coeff'] * sep[r], r)
                    deq = deq.subs([(r, rcoord), (s, scoord)])
                    try:
                        sdeq = solve(deq, y)
                    except NotImplementedError:
                        tempsol.append(deq)
                    else:
                        return [Eq(f(x), sol) for sol in sdeq]
            elif denom:
                return [Eq(f(x), solve(scoord - C1, y)[0])]
            elif num:
                return [Eq(f(x), solve(rcoord - C1, y)[0])]
    if tempsol:
        return [Eq(sol.subs(y, f(x)), 0) for sol in tempsol]
    return None

def _ode_lie_group(s, func, order, match):
    if False:
        for i in range(10):
            print('nop')
    heuristics = lie_heuristics
    inf = {}
    f = func.func
    x = func.args[0]
    df = func.diff(x)
    xi = Function('xi')
    eta = Function('eta')
    xis = match['xi']
    etas = match['eta']
    y = match.pop('y', None)
    if y:
        h = -simplify(match[match['d']] / match[match['e']])
        y = y
    else:
        y = Dummy('y')
        h = s.subs(func, y)
    if xis is not None and etas is not None:
        inf = [{xi(x, f(x)): S(xis), eta(x, f(x)): S(etas)}]
        if checkinfsol(Eq(df, s), inf, func=f(x), order=1)[0][0]:
            heuristics = ['user_defined'] + list(heuristics)
    match = {'h': h, 'y': y}
    sol = None
    for heuristic in heuristics:
        sol = _ode_lie_group_try_heuristic(Eq(df, s), heuristic, func, match, inf)
        if sol:
            return sol
    return sol

def infinitesimals(eq, func=None, order=None, hint='default', match=None):
    if False:
        i = 10
        return i + 15
    "\n    The infinitesimal functions of an ordinary differential equation, `\\xi(x,y)`\n    and `\\eta(x,y)`, are the infinitesimals of the Lie group of point transformations\n    for which the differential equation is invariant. So, the ODE `y'=f(x,y)`\n    would admit a Lie group `x^*=X(x,y;\\varepsilon)=x+\\varepsilon\\xi(x,y)`,\n    `y^*=Y(x,y;\\varepsilon)=y+\\varepsilon\\eta(x,y)` such that `(y^*)'=f(x^*, y^*)`.\n    A change of coordinates, to `r(x,y)` and `s(x,y)`, can be performed so this Lie group\n    becomes the translation group, `r^*=r` and `s^*=s+\\varepsilon`.\n    They are tangents to the coordinate curves of the new system.\n\n    Consider the transformation `(x, y) \\to (X, Y)` such that the\n    differential equation remains invariant. `\\xi` and `\\eta` are the tangents to\n    the transformed coordinates `X` and `Y`, at `\\varepsilon=0`.\n\n    .. math:: \\left(\\frac{\\partial X(x,y;\\varepsilon)}{\\partial\\varepsilon\n                }\\right)|_{\\varepsilon=0} = \\xi,\n              \\left(\\frac{\\partial Y(x,y;\\varepsilon)}{\\partial\\varepsilon\n                }\\right)|_{\\varepsilon=0} = \\eta,\n\n    The infinitesimals can be found by solving the following PDE:\n\n        >>> from sympy import Function, Eq, pprint\n        >>> from sympy.abc import x, y\n        >>> xi, eta, h = map(Function, ['xi', 'eta', 'h'])\n        >>> h = h(x, y)  # dy/dx = h\n        >>> eta = eta(x, y)\n        >>> xi = xi(x, y)\n        >>> genform = Eq(eta.diff(x) + (eta.diff(y) - xi.diff(x))*h\n        ... - (xi.diff(y))*h**2 - xi*(h.diff(x)) - eta*(h.diff(y)), 0)\n        >>> pprint(genform)\n        /d               d           \\                     d              2       d\n        >\n        |--(eta(x, y)) - --(xi(x, y))|*h(x, y) - eta(x, y)*--(h(x, y)) - h (x, y)*--(x\n        >\n        \\dy              dx          /                     dy                     dy\n        >\n        <BLANKLINE>\n        >                     d             d\n        > i(x, y)) - xi(x, y)*--(h(x, y)) + --(eta(x, y)) = 0\n        >                     dx            dx\n\n    Solving the above mentioned PDE is not trivial, and can be solved only by\n    making intelligent assumptions for `\\xi` and `\\eta` (heuristics). Once an\n    infinitesimal is found, the attempt to find more heuristics stops. This is done to\n    optimise the speed of solving the differential equation. If a list of all the\n    infinitesimals is needed, ``hint`` should be flagged as ``all``, which gives\n    the complete list of infinitesimals. If the infinitesimals for a particular\n    heuristic needs to be found, it can be passed as a flag to ``hint``.\n\n    Examples\n    ========\n\n    >>> from sympy import Function\n    >>> from sympy.solvers.ode.lie_group import infinitesimals\n    >>> from sympy.abc import x\n    >>> f = Function('f')\n    >>> eq = f(x).diff(x) - x**2*f(x)\n    >>> infinitesimals(eq)\n    [{eta(x, f(x)): exp(x**3/3), xi(x, f(x)): 0}]\n\n    References\n    ==========\n\n    - Solving differential equations by Symmetry Groups,\n      John Starrett, pp. 1 - pp. 14\n\n    "
    if isinstance(eq, Equality):
        eq = eq.lhs - eq.rhs
    if not func:
        (eq, func) = _preprocess(eq)
    variables = func.args
    if len(variables) != 1:
        raise ValueError("ODE's have only one independent variable")
    else:
        x = variables[0]
        if not order:
            order = ode_order(eq, func)
        if order != 1:
            raise NotImplementedError("Infinitesimals for only first order ODE's have been implemented")
        else:
            df = func.diff(x)
            a = Wild('a', exclude=[df])
            b = Wild('b', exclude=[df])
            if match:
                h = match['h']
                y = match['y']
            else:
                match = collect(expand(eq), df).match(a * df + b)
                if match:
                    h = -simplify(match[b] / match[a])
                else:
                    try:
                        sol = solve(eq, df)
                    except NotImplementedError:
                        raise NotImplementedError('Infinitesimals for the first order ODE could not be found')
                    else:
                        h = sol[0]
                y = Dummy('y')
                h = h.subs(func, y)
            u = Dummy('u')
            hx = h.diff(x)
            hy = h.diff(y)
            hinv = (1 / h).subs([(x, u), (y, x)]).subs(u, y)
            match = {'h': h, 'func': func, 'hx': hx, 'hy': hy, 'y': y, 'hinv': hinv}
            if hint == 'all':
                xieta = []
                for heuristic in lie_heuristics:
                    function = globals()['lie_heuristic_' + heuristic]
                    inflist = function(match, comp=True)
                    if inflist:
                        xieta.extend([inf for inf in inflist if inf not in xieta])
                if xieta:
                    return xieta
                else:
                    raise NotImplementedError('Infinitesimals could not be found for the given ODE')
            elif hint == 'default':
                for heuristic in lie_heuristics:
                    function = globals()['lie_heuristic_' + heuristic]
                    xieta = function(match, comp=False)
                    if xieta:
                        return xieta
                raise NotImplementedError('Infinitesimals could not be found for the given ODE')
            elif hint not in lie_heuristics:
                raise ValueError('Heuristic not recognized: ' + hint)
            else:
                function = globals()['lie_heuristic_' + hint]
                xieta = function(match, comp=True)
                if xieta:
                    return xieta
                else:
                    raise ValueError('Infinitesimals could not be found using the given heuristic')

def lie_heuristic_abaco1_simple(match, comp=False):
    if False:
        return 10
    "\n    The first heuristic uses the following four sets of\n    assumptions on `\\xi` and `\\eta`\n\n    .. math:: \\xi = 0, \\eta = f(x)\n\n    .. math:: \\xi = 0, \\eta = f(y)\n\n    .. math:: \\xi = f(x), \\eta = 0\n\n    .. math:: \\xi = f(y), \\eta = 0\n\n    The success of this heuristic is determined by algebraic factorisation.\n    For the first assumption `\\xi = 0` and `\\eta` to be a function of `x`, the PDE\n\n    .. math:: \\frac{\\partial \\eta}{\\partial x} + (\\frac{\\partial \\eta}{\\partial y}\n                - \\frac{\\partial \\xi}{\\partial x})*h\n                - \\frac{\\partial \\xi}{\\partial y}*h^{2}\n                - \\xi*\\frac{\\partial h}{\\partial x} - \\eta*\\frac{\\partial h}{\\partial y} = 0\n\n    reduces to `f'(x) - f\\frac{\\partial h}{\\partial y} = 0`\n    If `\\frac{\\partial h}{\\partial y}` is a function of `x`, then this can usually\n    be integrated easily. A similar idea is applied to the other 3 assumptions as well.\n\n\n    References\n    ==========\n\n    - E.S Cheb-Terrab, L.G.S Duarte and L.A,C.P da Mota, Computer Algebra\n      Solving of First Order ODEs Using Symmetry Methods, pp. 8\n\n\n    "
    xieta = []
    y = match['y']
    h = match['h']
    func = match['func']
    x = func.args[0]
    hx = match['hx']
    hy = match['hy']
    xi = Function('xi')(x, func)
    eta = Function('eta')(x, func)
    hysym = hy.free_symbols
    if y not in hysym:
        try:
            fx = exp(integrate(hy, x))
        except NotImplementedError:
            pass
        else:
            inf = {xi: S.Zero, eta: fx}
            if not comp:
                return [inf]
            if comp and inf not in xieta:
                xieta.append(inf)
    factor = hy / h
    facsym = factor.free_symbols
    if x not in facsym:
        try:
            fy = exp(integrate(factor, y))
        except NotImplementedError:
            pass
        else:
            inf = {xi: S.Zero, eta: fy.subs(y, func)}
            if not comp:
                return [inf]
            if comp and inf not in xieta:
                xieta.append(inf)
    factor = -hx / h
    facsym = factor.free_symbols
    if y not in facsym:
        try:
            fx = exp(integrate(factor, x))
        except NotImplementedError:
            pass
        else:
            inf = {xi: fx, eta: S.Zero}
            if not comp:
                return [inf]
            if comp and inf not in xieta:
                xieta.append(inf)
    factor = -hx / h ** 2
    facsym = factor.free_symbols
    if x not in facsym:
        try:
            fy = exp(integrate(factor, y))
        except NotImplementedError:
            pass
        else:
            inf = {xi: fy.subs(y, func), eta: S.Zero}
            if not comp:
                return [inf]
            if comp and inf not in xieta:
                xieta.append(inf)
    if xieta:
        return xieta

def lie_heuristic_abaco1_product(match, comp=False):
    if False:
        i = 10
        return i + 15
    '\n    The second heuristic uses the following two assumptions on `\\xi` and `\\eta`\n\n    .. math:: \\eta = 0, \\xi = f(x)*g(y)\n\n    .. math:: \\eta = f(x)*g(y), \\xi = 0\n\n    The first assumption of this heuristic holds good if\n    `\\frac{1}{h^{2}}\\frac{\\partial^2}{\\partial x \\partial y}\\log(h)` is\n    separable in `x` and `y`, then the separated factors containing `x`\n    is `f(x)`, and `g(y)` is obtained by\n\n    .. math:: e^{\\int f\\frac{\\partial}{\\partial x}\\left(\\frac{1}{f*h}\\right)\\,dy}\n\n    provided `f\\frac{\\partial}{\\partial x}\\left(\\frac{1}{f*h}\\right)` is a function\n    of `y` only.\n\n    The second assumption holds good if `\\frac{dy}{dx} = h(x, y)` is rewritten as\n    `\\frac{dy}{dx} = \\frac{1}{h(y, x)}` and the same properties of the first assumption\n    satisfies. After obtaining `f(x)` and `g(y)`, the coordinates are again\n    interchanged, to get `\\eta` as `f(x)*g(y)`\n\n\n    References\n    ==========\n    - E.S. Cheb-Terrab, A.D. Roche, Symmetries and First Order\n      ODE Patterns, pp. 7 - pp. 8\n\n    '
    xieta = []
    y = match['y']
    h = match['h']
    hinv = match['hinv']
    func = match['func']
    x = func.args[0]
    xi = Function('xi')(x, func)
    eta = Function('eta')(x, func)
    inf = separatevars(log(h).diff(y).diff(x) / h ** 2, dict=True, symbols=[x, y])
    if inf and inf['coeff']:
        fx = inf[x]
        gy = simplify(fx * (1 / (fx * h)).diff(x))
        gysyms = gy.free_symbols
        if x not in gysyms:
            gy = exp(integrate(gy, y))
            inf = {eta: S.Zero, xi: (fx * gy).subs(y, func)}
            if not comp:
                return [inf]
            if comp and inf not in xieta:
                xieta.append(inf)
    u1 = Dummy('u1')
    inf = separatevars(log(hinv).diff(y).diff(x) / hinv ** 2, dict=True, symbols=[x, y])
    if inf and inf['coeff']:
        fx = inf[x]
        gy = simplify(fx * (1 / (fx * hinv)).diff(x))
        gysyms = gy.free_symbols
        if x not in gysyms:
            gy = exp(integrate(gy, y))
            etaval = fx * gy
            etaval = etaval.subs([(x, u1), (y, x)]).subs(u1, y)
            inf = {eta: etaval.subs(y, func), xi: S.Zero}
            if not comp:
                return [inf]
            if comp and inf not in xieta:
                xieta.append(inf)
    if xieta:
        return xieta

def lie_heuristic_bivariate(match, comp=False):
    if False:
        for i in range(10):
            print('nop')
    '\n    The third heuristic assumes the infinitesimals `\\xi` and `\\eta`\n    to be bi-variate polynomials in `x` and `y`. The assumption made here\n    for the logic below is that `h` is a rational function in `x` and `y`\n    though that may not be necessary for the infinitesimals to be\n    bivariate polynomials. The coefficients of the infinitesimals\n    are found out by substituting them in the PDE and grouping similar terms\n    that are polynomials and since they form a linear system, solve and check\n    for non trivial solutions. The degree of the assumed bivariates\n    are increased till a certain maximum value.\n\n    References\n    ==========\n    - Lie Groups and Differential Equations\n      pp. 327 - pp. 329\n\n    '
    h = match['h']
    hx = match['hx']
    hy = match['hy']
    func = match['func']
    x = func.args[0]
    y = match['y']
    xi = Function('xi')(x, func)
    eta = Function('eta')(x, func)
    if h.is_rational_function():
        (etax, etay, etad, xix, xiy, xid) = symbols('etax etay etad xix xiy xid')
        ipde = etax + (etay - xix) * h - xiy * h ** 2 - xid * hx - etad * hy
        (num, denom) = cancel(ipde).as_numer_denom()
        deg = Poly(num, x, y).total_degree()
        deta = Function('deta')(x, y)
        dxi = Function('dxi')(x, y)
        ipde = deta.diff(x) + (deta.diff(y) - dxi.diff(x)) * h - dxi.diff(y) * h ** 2 - dxi * hx - deta * hy
        xieq = Symbol('xi0')
        etaeq = Symbol('eta0')
        for i in range(deg + 1):
            if i:
                xieq += Add(*[Symbol('xi_' + str(power) + '_' + str(i - power)) * x ** power * y ** (i - power) for power in range(i + 1)])
                etaeq += Add(*[Symbol('eta_' + str(power) + '_' + str(i - power)) * x ** power * y ** (i - power) for power in range(i + 1)])
            (pden, denom) = ipde.subs({dxi: xieq, deta: etaeq}).doit().as_numer_denom()
            pden = expand(pden)
            if pden.is_polynomial(x, y) and pden.is_Add:
                polyy = Poly(pden, x, y).as_dict()
            if polyy:
                symset = xieq.free_symbols.union(etaeq.free_symbols) - {x, y}
                soldict = solve(polyy.values(), *symset)
                if isinstance(soldict, list):
                    soldict = soldict[0]
                if any(soldict.values()):
                    xired = xieq.subs(soldict)
                    etared = etaeq.subs(soldict)
                    dict_ = {sym: 1 for sym in symset}
                    inf = {eta: etared.subs(dict_).subs(y, func), xi: xired.subs(dict_).subs(y, func)}
                    return [inf]

def lie_heuristic_chi(match, comp=False):
    if False:
        while True:
            i = 10
    '\n    The aim of the fourth heuristic is to find the function `\\chi(x, y)`\n    that satisfies the PDE `\\frac{d\\chi}{dx} + h\\frac{d\\chi}{dx}\n    - \\frac{\\partial h}{\\partial y}\\chi = 0`.\n\n    This assumes `\\chi` to be a bivariate polynomial in `x` and `y`. By intuition,\n    `h` should be a rational function in `x` and `y`. The method used here is\n    to substitute a general binomial for `\\chi` up to a certain maximum degree\n    is reached. The coefficients of the polynomials, are calculated by by collecting\n    terms of the same order in `x` and `y`.\n\n    After finding `\\chi`, the next step is to use `\\eta = \\xi*h + \\chi`, to\n    determine `\\xi` and `\\eta`. This can be done by dividing `\\chi` by `h`\n    which would give `-\\xi` as the quotient and `\\eta` as the remainder.\n\n\n    References\n    ==========\n    - E.S Cheb-Terrab, L.G.S Duarte and L.A,C.P da Mota, Computer Algebra\n      Solving of First Order ODEs Using Symmetry Methods, pp. 8\n\n    '
    h = match['h']
    hy = match['hy']
    func = match['func']
    x = func.args[0]
    y = match['y']
    xi = Function('xi')(x, func)
    eta = Function('eta')(x, func)
    if h.is_rational_function():
        (schi, schix, schiy) = symbols('schi, schix, schiy')
        cpde = schix + h * schiy - hy * schi
        (num, denom) = cancel(cpde).as_numer_denom()
        deg = Poly(num, x, y).total_degree()
        chi = Function('chi')(x, y)
        chix = chi.diff(x)
        chiy = chi.diff(y)
        cpde = chix + h * chiy - hy * chi
        chieq = Symbol('chi')
        for i in range(1, deg + 1):
            chieq += Add(*[Symbol('chi_' + str(power) + '_' + str(i - power)) * x ** power * y ** (i - power) for power in range(i + 1)])
            (cnum, cden) = cancel(cpde.subs({chi: chieq}).doit()).as_numer_denom()
            cnum = expand(cnum)
            if cnum.is_polynomial(x, y) and cnum.is_Add:
                cpoly = Poly(cnum, x, y).as_dict()
                if cpoly:
                    solsyms = chieq.free_symbols - {x, y}
                    soldict = solve(cpoly.values(), *solsyms)
                    if isinstance(soldict, list):
                        soldict = soldict[0]
                    if any(soldict.values()):
                        chieq = chieq.subs(soldict)
                        dict_ = {sym: 1 for sym in solsyms}
                        chieq = chieq.subs(dict_)
                        (xic, etac) = div(chieq, h)
                        inf = {eta: etac.subs(y, func), xi: -xic.subs(y, func)}
                        return [inf]

def lie_heuristic_function_sum(match, comp=False):
    if False:
        while True:
            i = 10
    "\n    This heuristic uses the following two assumptions on `\\xi` and `\\eta`\n\n    .. math:: \\eta = 0, \\xi = f(x) + g(y)\n\n    .. math:: \\eta = f(x) + g(y), \\xi = 0\n\n    The first assumption of this heuristic holds good if\n\n    .. math:: \\frac{\\partial}{\\partial y}[(h\\frac{\\partial^{2}}{\n                \\partial x^{2}}(h^{-1}))^{-1}]\n\n    is separable in `x` and `y`,\n\n    1. The separated factors containing `y` is `\\frac{\\partial g}{\\partial y}`.\n       From this `g(y)` can be determined.\n    2. The separated factors containing `x` is `f''(x)`.\n    3. `h\\frac{\\partial^{2}}{\\partial x^{2}}(h^{-1})` equals\n       `\\frac{f''(x)}{f(x) + g(y)}`. From this `f(x)` can be determined.\n\n    The second assumption holds good if `\\frac{dy}{dx} = h(x, y)` is rewritten as\n    `\\frac{dy}{dx} = \\frac{1}{h(y, x)}` and the same properties of the first\n    assumption satisfies. After obtaining `f(x)` and `g(y)`, the coordinates\n    are again interchanged, to get `\\eta` as `f(x) + g(y)`.\n\n    For both assumptions, the constant factors are separated among `g(y)`\n    and `f''(x)`, such that `f''(x)` obtained from 3] is the same as that\n    obtained from 2]. If not possible, then this heuristic fails.\n\n\n    References\n    ==========\n    - E.S. Cheb-Terrab, A.D. Roche, Symmetries and First Order\n      ODE Patterns, pp. 7 - pp. 8\n\n    "
    xieta = []
    h = match['h']
    func = match['func']
    hinv = match['hinv']
    x = func.args[0]
    y = match['y']
    xi = Function('xi')(x, func)
    eta = Function('eta')(x, func)
    for odefac in [h, hinv]:
        factor = odefac * (1 / odefac).diff(x, 2)
        sep = separatevars((1 / factor).diff(y), dict=True, symbols=[x, y])
        if sep and sep['coeff'] and sep[x].has(x) and sep[y].has(y):
            k = Dummy('k')
            try:
                gy = k * integrate(sep[y], y)
            except NotImplementedError:
                pass
            else:
                fdd = 1 / (k * sep[x] * sep['coeff'])
                fx = simplify(fdd / factor - gy)
                check = simplify(fx.diff(x, 2) - fdd)
                if fx:
                    if not check:
                        fx = fx.subs(k, 1)
                        gy = gy / k
                    else:
                        sol = solve(check, k)
                        if sol:
                            sol = sol[0]
                            fx = fx.subs(k, sol)
                            gy = gy / k * sol
                        else:
                            continue
                    if odefac == hinv:
                        fx = fx.subs(x, y)
                        gy = gy.subs(y, x)
                    etaval = factor_terms(fx + gy)
                    if etaval.is_Mul:
                        etaval = Mul(*[arg for arg in etaval.args if arg.has(x, y)])
                    if odefac == hinv:
                        inf = {eta: etaval.subs(y, func), xi: S.Zero}
                    else:
                        inf = {xi: etaval.subs(y, func), eta: S.Zero}
                    if not comp:
                        return [inf]
                    else:
                        xieta.append(inf)
        if xieta:
            return xieta

def lie_heuristic_abaco2_similar(match, comp=False):
    if False:
        return 10
    "\n    This heuristic uses the following two assumptions on `\\xi` and `\\eta`\n\n    .. math:: \\eta = g(x), \\xi = f(x)\n\n    .. math:: \\eta = f(y), \\xi = g(y)\n\n    For the first assumption,\n\n    1. First `\\frac{\\frac{\\partial h}{\\partial y}}{\\frac{\\partial^{2} h}{\n       \\partial yy}}` is calculated. Let us say this value is A\n\n    2. If this is constant, then `h` is matched to the form `A(x) + B(x)e^{\n       \\frac{y}{C}}` then, `\\frac{e^{\\int \\frac{A(x)}{C} \\,dx}}{B(x)}` gives `f(x)`\n       and `A(x)*f(x)` gives `g(x)`\n\n    3. Otherwise `\\frac{\\frac{\\partial A}{\\partial X}}{\\frac{\\partial A}{\n       \\partial Y}} = \\gamma` is calculated. If\n\n       a] `\\gamma` is a function of `x` alone\n\n       b] `\\frac{\\gamma\\frac{\\partial h}{\\partial y} - \\gamma'(x) - \\frac{\n       \\partial h}{\\partial x}}{h + \\gamma} = G` is a function of `x` alone.\n       then, `e^{\\int G \\,dx}` gives `f(x)` and `-\\gamma*f(x)` gives `g(x)`\n\n    The second assumption holds good if `\\frac{dy}{dx} = h(x, y)` is rewritten as\n    `\\frac{dy}{dx} = \\frac{1}{h(y, x)}` and the same properties of the first assumption\n    satisfies. After obtaining `f(x)` and `g(x)`, the coordinates are again\n    interchanged, to get `\\xi` as `f(x^*)` and `\\eta` as `g(y^*)`\n\n    References\n    ==========\n    - E.S. Cheb-Terrab, A.D. Roche, Symmetries and First Order\n      ODE Patterns, pp. 10 - pp. 12\n\n    "
    h = match['h']
    hx = match['hx']
    hy = match['hy']
    func = match['func']
    hinv = match['hinv']
    x = func.args[0]
    y = match['y']
    xi = Function('xi')(x, func)
    eta = Function('eta')(x, func)
    factor = cancel(h.diff(y) / h.diff(y, 2))
    factorx = factor.diff(x)
    factory = factor.diff(y)
    if not factor.has(x) and (not factor.has(y)):
        A = Wild('A', exclude=[y])
        B = Wild('B', exclude=[y])
        C = Wild('C', exclude=[x, y])
        match = h.match(A + B * exp(y / C))
        try:
            tau = exp(-integrate(match[A] / match[C]), x) / match[B]
        except NotImplementedError:
            pass
        else:
            gx = match[A] * tau
            return [{xi: tau, eta: gx}]
    else:
        gamma = cancel(factorx / factory)
        if not gamma.has(y):
            tauint = cancel((gamma * hy - gamma.diff(x) - hx) / (h + gamma))
            if not tauint.has(y):
                try:
                    tau = exp(integrate(tauint, x))
                except NotImplementedError:
                    pass
                else:
                    gx = -tau * gamma
                    return [{xi: tau, eta: gx}]
    factor = cancel(hinv.diff(y) / hinv.diff(y, 2))
    factorx = factor.diff(x)
    factory = factor.diff(y)
    if not factor.has(x) and (not factor.has(y)):
        A = Wild('A', exclude=[y])
        B = Wild('B', exclude=[y])
        C = Wild('C', exclude=[x, y])
        match = h.match(A + B * exp(y / C))
        try:
            tau = exp(-integrate(match[A] / match[C]), x) / match[B]
        except NotImplementedError:
            pass
        else:
            gx = match[A] * tau
            return [{eta: tau.subs(x, func), xi: gx.subs(x, func)}]
    else:
        gamma = cancel(factorx / factory)
        if not gamma.has(y):
            tauint = cancel((gamma * hinv.diff(y) - gamma.diff(x) - hinv.diff(x)) / (hinv + gamma))
            if not tauint.has(y):
                try:
                    tau = exp(integrate(tauint, x))
                except NotImplementedError:
                    pass
                else:
                    gx = -tau * gamma
                    return [{eta: tau.subs(x, func), xi: gx.subs(x, func)}]

def lie_heuristic_abaco2_unique_unknown(match, comp=False):
    if False:
        for i in range(10):
            print('nop')
    '\n    This heuristic assumes the presence of unknown functions or known functions\n    with non-integer powers.\n\n    1. A list of all functions and non-integer powers containing x and y\n    2. Loop over each element `f` in the list, find `\\frac{\\frac{\\partial f}{\\partial x}}{\n       \\frac{\\partial f}{\\partial x}} = R`\n\n       If it is separable in `x` and `y`, let `X` be the factors containing `x`. Then\n\n       a] Check if `\\xi = X` and `\\eta = -\\frac{X}{R}` satisfy the PDE. If yes, then return\n          `\\xi` and `\\eta`\n       b] Check if `\\xi = \\frac{-R}{X}` and `\\eta = -\\frac{1}{X}` satisfy the PDE.\n           If yes, then return `\\xi` and `\\eta`\n\n       If not, then check if\n\n       a] :math:`\\xi = -R,\\eta = 1`\n\n       b] :math:`\\xi = 1, \\eta = -\\frac{1}{R}`\n\n       are solutions.\n\n    References\n    ==========\n    - E.S. Cheb-Terrab, A.D. Roche, Symmetries and First Order\n      ODE Patterns, pp. 10 - pp. 12\n\n    '
    h = match['h']
    hx = match['hx']
    hy = match['hy']
    func = match['func']
    x = func.args[0]
    y = match['y']
    xi = Function('xi')(x, func)
    eta = Function('eta')(x, func)
    funclist = []
    for atom in h.atoms(Pow):
        (base, exp) = atom.as_base_exp()
        if base.has(x) and base.has(y):
            if not exp.is_Integer:
                funclist.append(atom)
    for function in h.atoms(AppliedUndef):
        syms = function.free_symbols
        if x in syms and y in syms:
            funclist.append(function)
    for f in funclist:
        frac = cancel(f.diff(y) / f.diff(x))
        sep = separatevars(frac, dict=True, symbols=[x, y])
        if sep and sep['coeff']:
            xitry1 = sep[x]
            etatry1 = -1 / (sep[y] * sep['coeff'])
            pde1 = etatry1.diff(y) * h - xitry1.diff(x) * h - xitry1 * hx - etatry1 * hy
            if not simplify(pde1):
                return [{xi: xitry1, eta: etatry1.subs(y, func)}]
            xitry2 = 1 / etatry1
            etatry2 = 1 / xitry1
            pde2 = etatry2.diff(x) - xitry2.diff(y) * h ** 2 - xitry2 * hx - etatry2 * hy
            if not simplify(expand(pde2)):
                return [{xi: xitry2.subs(y, func), eta: etatry2}]
        else:
            etatry = -1 / frac
            pde = etatry.diff(x) + etatry.diff(y) * h - hx - etatry * hy
            if not simplify(pde):
                return [{xi: S.One, eta: etatry.subs(y, func)}]
            xitry = -frac
            pde = -xitry.diff(x) * h - xitry.diff(y) * h ** 2 - xitry * hx - hy
            if not simplify(expand(pde)):
                return [{xi: xitry.subs(y, func), eta: S.One}]

def lie_heuristic_abaco2_unique_general(match, comp=False):
    if False:
        print('Hello World!')
    '\n    This heuristic finds if infinitesimals of the form `\\eta = f(x)`, `\\xi = g(y)`\n    without making any assumptions on `h`.\n\n    The complete sequence of steps is given in the paper mentioned below.\n\n    References\n    ==========\n    - E.S. Cheb-Terrab, A.D. Roche, Symmetries and First Order\n      ODE Patterns, pp. 10 - pp. 12\n\n    '
    hx = match['hx']
    hy = match['hy']
    func = match['func']
    x = func.args[0]
    y = match['y']
    xi = Function('xi')(x, func)
    eta = Function('eta')(x, func)
    A = hx.diff(y)
    B = hy.diff(y) + hy ** 2
    C = hx.diff(x) - hx ** 2
    if not (A and B and C):
        return
    Ax = A.diff(x)
    Ay = A.diff(y)
    Axy = Ax.diff(y)
    Axx = Ax.diff(x)
    Ayy = Ay.diff(y)
    D = simplify(2 * Axy + hx * Ay - Ax * hy + (hx * hy + 2 * A) * A) * A - 3 * Ax * Ay
    if not D:
        E1 = simplify(3 * Ax ** 2 + ((hx ** 2 + 2 * C) * A - 2 * Axx) * A)
        if E1:
            E2 = simplify((2 * Ayy + (2 * B - hy ** 2) * A) * A - 3 * Ay ** 2)
            if not E2:
                E3 = simplify(E1 * ((28 * Ax + 4 * hx * A) * A ** 3 - E1 * (hy * A + Ay)) - E1.diff(x) * 8 * A ** 4)
                if not E3:
                    etaval = cancel((4 * A ** 3 * (Ax - hx * A) + E1 * (hy * A - Ay)) / (S(2) * A * E1))
                    if x not in etaval:
                        try:
                            etaval = exp(integrate(etaval, y))
                        except NotImplementedError:
                            pass
                        else:
                            xival = -4 * A ** 3 * etaval / E1
                            if y not in xival:
                                return [{xi: xival, eta: etaval.subs(y, func)}]
    else:
        E1 = simplify((2 * Ayy + (2 * B - hy ** 2) * A) * A - 3 * Ay ** 2)
        if E1:
            E2 = simplify(4 * A ** 3 * D - D ** 2 + E1 * ((2 * Axx - (hx ** 2 + 2 * C) * A) * A - 3 * Ax ** 2))
            if not E2:
                E3 = simplify(-(A * D) * E1.diff(y) + ((E1.diff(x) - hy * D) * A + 3 * Ay * D + (A * hx - 3 * Ax) * E1) * E1)
                if not E3:
                    etaval = cancel(((A * hx - Ax) * E1 - (Ay + A * hy) * D) / (S(2) * A * D))
                    if x not in etaval:
                        try:
                            etaval = exp(integrate(etaval, y))
                        except NotImplementedError:
                            pass
                        else:
                            xival = -E1 * etaval / D
                            if y not in xival:
                                return [{xi: xival, eta: etaval.subs(y, func)}]

def lie_heuristic_linear(match, comp=False):
    if False:
        return 10
    '\n    This heuristic assumes\n\n    1. `\\xi = ax + by + c` and\n    2. `\\eta = fx + gy + h`\n\n    After substituting the following assumptions in the determining PDE, it\n    reduces to\n\n    .. math:: f + (g - a)h - bh^{2} - (ax + by + c)\\frac{\\partial h}{\\partial x}\n                 - (fx + gy + c)\\frac{\\partial h}{\\partial y}\n\n    Solving the reduced PDE obtained, using the method of characteristics, becomes\n    impractical. The method followed is grouping similar terms and solving the system\n    of linear equations obtained. The difference between the bivariate heuristic is that\n    `h` need not be a rational function in this case.\n\n    References\n    ==========\n    - E.S. Cheb-Terrab, A.D. Roche, Symmetries and First Order\n      ODE Patterns, pp. 10 - pp. 12\n\n    '
    h = match['h']
    hx = match['hx']
    hy = match['hy']
    func = match['func']
    x = func.args[0]
    y = match['y']
    xi = Function('xi')(x, func)
    eta = Function('eta')(x, func)
    coeffdict = {}
    symbols = numbered_symbols('c', cls=Dummy)
    symlist = [next(symbols) for _ in islice(symbols, 6)]
    (C0, C1, C2, C3, C4, C5) = symlist
    pde = C3 + (C4 - C0) * h - (C0 * x + C1 * y + C2) * hx - (C3 * x + C4 * y + C5) * hy - C1 * h ** 2
    (pde, denom) = pde.as_numer_denom()
    pde = powsimp(expand(pde))
    if pde.is_Add:
        terms = pde.args
        for term in terms:
            if term.is_Mul:
                rem = Mul(*[m for m in term.args if not m.has(x, y)])
                xypart = term / rem
                if xypart not in coeffdict:
                    coeffdict[xypart] = rem
                else:
                    coeffdict[xypart] += rem
            elif term not in coeffdict:
                coeffdict[term] = S.One
            else:
                coeffdict[term] += S.One
    sollist = coeffdict.values()
    soldict = solve(sollist, symlist)
    if soldict:
        if isinstance(soldict, list):
            soldict = soldict[0]
        subval = soldict.values()
        if any((t for t in subval)):
            onedict = dict(zip(symlist, [1] * 6))
            xival = C0 * x + C1 * func + C2
            etaval = C3 * x + C4 * func + C5
            xival = xival.subs(soldict)
            etaval = etaval.subs(soldict)
            xival = xival.subs(onedict)
            etaval = etaval.subs(onedict)
            return [{xi: xival, eta: etaval}]

def _lie_group_remove(coords):
    if False:
        for i in range(10):
            print('nop')
    '\n    This function is strictly meant for internal use by the Lie group ODE solving\n    method. It replaces arbitrary functions returned by pdsolve as follows:\n\n    1] If coords is an arbitrary function, then its argument is returned.\n    2] An arbitrary function in an Add object is replaced by zero.\n    3] An arbitrary function in a Mul object is replaced by one.\n    4] If there is no arbitrary function coords is returned unchanged.\n\n    Examples\n    ========\n\n    >>> from sympy.solvers.ode.lie_group import _lie_group_remove\n    >>> from sympy import Function\n    >>> from sympy.abc import x, y\n    >>> F = Function("F")\n    >>> eq = x**2*y\n    >>> _lie_group_remove(eq)\n    x**2*y\n    >>> eq = F(x**2*y)\n    >>> _lie_group_remove(eq)\n    x**2*y\n    >>> eq = x*y**2 + F(x**3)\n    >>> _lie_group_remove(eq)\n    x*y**2\n    >>> eq = (F(x**3) + y)*x**4\n    >>> _lie_group_remove(eq)\n    x**4*y\n\n    '
    if isinstance(coords, AppliedUndef):
        return coords.args[0]
    elif coords.is_Add:
        subfunc = coords.atoms(AppliedUndef)
        if subfunc:
            for func in subfunc:
                coords = coords.subs(func, 0)
        return coords
    elif coords.is_Pow:
        (base, expr) = coords.as_base_exp()
        base = _lie_group_remove(base)
        expr = _lie_group_remove(expr)
        return base ** expr
    elif coords.is_Mul:
        mulargs = []
        coordargs = coords.args
        for arg in coordargs:
            if not isinstance(coords, AppliedUndef):
                mulargs.append(_lie_group_remove(arg))
        return Mul(*mulargs)
    return coords