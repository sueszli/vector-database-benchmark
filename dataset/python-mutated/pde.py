"""
This module contains pdsolve() and different helper functions that it
uses. It is heavily inspired by the ode module and hence the basic
infrastructure remains the same.

**Functions in this module**

    These are the user functions in this module:

    - pdsolve()     - Solves PDE's
    - classify_pde() - Classifies PDEs into possible hints for dsolve().
    - pde_separate() - Separate variables in partial differential equation either by
                       additive or multiplicative separation approach.

    These are the helper functions in this module:

    - pde_separate_add() - Helper function for searching additive separable solutions.
    - pde_separate_mul() - Helper function for searching multiplicative
                           separable solutions.

**Currently implemented solver methods**

The following methods are implemented for solving partial differential
equations.  See the docstrings of the various pde_hint() functions for
more information on each (run help(pde)):

  - 1st order linear homogeneous partial differential equations
    with constant coefficients.
  - 1st order linear general partial differential equations
    with constant coefficients.
  - 1st order linear partial differential equations with
    variable coefficients.

"""
from functools import reduce
from itertools import combinations_with_replacement
from sympy.simplify import simplify
from sympy.core import Add, S
from sympy.core.function import Function, expand, AppliedUndef, Subs
from sympy.core.relational import Equality, Eq
from sympy.core.symbol import Symbol, Wild, symbols
from sympy.functions import exp
from sympy.integrals.integrals import Integral, integrate
from sympy.utilities.iterables import has_dups, is_sequence
from sympy.utilities.misc import filldedent
from sympy.solvers.deutils import _preprocess, ode_order, _desolve
from sympy.solvers.solvers import solve
from sympy.simplify.radsimp import collect
import operator
allhints = ('1st_linear_constant_coeff_homogeneous', '1st_linear_constant_coeff', '1st_linear_constant_coeff_Integral', '1st_linear_variable_coeff')

def pdsolve(eq, func=None, hint='default', dict=False, solvefun=None, **kwargs):
    if False:
        return 10
    '\n    Solves any (supported) kind of partial differential equation.\n\n    **Usage**\n\n        pdsolve(eq, f(x,y), hint) -> Solve partial differential equation\n        eq for function f(x,y), using method hint.\n\n    **Details**\n\n        ``eq`` can be any supported partial differential equation (see\n            the pde docstring for supported methods).  This can either\n            be an Equality, or an expression, which is assumed to be\n            equal to 0.\n\n        ``f(x,y)`` is a function of two variables whose derivatives in that\n            variable make up the partial differential equation. In many\n            cases it is not necessary to provide this; it will be autodetected\n            (and an error raised if it could not be detected).\n\n        ``hint`` is the solving method that you want pdsolve to use.  Use\n            classify_pde(eq, f(x,y)) to get all of the possible hints for\n            a PDE.  The default hint, \'default\', will use whatever hint\n            is returned first by classify_pde().  See Hints below for\n            more options that you can use for hint.\n\n        ``solvefun`` is the convention used for arbitrary functions returned\n            by the PDE solver. If not set by the user, it is set by default\n            to be F.\n\n    **Hints**\n\n        Aside from the various solving methods, there are also some\n        meta-hints that you can pass to pdsolve():\n\n        "default":\n                This uses whatever hint is returned first by\n                classify_pde(). This is the default argument to\n                pdsolve().\n\n        "all":\n                To make pdsolve apply all relevant classification hints,\n                use pdsolve(PDE, func, hint="all").  This will return a\n                dictionary of hint:solution terms.  If a hint causes\n                pdsolve to raise the NotImplementedError, value of that\n                hint\'s key will be the exception object raised.  The\n                dictionary will also include some special keys:\n\n                - order: The order of the PDE.  See also ode_order() in\n                  deutils.py\n                - default: The solution that would be returned by\n                  default.  This is the one produced by the hint that\n                  appears first in the tuple returned by classify_pde().\n\n        "all_Integral":\n                This is the same as "all", except if a hint also has a\n                corresponding "_Integral" hint, it only returns the\n                "_Integral" hint.  This is useful if "all" causes\n                pdsolve() to hang because of a difficult or impossible\n                integral.  This meta-hint will also be much faster than\n                "all", because integrate() is an expensive routine.\n\n        See also the classify_pde() docstring for more info on hints,\n        and the pde docstring for a list of all supported hints.\n\n    **Tips**\n        - You can declare the derivative of an unknown function this way:\n\n            >>> from sympy import Function, Derivative\n            >>> from sympy.abc import x, y # x and y are the independent variables\n            >>> f = Function("f")(x, y) # f is a function of x and y\n            >>> # fx will be the partial derivative of f with respect to x\n            >>> fx = Derivative(f, x)\n            >>> # fy will be the partial derivative of f with respect to y\n            >>> fy = Derivative(f, y)\n\n        - See test_pde.py for many tests, which serves also as a set of\n          examples for how to use pdsolve().\n        - pdsolve always returns an Equality class (except for the case\n          when the hint is "all" or "all_Integral"). Note that it is not possible\n          to get an explicit solution for f(x, y) as in the case of ODE\'s\n        - Do help(pde.pde_hintname) to get help more information on a\n          specific hint\n\n\n    Examples\n    ========\n\n    >>> from sympy.solvers.pde import pdsolve\n    >>> from sympy import Function, Eq\n    >>> from sympy.abc import x, y\n    >>> f = Function(\'f\')\n    >>> u = f(x, y)\n    >>> ux = u.diff(x)\n    >>> uy = u.diff(y)\n    >>> eq = Eq(1 + (2*(ux/u)) + (3*(uy/u)), 0)\n    >>> pdsolve(eq)\n    Eq(f(x, y), F(3*x - 2*y)*exp(-2*x/13 - 3*y/13))\n\n    '
    if not solvefun:
        solvefun = Function('F')
    hints = _desolve(eq, func=func, hint=hint, simplify=True, type='pde', **kwargs)
    eq = hints.pop('eq', False)
    all_ = hints.pop('all', False)
    if all_:
        pdedict = {}
        failed_hints = {}
        gethints = classify_pde(eq, dict=True)
        pdedict.update({'order': gethints['order'], 'default': gethints['default']})
        for hint in hints:
            try:
                rv = _helper_simplify(eq, hint, hints[hint]['func'], hints[hint]['order'], hints[hint][hint], solvefun)
            except NotImplementedError as detail:
                failed_hints[hint] = detail
            else:
                pdedict[hint] = rv
        pdedict.update(failed_hints)
        return pdedict
    else:
        return _helper_simplify(eq, hints['hint'], hints['func'], hints['order'], hints[hints['hint']], solvefun)

def _helper_simplify(eq, hint, func, order, match, solvefun):
    if False:
        while True:
            i = 10
    'Helper function of pdsolve that calls the respective\n    pde functions to solve for the partial differential\n    equations. This minimizes the computation in\n    calling _desolve multiple times.\n    '
    if hint.endswith('_Integral'):
        solvefunc = globals()['pde_' + hint[:-len('_Integral')]]
    else:
        solvefunc = globals()['pde_' + hint]
    return _handle_Integral(solvefunc(eq, func, order, match, solvefun), func, order, hint)

def _handle_Integral(expr, func, order, hint):
    if False:
        return 10
    '\n    Converts a solution with integrals in it into an actual solution.\n\n    Simplifies the integral mainly using doit()\n    '
    if hint.endswith('_Integral'):
        return expr
    elif hint == '1st_linear_constant_coeff':
        return simplify(expr.doit())
    else:
        return expr

def classify_pde(eq, func=None, dict=False, *, prep=True, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns a tuple of possible pdsolve() classifications for a PDE.\n\n    The tuple is ordered so that first item is the classification that\n    pdsolve() uses to solve the PDE by default.  In general,\n    classifications near the beginning of the list will produce\n    better solutions faster than those near the end, though there are\n    always exceptions.  To make pdsolve use a different classification,\n    use pdsolve(PDE, func, hint=<classification>).  See also the pdsolve()\n    docstring for different meta-hints you can use.\n\n    If ``dict`` is true, classify_pde() will return a dictionary of\n    hint:match expression terms. This is intended for internal use by\n    pdsolve().  Note that because dictionaries are ordered arbitrarily,\n    this will most likely not be in the same order as the tuple.\n\n    You can get help on different hints by doing help(pde.pde_hintname),\n    where hintname is the name of the hint without "_Integral".\n\n    See sympy.pde.allhints or the sympy.pde docstring for a list of all\n    supported hints that can be returned from classify_pde.\n\n\n    Examples\n    ========\n\n    >>> from sympy.solvers.pde import classify_pde\n    >>> from sympy import Function, Eq\n    >>> from sympy.abc import x, y\n    >>> f = Function(\'f\')\n    >>> u = f(x, y)\n    >>> ux = u.diff(x)\n    >>> uy = u.diff(y)\n    >>> eq = Eq(1 + (2*(ux/u)) + (3*(uy/u)), 0)\n    >>> classify_pde(eq)\n    (\'1st_linear_constant_coeff_homogeneous\',)\n    '
    if func and len(func.args) != 2:
        raise NotImplementedError('Right now only partial differential equations of two variables are supported')
    if prep or func is None:
        (prep, func_) = _preprocess(eq, func)
        if func is None:
            func = func_
    if isinstance(eq, Equality):
        if eq.rhs != 0:
            return classify_pde(eq.lhs - eq.rhs, func)
        eq = eq.lhs
    f = func.func
    x = func.args[0]
    y = func.args[1]
    fx = f(x, y).diff(x)
    fy = f(x, y).diff(y)
    order = ode_order(eq, f(x, y))
    matching_hints = {'order': order}
    if not order:
        if dict:
            matching_hints['default'] = None
            return matching_hints
        else:
            return ()
    eq = expand(eq)
    a = Wild('a', exclude=[f(x, y)])
    b = Wild('b', exclude=[f(x, y), fx, fy, x, y])
    c = Wild('c', exclude=[f(x, y), fx, fy, x, y])
    d = Wild('d', exclude=[f(x, y), fx, fy, x, y])
    e = Wild('e', exclude=[f(x, y), fx, fy])
    n = Wild('n', exclude=[x, y])
    reduced_eq = None
    if eq.is_Add:
        var = set(combinations_with_replacement((x, y), order))
        dummyvar = var.copy()
        power = None
        for i in var:
            coeff = eq.coeff(f(x, y).diff(*i))
            if coeff != 1:
                match = coeff.match(a * f(x, y) ** n)
                if match and match[a]:
                    power = match[n]
                    dummyvar.remove(i)
                    break
            dummyvar.remove(i)
        for i in dummyvar:
            coeff = eq.coeff(f(x, y).diff(*i))
            if coeff != 1:
                match = coeff.match(a * f(x, y) ** n)
                if match and match[a] and (match[n] < power):
                    power = match[n]
        if power:
            den = f(x, y) ** power
            reduced_eq = Add(*[arg / den for arg in eq.args])
    if not reduced_eq:
        reduced_eq = eq
    if order == 1:
        reduced_eq = collect(reduced_eq, f(x, y))
        r = reduced_eq.match(b * fx + c * fy + d * f(x, y) + e)
        if r:
            if not r[e]:
                r.update({'b': b, 'c': c, 'd': d})
                matching_hints['1st_linear_constant_coeff_homogeneous'] = r
            elif r[b] ** 2 + r[c] ** 2 != 0:
                r.update({'b': b, 'c': c, 'd': d, 'e': e})
                matching_hints['1st_linear_constant_coeff'] = r
                matching_hints['1st_linear_constant_coeff_Integral'] = r
        else:
            b = Wild('b', exclude=[f(x, y), fx, fy])
            c = Wild('c', exclude=[f(x, y), fx, fy])
            d = Wild('d', exclude=[f(x, y), fx, fy])
            r = reduced_eq.match(b * fx + c * fy + d * f(x, y) + e)
            if r:
                r.update({'b': b, 'c': c, 'd': d, 'e': e})
                matching_hints['1st_linear_variable_coeff'] = r
    retlist = [i for i in allhints if i in matching_hints]
    if dict:
        matching_hints['default'] = None
        matching_hints['ordered_hints'] = tuple(retlist)
        for i in allhints:
            if i in matching_hints:
                matching_hints['default'] = i
                break
        return matching_hints
    else:
        return tuple(retlist)

def checkpdesol(pde, sol, func=None, solve_for_func=True):
    if False:
        print('Hello World!')
    "\n    Checks if the given solution satisfies the partial differential\n    equation.\n\n    pde is the partial differential equation which can be given in the\n    form of an equation or an expression. sol is the solution for which\n    the pde is to be checked. This can also be given in an equation or\n    an expression form. If the function is not provided, the helper\n    function _preprocess from deutils is used to identify the function.\n\n    If a sequence of solutions is passed, the same sort of container will be\n    used to return the result for each solution.\n\n    The following methods are currently being implemented to check if the\n    solution satisfies the PDE:\n\n        1. Directly substitute the solution in the PDE and check. If the\n           solution has not been solved for f, then it will solve for f\n           provided solve_for_func has not been set to False.\n\n    If the solution satisfies the PDE, then a tuple (True, 0) is returned.\n    Otherwise a tuple (False, expr) where expr is the value obtained\n    after substituting the solution in the PDE. However if a known solution\n    returns False, it may be due to the inability of doit() to simplify it to zero.\n\n    Examples\n    ========\n\n    >>> from sympy import Function, symbols\n    >>> from sympy.solvers.pde import checkpdesol, pdsolve\n    >>> x, y = symbols('x y')\n    >>> f = Function('f')\n    >>> eq = 2*f(x,y) + 3*f(x,y).diff(x) + 4*f(x,y).diff(y)\n    >>> sol = pdsolve(eq)\n    >>> assert checkpdesol(eq, sol)[0]\n    >>> eq = x*f(x,y) + f(x,y).diff(x)\n    >>> checkpdesol(eq, sol)\n    (False, (x*F(4*x - 3*y) - 6*F(4*x - 3*y)/25 + 4*Subs(Derivative(F(_xi_1), _xi_1), _xi_1, 4*x - 3*y))*exp(-6*x/25 - 8*y/25))\n    "
    if not isinstance(pde, Equality):
        pde = Eq(pde, 0)
    if func is None:
        try:
            (_, func) = _preprocess(pde.lhs)
        except ValueError:
            funcs = [s.atoms(AppliedUndef) for s in (sol if is_sequence(sol, set) else [sol])]
            funcs = set().union(funcs)
            if len(funcs) != 1:
                raise ValueError('must pass func arg to checkpdesol for this case.')
            func = funcs.pop()
    if is_sequence(sol, set):
        return type(sol)([checkpdesol(pde, i, func=func, solve_for_func=solve_for_func) for i in sol])
    if not isinstance(sol, Equality):
        sol = Eq(func, sol)
    elif sol.rhs == func:
        sol = sol.reversed
    solved = sol.lhs == func and (not sol.rhs.has(func))
    if solve_for_func and (not solved):
        solved = solve(sol, func)
        if solved:
            if len(solved) == 1:
                return checkpdesol(pde, Eq(func, solved[0]), func=func, solve_for_func=False)
            else:
                return checkpdesol(pde, [Eq(func, t) for t in solved], func=func, solve_for_func=False)
    if sol.lhs == func:
        pde = pde.lhs - pde.rhs
        s = simplify(pde.subs(func, sol.rhs).doit())
        return (s is S.Zero, s)
    raise NotImplementedError(filldedent('\n        Unable to test if %s is a solution to %s.' % (sol, pde)))

def pde_1st_linear_constant_coeff_homogeneous(eq, func, order, match, solvefun):
    if False:
        print('Hello World!')
    '\n    Solves a first order linear homogeneous\n    partial differential equation with constant coefficients.\n\n    The general form of this partial differential equation is\n\n    .. math:: a \\frac{\\partial f(x,y)}{\\partial x}\n              + b \\frac{\\partial f(x,y)}{\\partial y} + c f(x,y) = 0\n\n    where `a`, `b` and `c` are constants.\n\n    The general solution is of the form:\n\n    .. math::\n        f(x, y) = F(- a y + b x ) e^{- \\frac{c (a x + b y)}{a^2 + b^2}}\n\n    and can be found in SymPy with ``pdsolve``::\n\n        >>> from sympy.solvers import pdsolve\n        >>> from sympy.abc import x, y, a, b, c\n        >>> from sympy import Function, pprint\n        >>> f = Function(\'f\')\n        >>> u = f(x,y)\n        >>> ux = u.diff(x)\n        >>> uy = u.diff(y)\n        >>> genform = a*ux + b*uy + c*u\n        >>> pprint(genform)\n          d               d\n        a*--(f(x, y)) + b*--(f(x, y)) + c*f(x, y)\n          dx              dy\n\n        >>> pprint(pdsolve(genform))\n                                 -c*(a*x + b*y)\n                                 ---------------\n                                      2    2\n                                     a  + b\n        f(x, y) = F(-a*y + b*x)*e\n\n    Examples\n    ========\n\n    >>> from sympy import pdsolve\n    >>> from sympy import Function, pprint\n    >>> from sympy.abc import x,y\n    >>> f = Function(\'f\')\n    >>> pdsolve(f(x,y) + f(x,y).diff(x) + f(x,y).diff(y))\n    Eq(f(x, y), F(x - y)*exp(-x/2 - y/2))\n    >>> pprint(pdsolve(f(x,y) + f(x,y).diff(x) + f(x,y).diff(y)))\n                          x   y\n                        - - - -\n                          2   2\n    f(x, y) = F(x - y)*e\n\n    References\n    ==========\n\n    - Viktor Grigoryan, "Partial Differential Equations"\n      Math 124A - Fall 2010, pp.7\n\n    '
    f = func.func
    x = func.args[0]
    y = func.args[1]
    b = match[match['b']]
    c = match[match['c']]
    d = match[match['d']]
    return Eq(f(x, y), exp(-S(d) / (b ** 2 + c ** 2) * (b * x + c * y)) * solvefun(c * x - b * y))

def pde_1st_linear_constant_coeff(eq, func, order, match, solvefun):
    if False:
        for i in range(10):
            print('nop')
    '\n    Solves a first order linear partial differential equation\n    with constant coefficients.\n\n    The general form of this partial differential equation is\n\n    .. math:: a \\frac{\\partial f(x,y)}{\\partial x}\n              + b \\frac{\\partial f(x,y)}{\\partial y}\n              + c f(x,y) = G(x,y)\n\n    where `a`, `b` and `c` are constants and `G(x, y)` can be an arbitrary\n    function in `x` and `y`.\n\n    The general solution of the PDE is:\n\n    .. math::\n        f(x, y) = \\left. \\left[F(\\eta) + \\frac{1}{a^2 + b^2}\n        \\int\\limits^{a x + b y} G\\left(\\frac{a \\xi + b \\eta}{a^2 + b^2},\n        \\frac{- a \\eta + b \\xi}{a^2 + b^2} \\right)\n        e^{\\frac{c \\xi}{a^2 + b^2}}\\, d\\xi\\right]\n        e^{- \\frac{c \\xi}{a^2 + b^2}}\n        \\right|_{\\substack{\\eta=- a y + b x\\\\ \\xi=a x + b y }}\\, ,\n\n    where `F(\\eta)` is an arbitrary single-valued function. The solution\n    can be found in SymPy with ``pdsolve``::\n\n        >>> from sympy.solvers import pdsolve\n        >>> from sympy.abc import x, y, a, b, c\n        >>> from sympy import Function, pprint\n        >>> f = Function(\'f\')\n        >>> G = Function(\'G\')\n        >>> u = f(x, y)\n        >>> ux = u.diff(x)\n        >>> uy = u.diff(y)\n        >>> genform = a*ux + b*uy + c*u - G(x,y)\n        >>> pprint(genform)\n          d               d\n        a*--(f(x, y)) + b*--(f(x, y)) + c*f(x, y) - G(x, y)\n          dx              dy\n        >>> pprint(pdsolve(genform, hint=\'1st_linear_constant_coeff_Integral\'))\n                  //          a*x + b*y                                             \\  >\n                  ||              /                                                 |  >\n                  ||             |                                                  |  >\n                  ||             |                                      c*xi        |  >\n                  ||             |                                     -------      |  >\n                  ||             |                                      2    2      |  >\n                  ||             |      /a*xi + b*eta  -a*eta + b*xi\\  a  + b       |  >\n                  ||             |     G|------------, -------------|*e        d(xi)|  >\n                  ||             |      |   2    2         2    2   |               |  >\n                  ||             |      \\  a  + b         a  + b    /               |  >\n                  ||             |                                                  |  >\n                  ||            /                                                   |  >\n                  ||                                                                |  >\n        f(x, y) = ||F(eta) + -------------------------------------------------------|* >\n                  ||                                  2    2                        |  >\n                  \\\\                                 a  + b                         /  >\n        <BLANKLINE>\n        >         \\|\n        >         ||\n        >         ||\n        >         ||\n        >         ||\n        >         ||\n        >         ||\n        >         ||\n        >         ||\n        >  -c*xi  ||\n        >  -------||\n        >   2    2||\n        >  a  + b ||\n        > e       ||\n        >         ||\n        >         /|eta=-a*y + b*x, xi=a*x + b*y\n\n    Examples\n    ========\n\n    >>> from sympy.solvers.pde import pdsolve\n    >>> from sympy import Function, pprint, exp\n    >>> from sympy.abc import x,y\n    >>> f = Function(\'f\')\n    >>> eq = -2*f(x,y).diff(x) + 4*f(x,y).diff(y) + 5*f(x,y) - exp(x + 3*y)\n    >>> pdsolve(eq)\n    Eq(f(x, y), (F(4*x + 2*y)*exp(x/2) + exp(x + 4*y)/15)*exp(-y))\n\n    References\n    ==========\n\n    - Viktor Grigoryan, "Partial Differential Equations"\n      Math 124A - Fall 2010, pp.7\n\n    '
    (xi, eta) = symbols('xi eta')
    f = func.func
    x = func.args[0]
    y = func.args[1]
    b = match[match['b']]
    c = match[match['c']]
    d = match[match['d']]
    e = -match[match['e']]
    expterm = exp(-S(d) / (b ** 2 + c ** 2) * xi)
    functerm = solvefun(eta)
    solvedict = solve((b * x + c * y - xi, c * x - b * y - eta), x, y)
    genterm = 1 / S(b ** 2 + c ** 2) * Integral((1 / expterm * e).subs(solvedict), (xi, b * x + c * y))
    return Eq(f(x, y), Subs(expterm * (functerm + genterm), (eta, xi), (c * x - b * y, b * x + c * y)))

def pde_1st_linear_variable_coeff(eq, func, order, match, solvefun):
    if False:
        i = 10
        return i + 15
    '\n    Solves a first order linear partial differential equation\n    with variable coefficients. The general form of this partial\n    differential equation is\n\n    .. math:: a(x, y) \\frac{\\partial f(x, y)}{\\partial x}\n                + b(x, y) \\frac{\\partial f(x, y)}{\\partial y}\n                + c(x, y) f(x, y) = G(x, y)\n\n    where `a(x, y)`, `b(x, y)`, `c(x, y)` and `G(x, y)` are arbitrary\n    functions in `x` and `y`. This PDE is converted into an ODE by\n    making the following transformation:\n\n    1. `\\xi` as `x`\n\n    2. `\\eta` as the constant in the solution to the differential\n       equation `\\frac{dy}{dx} = -\\frac{b}{a}`\n\n    Making the previous substitutions reduces it to the linear ODE\n\n    .. math:: a(\\xi, \\eta)\\frac{du}{d\\xi} + c(\\xi, \\eta)u - G(\\xi, \\eta) = 0\n\n    which can be solved using ``dsolve``.\n\n    >>> from sympy.abc import x, y\n    >>> from sympy import Function, pprint\n    >>> a, b, c, G, f= [Function(i) for i in [\'a\', \'b\', \'c\', \'G\', \'f\']]\n    >>> u = f(x,y)\n    >>> ux = u.diff(x)\n    >>> uy = u.diff(y)\n    >>> genform = a(x, y)*u + b(x, y)*ux + c(x, y)*uy - G(x,y)\n    >>> pprint(genform)\n                                         d                     d\n    -G(x, y) + a(x, y)*f(x, y) + b(x, y)*--(f(x, y)) + c(x, y)*--(f(x, y))\n                                         dx                    dy\n\n\n    Examples\n    ========\n\n    >>> from sympy.solvers.pde import pdsolve\n    >>> from sympy import Function, pprint\n    >>> from sympy.abc import x,y\n    >>> f = Function(\'f\')\n    >>> eq =  x*(u.diff(x)) - y*(u.diff(y)) + y**2*u - y**2\n    >>> pdsolve(eq)\n    Eq(f(x, y), F(x*y)*exp(y**2/2) + 1)\n\n    References\n    ==========\n\n    - Viktor Grigoryan, "Partial Differential Equations"\n      Math 124A - Fall 2010, pp.7\n\n    '
    from sympy.solvers.ode import dsolve
    (xi, eta) = symbols('xi eta')
    f = func.func
    x = func.args[0]
    y = func.args[1]
    b = match[match['b']]
    c = match[match['c']]
    d = match[match['d']]
    e = -match[match['e']]
    if not d:
        if not (b and c):
            if c:
                try:
                    tsol = integrate(e / c, y)
                except NotImplementedError:
                    raise NotImplementedError('Unable to find a solution due to inability of integrate')
                else:
                    return Eq(f(x, y), solvefun(x) + tsol)
            if b:
                try:
                    tsol = integrate(e / b, x)
                except NotImplementedError:
                    raise NotImplementedError('Unable to find a solution due to inability of integrate')
                else:
                    return Eq(f(x, y), solvefun(y) + tsol)
    if not c:
        plode = f(x).diff(x) * b + d * f(x) - e
        sol = dsolve(plode, f(x))
        syms = sol.free_symbols - plode.free_symbols - {x, y}
        rhs = _simplify_variable_coeff(sol.rhs, syms, solvefun, y)
        return Eq(f(x, y), rhs)
    if not b:
        plode = f(y).diff(y) * c + d * f(y) - e
        sol = dsolve(plode, f(y))
        syms = sol.free_symbols - plode.free_symbols - {x, y}
        rhs = _simplify_variable_coeff(sol.rhs, syms, solvefun, x)
        return Eq(f(x, y), rhs)
    dummy = Function('d')
    h = (c / b).subs(y, dummy(x))
    sol = dsolve(dummy(x).diff(x) - h, dummy(x))
    if isinstance(sol, list):
        sol = sol[0]
    solsym = sol.free_symbols - h.free_symbols - {x, y}
    if len(solsym) == 1:
        solsym = solsym.pop()
        etat = solve(sol, solsym)[0].subs(dummy(x), y)
        ysub = solve(eta - etat, y)[0]
        deq = (b * f(x).diff(x) + d * f(x) - e).subs(y, ysub)
        final = dsolve(deq, f(x), hint='1st_linear').rhs
        if isinstance(final, list):
            final = final[0]
        finsyms = final.free_symbols - deq.free_symbols - {x, y}
        rhs = _simplify_variable_coeff(final, finsyms, solvefun, etat)
        return Eq(f(x, y), rhs)
    else:
        raise NotImplementedError('Cannot solve the partial differential equation due to inability of constantsimp')

def _simplify_variable_coeff(sol, syms, func, funcarg):
    if False:
        return 10
    '\n    Helper function to replace constants by functions in 1st_linear_variable_coeff\n    '
    eta = Symbol('eta')
    if len(syms) == 1:
        sym = syms.pop()
        final = sol.subs(sym, func(funcarg))
    else:
        for (key, sym) in enumerate(syms):
            final = sol.subs(sym, func(funcarg))
    return simplify(final.subs(eta, funcarg))

def pde_separate(eq, fun, sep, strategy='mul'):
    if False:
        print('Hello World!')
    "Separate variables in partial differential equation either by additive\n    or multiplicative separation approach. It tries to rewrite an equation so\n    that one of the specified variables occurs on a different side of the\n    equation than the others.\n\n    :param eq: Partial differential equation\n\n    :param fun: Original function F(x, y, z)\n\n    :param sep: List of separated functions [X(x), u(y, z)]\n\n    :param strategy: Separation strategy. You can choose between additive\n        separation ('add') and multiplicative separation ('mul') which is\n        default.\n\n    Examples\n    ========\n\n    >>> from sympy import E, Eq, Function, pde_separate, Derivative as D\n    >>> from sympy.abc import x, t\n    >>> u, X, T = map(Function, 'uXT')\n\n    >>> eq = Eq(D(u(x, t), x), E**(u(x, t))*D(u(x, t), t))\n    >>> pde_separate(eq, u(x, t), [X(x), T(t)], strategy='add')\n    [exp(-X(x))*Derivative(X(x), x), exp(T(t))*Derivative(T(t), t)]\n\n    >>> eq = Eq(D(u(x, t), x, 2), D(u(x, t), t, 2))\n    >>> pde_separate(eq, u(x, t), [X(x), T(t)], strategy='mul')\n    [Derivative(X(x), (x, 2))/X(x), Derivative(T(t), (t, 2))/T(t)]\n\n    See Also\n    ========\n    pde_separate_add, pde_separate_mul\n    "
    do_add = False
    if strategy == 'add':
        do_add = True
    elif strategy == 'mul':
        do_add = False
    else:
        raise ValueError('Unknown strategy: %s' % strategy)
    if isinstance(eq, Equality):
        if eq.rhs != 0:
            return pde_separate(Eq(eq.lhs - eq.rhs, 0), fun, sep, strategy)
    else:
        return pde_separate(Eq(eq, 0), fun, sep, strategy)
    if eq.rhs != 0:
        raise ValueError('Value should be 0')
    orig_args = list(fun.args)
    subs_args = [arg for s in sep for arg in s.args]
    if do_add:
        functions = reduce(operator.add, sep)
    else:
        functions = reduce(operator.mul, sep)
    if len(subs_args) != len(orig_args):
        raise ValueError('Variable counts do not match')
    if has_dups(subs_args):
        raise ValueError('Duplicate substitution arguments detected')
    if set(orig_args) != set(subs_args):
        raise ValueError('Arguments do not match')
    result = eq.lhs.subs(fun, functions).doit()
    if not do_add:
        eq = 0
        for i in result.args:
            eq += i / functions
        result = eq
    svar = subs_args[0]
    dvar = subs_args[1:]
    return _separate(result, svar, dvar)

def pde_separate_add(eq, fun, sep):
    if False:
        for i in range(10):
            print('nop')
    "\n    Helper function for searching additive separable solutions.\n\n    Consider an equation of two independent variables x, y and a dependent\n    variable w, we look for the product of two functions depending on different\n    arguments:\n\n    `w(x, y, z) = X(x) + y(y, z)`\n\n    Examples\n    ========\n\n    >>> from sympy import E, Eq, Function, pde_separate_add, Derivative as D\n    >>> from sympy.abc import x, t\n    >>> u, X, T = map(Function, 'uXT')\n\n    >>> eq = Eq(D(u(x, t), x), E**(u(x, t))*D(u(x, t), t))\n    >>> pde_separate_add(eq, u(x, t), [X(x), T(t)])\n    [exp(-X(x))*Derivative(X(x), x), exp(T(t))*Derivative(T(t), t)]\n\n    "
    return pde_separate(eq, fun, sep, strategy='add')

def pde_separate_mul(eq, fun, sep):
    if False:
        for i in range(10):
            print('nop')
    "\n    Helper function for searching multiplicative separable solutions.\n\n    Consider an equation of two independent variables x, y and a dependent\n    variable w, we look for the product of two functions depending on different\n    arguments:\n\n    `w(x, y, z) = X(x)*u(y, z)`\n\n    Examples\n    ========\n\n    >>> from sympy import Function, Eq, pde_separate_mul, Derivative as D\n    >>> from sympy.abc import x, y\n    >>> u, X, Y = map(Function, 'uXY')\n\n    >>> eq = Eq(D(u(x, y), x, 2), D(u(x, y), y, 2))\n    >>> pde_separate_mul(eq, u(x, y), [X(x), Y(y)])\n    [Derivative(X(x), (x, 2))/X(x), Derivative(Y(y), (y, 2))/Y(y)]\n\n    "
    return pde_separate(eq, fun, sep, strategy='mul')

def _separate(eq, dep, others):
    if False:
        for i in range(10):
            print('nop')
    'Separate expression into two parts based on dependencies of variables.'
    terms = set()
    for term in eq.args:
        if term.is_Mul:
            for i in term.args:
                if i.is_Derivative and (not i.has(*others)):
                    terms.add(term)
                    continue
        elif term.is_Derivative and (not term.has(*others)):
            terms.add(term)
    div = set()
    for term in terms:
        (ext, sep) = term.expand().as_independent(dep)
        if sep.has(*others):
            return None
        div.add(ext)
    if len(div) > 0:
        eq = Add(*[simplify(Add(*[term / i for i in div])) for term in eq.args])
    div = set()
    lhs = rhs = 0
    for term in eq.args:
        if not term.has(*others):
            lhs += term
            continue
        (temp, sep) = term.expand().as_independent(dep)
        if sep.has(*others):
            return None
        div.add(sep)
        rhs -= term.expand()
    fulldiv = reduce(operator.add, div)
    lhs = simplify(lhs / fulldiv).expand()
    rhs = simplify(rhs / fulldiv).expand()
    if lhs.has(*others) or rhs.has(dep):
        return None
    return [lhs, rhs]