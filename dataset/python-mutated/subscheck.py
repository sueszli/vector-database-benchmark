from sympy.core import S, Pow
from sympy.core.function import Derivative, AppliedUndef, diff
from sympy.core.relational import Equality, Eq
from sympy.core.symbol import Dummy
from sympy.core.sympify import sympify
from sympy.logic.boolalg import BooleanAtom
from sympy.functions import exp
from sympy.series import Order
from sympy.simplify.simplify import simplify, posify, besselsimp
from sympy.simplify.trigsimp import trigsimp
from sympy.simplify.sqrtdenest import sqrtdenest
from sympy.solvers import solve
from sympy.solvers.deutils import _preprocess, ode_order
from sympy.utilities.iterables import iterable, is_sequence

def sub_func_doit(eq, func, new):
    if False:
        return 10
    "\n    When replacing the func with something else, we usually want the\n    derivative evaluated, so this function helps in making that happen.\n\n    Examples\n    ========\n\n    >>> from sympy import Derivative, symbols, Function\n    >>> from sympy.solvers.ode.subscheck import sub_func_doit\n    >>> x, z = symbols('x, z')\n    >>> y = Function('y')\n\n    >>> sub_func_doit(3*Derivative(y(x), x) - 1, y(x), x)\n    2\n\n    >>> sub_func_doit(x*Derivative(y(x), x) - y(x)**2 + y(x), y(x),\n    ... 1/(x*(z + 1/x)))\n    x*(-1/(x**2*(z + 1/x)) + 1/(x**3*(z + 1/x)**2)) + 1/(x*(z + 1/x))\n    ...- 1/(x**2*(z + 1/x)**2)\n    "
    reps = {func: new}
    for d in eq.atoms(Derivative):
        if d.expr == func:
            reps[d] = new.diff(*d.variable_count)
        else:
            reps[d] = d.xreplace({func: new}).doit(deep=False)
    return eq.xreplace(reps)

def checkodesol(ode, sol, func=None, order='auto', solve_for_func=True):
    if False:
        i = 10
        return i + 15
    "\n    Substitutes ``sol`` into ``ode`` and checks that the result is ``0``.\n\n    This works when ``func`` is one function, like `f(x)` or a list of\n    functions like `[f(x), g(x)]` when `ode` is a system of ODEs.  ``sol`` can\n    be a single solution or a list of solutions.  Each solution may be an\n    :py:class:`~sympy.core.relational.Equality` that the solution satisfies,\n    e.g. ``Eq(f(x), C1), Eq(f(x) + C1, 0)``; or simply an\n    :py:class:`~sympy.core.expr.Expr`, e.g. ``f(x) - C1``. In most cases it\n    will not be necessary to explicitly identify the function, but if the\n    function cannot be inferred from the original equation it can be supplied\n    through the ``func`` argument.\n\n    If a sequence of solutions is passed, the same sort of container will be\n    used to return the result for each solution.\n\n    It tries the following methods, in order, until it finds zero equivalence:\n\n    1. Substitute the solution for `f` in the original equation.  This only\n       works if ``ode`` is solved for `f`.  It will attempt to solve it first\n       unless ``solve_for_func == False``.\n    2. Take `n` derivatives of the solution, where `n` is the order of\n       ``ode``, and check to see if that is equal to the solution.  This only\n       works on exact ODEs.\n    3. Take the 1st, 2nd, ..., `n`\\th derivatives of the solution, each time\n       solving for the derivative of `f` of that order (this will always be\n       possible because `f` is a linear operator). Then back substitute each\n       derivative into ``ode`` in reverse order.\n\n    This function returns a tuple.  The first item in the tuple is ``True`` if\n    the substitution results in ``0``, and ``False`` otherwise. The second\n    item in the tuple is what the substitution results in.  It should always\n    be ``0`` if the first item is ``True``. Sometimes this function will\n    return ``False`` even when an expression is identically equal to ``0``.\n    This happens when :py:meth:`~sympy.simplify.simplify.simplify` does not\n    reduce the expression to ``0``.  If an expression returned by this\n    function vanishes identically, then ``sol`` really is a solution to\n    the ``ode``.\n\n    If this function seems to hang, it is probably because of a hard\n    simplification.\n\n    To use this function to test, test the first item of the tuple.\n\n    Examples\n    ========\n\n    >>> from sympy import (Eq, Function, checkodesol, symbols,\n    ...     Derivative, exp)\n    >>> x, C1, C2 = symbols('x,C1,C2')\n    >>> f, g = symbols('f g', cls=Function)\n    >>> checkodesol(f(x).diff(x), Eq(f(x), C1))\n    (True, 0)\n    >>> assert checkodesol(f(x).diff(x), C1)[0]\n    >>> assert not checkodesol(f(x).diff(x), x)[0]\n    >>> checkodesol(f(x).diff(x, 2), x**2)\n    (False, 2)\n\n    >>> eqs = [Eq(Derivative(f(x), x), f(x)), Eq(Derivative(g(x), x), g(x))]\n    >>> sol = [Eq(f(x), C1*exp(x)), Eq(g(x), C2*exp(x))]\n    >>> checkodesol(eqs, sol)\n    (True, [0, 0])\n\n    "
    if iterable(ode):
        return checksysodesol(ode, sol, func=func)
    if not isinstance(ode, Equality):
        ode = Eq(ode, 0)
    if func is None:
        try:
            (_, func) = _preprocess(ode.lhs)
        except ValueError:
            funcs = [s.atoms(AppliedUndef) for s in (sol if is_sequence(sol, set) else [sol])]
            funcs = set().union(*funcs)
            if len(funcs) != 1:
                raise ValueError('must pass func arg to checkodesol for this case.')
            func = funcs.pop()
    if not isinstance(func, AppliedUndef) or len(func.args) != 1:
        raise ValueError('func must be a function of one variable, not %s' % func)
    if is_sequence(sol, set):
        return type(sol)([checkodesol(ode, i, order=order, solve_for_func=solve_for_func) for i in sol])
    if not isinstance(sol, Equality):
        sol = Eq(func, sol)
    elif sol.rhs == func:
        sol = sol.reversed
    if order == 'auto':
        order = ode_order(ode, func)
    solved = sol.lhs == func and (not sol.rhs.has(func))
    if solve_for_func and (not solved):
        rhs = solve(sol, func)
        if rhs:
            eqs = [Eq(func, t) for t in rhs]
            if len(rhs) == 1:
                eqs = eqs[0]
            return checkodesol(ode, eqs, order=order, solve_for_func=False)
    x = func.args[0]
    if sol.has(Order):
        assert sol.lhs == func
        Oterm = sol.rhs.getO()
        solrhs = sol.rhs.removeO()
        Oexpr = Oterm.expr
        assert isinstance(Oexpr, Pow)
        sorder = Oexpr.exp
        assert Oterm == Order(x ** sorder)
        odesubs = (ode.lhs - ode.rhs).subs(func, solrhs).doit().expand()
        neworder = Order(x ** (sorder - order))
        odesubs = odesubs + neworder
        assert odesubs.getO() == neworder
        residual = odesubs.removeO()
        return (residual == 0, residual)
    s = True
    testnum = 0
    while s:
        if testnum == 0:
            ode_diff = ode.lhs - ode.rhs
            if sol.lhs == func:
                s = sub_func_doit(ode_diff, func, sol.rhs)
                s = besselsimp(s)
            else:
                testnum += 1
                continue
            ss = simplify(s.rewrite(exp))
            if ss:
                s = ss.expand(force=True)
            else:
                s = 0
            testnum += 1
        elif testnum == 1:
            s = simplify(trigsimp(diff(sol.lhs, x, order) - diff(sol.rhs, x, order)) - trigsimp(ode.lhs) + trigsimp(ode.rhs))
            testnum += 1
        elif testnum == 2:
            if sol.lhs == func and (not sol.rhs.has(func)):
                diffsols = {0: sol.rhs}
            elif sol.rhs == func and (not sol.lhs.has(func)):
                diffsols = {0: sol.lhs}
            else:
                diffsols = {}
            sol = sol.lhs - sol.rhs
            for i in range(1, order + 1):
                if i == 1:
                    ds = sol.diff(x)
                    try:
                        sdf = solve(ds, func.diff(x, i))
                        if not sdf:
                            raise NotImplementedError
                    except NotImplementedError:
                        testnum += 1
                        break
                    else:
                        diffsols[i] = sdf[0]
                else:
                    diffsols[i] = diffsols[i - 1].diff(x)
            if testnum > 2:
                continue
            else:
                (lhs, rhs) = (ode.lhs, ode.rhs)
                for i in range(order, -1, -1):
                    if i == 0 and 0 not in diffsols:
                        break
                    lhs = sub_func_doit(lhs, func.diff(x, i), diffsols[i])
                    rhs = sub_func_doit(rhs, func.diff(x, i), diffsols[i])
                    ode_or_bool = Eq(lhs, rhs)
                    ode_or_bool = simplify(ode_or_bool)
                    if isinstance(ode_or_bool, (bool, BooleanAtom)):
                        if ode_or_bool:
                            lhs = rhs = S.Zero
                    else:
                        lhs = ode_or_bool.lhs
                        rhs = ode_or_bool.rhs
                num = trigsimp((lhs - rhs).as_numer_denom()[0])
                _func = Dummy('func')
                num = num.subs(func, _func)
                (num, reps) = posify(num)
                s = simplify(num).xreplace(reps).xreplace({_func: func})
                testnum += 1
        else:
            break
    if not s:
        return (True, s)
    elif s is True:
        raise NotImplementedError('Unable to test if ' + str(sol) + ' is a solution to ' + str(ode) + '.')
    else:
        return (False, s)

def checksysodesol(eqs, sols, func=None):
    if False:
        return 10
    "\n    Substitutes corresponding ``sols`` for each functions into each ``eqs`` and\n    checks that the result of substitutions for each equation is ``0``. The\n    equations and solutions passed can be any iterable.\n\n    This only works when each ``sols`` have one function only, like `x(t)` or `y(t)`.\n    For each function, ``sols`` can have a single solution or a list of solutions.\n    In most cases it will not be necessary to explicitly identify the function,\n    but if the function cannot be inferred from the original equation it\n    can be supplied through the ``func`` argument.\n\n    When a sequence of equations is passed, the same sequence is used to return\n    the result for each equation with each function substituted with corresponding\n    solutions.\n\n    It tries the following method to find zero equivalence for each equation:\n\n    Substitute the solutions for functions, like `x(t)` and `y(t)` into the\n    original equations containing those functions.\n    This function returns a tuple.  The first item in the tuple is ``True`` if\n    the substitution results for each equation is ``0``, and ``False`` otherwise.\n    The second item in the tuple is what the substitution results in.  Each element\n    of the ``list`` should always be ``0`` corresponding to each equation if the\n    first item is ``True``. Note that sometimes this function may return ``False``,\n    but with an expression that is identically equal to ``0``, instead of returning\n    ``True``.  This is because :py:meth:`~sympy.simplify.simplify.simplify` cannot\n    reduce the expression to ``0``.  If an expression returned by each function\n    vanishes identically, then ``sols`` really is a solution to ``eqs``.\n\n    If this function seems to hang, it is probably because of a difficult simplification.\n\n    Examples\n    ========\n\n    >>> from sympy import Eq, diff, symbols, sin, cos, exp, sqrt, S, Function\n    >>> from sympy.solvers.ode.subscheck import checksysodesol\n    >>> C1, C2 = symbols('C1:3')\n    >>> t = symbols('t')\n    >>> x, y = symbols('x, y', cls=Function)\n    >>> eq = (Eq(diff(x(t),t), x(t) + y(t) + 17), Eq(diff(y(t),t), -2*x(t) + y(t) + 12))\n    >>> sol = [Eq(x(t), (C1*sin(sqrt(2)*t) + C2*cos(sqrt(2)*t))*exp(t) - S(5)/3),\n    ... Eq(y(t), (sqrt(2)*C1*cos(sqrt(2)*t) - sqrt(2)*C2*sin(sqrt(2)*t))*exp(t) - S(46)/3)]\n    >>> checksysodesol(eq, sol)\n    (True, [0, 0])\n    >>> eq = (Eq(diff(x(t),t),x(t)*y(t)**4), Eq(diff(y(t),t),y(t)**3))\n    >>> sol = [Eq(x(t), C1*exp(-1/(4*(C2 + t)))), Eq(y(t), -sqrt(2)*sqrt(-1/(C2 + t))/2),\n    ... Eq(x(t), C1*exp(-1/(4*(C2 + t)))), Eq(y(t), sqrt(2)*sqrt(-1/(C2 + t))/2)]\n    >>> checksysodesol(eq, sol)\n    (True, [0, 0])\n\n    "

    def _sympify(eq):
        if False:
            return 10
        return list(map(sympify, eq if iterable(eq) else [eq]))
    eqs = _sympify(eqs)
    for i in range(len(eqs)):
        if isinstance(eqs[i], Equality):
            eqs[i] = eqs[i].lhs - eqs[i].rhs
    if func is None:
        funcs = []
        for eq in eqs:
            derivs = eq.atoms(Derivative)
            func = set().union(*[d.atoms(AppliedUndef) for d in derivs])
            funcs.extend(func)
        funcs = list(set(funcs))
    if not all((isinstance(func, AppliedUndef) and len(func.args) == 1 for func in funcs)) and len({func.args for func in funcs}) != 1:
        raise ValueError('func must be a function of one variable, not %s' % func)
    for sol in sols:
        if len(sol.atoms(AppliedUndef)) != 1:
            raise ValueError('solutions should have one function only')
    if len(funcs) != len({sol.lhs for sol in sols}):
        raise ValueError('number of solutions provided does not match the number of equations')
    dictsol = {}
    for sol in sols:
        func = list(sol.atoms(AppliedUndef))[0]
        if sol.rhs == func:
            sol = sol.reversed
        solved = sol.lhs == func and (not sol.rhs.has(func))
        if not solved:
            rhs = solve(sol, func)
            if not rhs:
                raise NotImplementedError
        else:
            rhs = sol.rhs
        dictsol[func] = rhs
    checkeq = []
    for eq in eqs:
        for func in funcs:
            eq = sub_func_doit(eq, func, dictsol[func])
        ss = simplify(eq)
        if ss != 0:
            eq = ss.expand(force=True)
            if eq != 0:
                eq = sqrtdenest(eq).simplify()
        else:
            eq = 0
        checkeq.append(eq)
    if len(set(checkeq)) == 1 and list(set(checkeq))[0] == 0:
        return (True, checkeq)
    else:
        return (False, checkeq)