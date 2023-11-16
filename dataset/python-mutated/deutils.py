"""Utility functions for classifying and solving
ordinary and partial differential equations.

Contains
========
_preprocess
ode_order
_desolve

"""
from sympy.core import Pow
from sympy.core.function import Derivative, AppliedUndef
from sympy.core.relational import Equality
from sympy.core.symbol import Wild

def _preprocess(expr, func=None, hint='_Integral'):
    if False:
        while True:
            i = 10
    'Prepare expr for solving by making sure that differentiation\n    is done so that only func remains in unevaluated derivatives and\n    (if hint does not end with _Integral) that doit is applied to all\n    other derivatives. If hint is None, do not do any differentiation.\n    (Currently this may cause some simple differential equations to\n    fail.)\n\n    In case func is None, an attempt will be made to autodetect the\n    function to be solved for.\n\n    >>> from sympy.solvers.deutils import _preprocess\n    >>> from sympy import Derivative, Function\n    >>> from sympy.abc import x, y, z\n    >>> f, g = map(Function, \'fg\')\n\n    If f(x)**p == 0 and p>0 then we can solve for f(x)=0\n    >>> _preprocess((f(x).diff(x)-4)**5, f(x))\n    (Derivative(f(x), x) - 4, f(x))\n\n    Apply doit to derivatives that contain more than the function\n    of interest:\n\n    >>> _preprocess(Derivative(f(x) + x, x))\n    (Derivative(f(x), x) + 1, f(x))\n\n    Do others if the differentiation variable(s) intersect with those\n    of the function of interest or contain the function of interest:\n\n    >>> _preprocess(Derivative(g(x), y, z), f(y))\n    (0, f(y))\n    >>> _preprocess(Derivative(f(y), z), f(y))\n    (0, f(y))\n\n    Do others if the hint does not end in \'_Integral\' (the default\n    assumes that it does):\n\n    >>> _preprocess(Derivative(g(x), y), f(x))\n    (Derivative(g(x), y), f(x))\n    >>> _preprocess(Derivative(f(x), y), f(x), hint=\'\')\n    (0, f(x))\n\n    Do not do any derivatives if hint is None:\n\n    >>> eq = Derivative(f(x) + 1, x) + Derivative(f(x), y)\n    >>> _preprocess(eq, f(x), hint=None)\n    (Derivative(f(x) + 1, x) + Derivative(f(x), y), f(x))\n\n    If it\'s not clear what the function of interest is, it must be given:\n\n    >>> eq = Derivative(f(x) + g(x), x)\n    >>> _preprocess(eq, g(x))\n    (Derivative(f(x), x) + Derivative(g(x), x), g(x))\n    >>> try: _preprocess(eq)\n    ... except ValueError: print("A ValueError was raised.")\n    A ValueError was raised.\n\n    '
    if isinstance(expr, Pow):
        if expr.exp.is_positive:
            expr = expr.base
    derivs = expr.atoms(Derivative)
    if not func:
        funcs = set().union(*[d.atoms(AppliedUndef) for d in derivs])
        if len(funcs) != 1:
            raise ValueError('The function cannot be automatically detected for %s.' % expr)
        func = funcs.pop()
    fvars = set(func.args)
    if hint is None:
        return (expr, func)
    reps = [(d, d.doit()) for d in derivs if not hint.endswith('_Integral') or d.has(func) or set(d.variables) & fvars]
    eq = expr.subs(reps)
    return (eq, func)

def ode_order(expr, func):
    if False:
        while True:
            i = 10
    "\n    Returns the order of a given differential\n    equation with respect to func.\n\n    This function is implemented recursively.\n\n    Examples\n    ========\n\n    >>> from sympy import Function\n    >>> from sympy.solvers.deutils import ode_order\n    >>> from sympy.abc import x\n    >>> f, g = map(Function, ['f', 'g'])\n    >>> ode_order(f(x).diff(x, 2) + f(x).diff(x)**2 +\n    ... f(x).diff(x), f(x))\n    2\n    >>> ode_order(f(x).diff(x, 2) + g(x).diff(x, 3), f(x))\n    2\n    >>> ode_order(f(x).diff(x, 2) + g(x).diff(x, 3), g(x))\n    3\n\n    "
    a = Wild('a', exclude=[func])
    if expr.match(a):
        return 0
    if isinstance(expr, Derivative):
        if expr.args[0] == func:
            return len(expr.variables)
        else:
            args = expr.args[0].args
            rv = len(expr.variables)
            if args:
                rv += max((ode_order(_, func) for _ in args))
            return rv
    else:
        return max((ode_order(_, func) for _ in expr.args)) if expr.args else 0

def _desolve(eq, func=None, hint='default', ics=None, simplify=True, *, prep=True, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    'This is a helper function to dsolve and pdsolve in the ode\n    and pde modules.\n\n    If the hint provided to the function is "default", then a dict with\n    the following keys are returned\n\n    \'func\'    - It provides the function for which the differential equation\n                has to be solved. This is useful when the expression has\n                more than one function in it.\n\n    \'default\' - The default key as returned by classifier functions in ode\n                and pde.py\n\n    \'hint\'    - The hint given by the user for which the differential equation\n                is to be solved. If the hint given by the user is \'default\',\n                then the value of \'hint\' and \'default\' is the same.\n\n    \'order\'   - The order of the function as returned by ode_order\n\n    \'match\'   - It returns the match as given by the classifier functions, for\n                the default hint.\n\n    If the hint provided to the function is not "default" and is not in\n    (\'all\', \'all_Integral\', \'best\'), then a dict with the above mentioned keys\n    is returned along with the keys which are returned when dict in\n    classify_ode or classify_pde is set True\n\n    If the hint given is in (\'all\', \'all_Integral\', \'best\'), then this function\n    returns a nested dict, with the keys, being the set of classified hints\n    returned by classifier functions, and the values being the dict of form\n    as mentioned above.\n\n    Key \'eq\' is a common key to all the above mentioned hints which returns an\n    expression if eq given by user is an Equality.\n\n    See Also\n    ========\n    classify_ode(ode.py)\n    classify_pde(pde.py)\n    '
    if isinstance(eq, Equality):
        eq = eq.lhs - eq.rhs
    if prep or func is None:
        (eq, func) = _preprocess(eq, func)
        prep = False
    type = kwargs.get('type', None)
    xi = kwargs.get('xi')
    eta = kwargs.get('eta')
    x0 = kwargs.get('x0', 0)
    terms = kwargs.get('n')
    if type == 'ode':
        from sympy.solvers.ode import classify_ode, allhints
        classifier = classify_ode
        string = 'ODE '
        dummy = ''
    elif type == 'pde':
        from sympy.solvers.pde import classify_pde, allhints
        classifier = classify_pde
        string = 'PDE '
        dummy = 'p'
    if kwargs.get('classify', True):
        hints = classifier(eq, func, dict=True, ics=ics, xi=xi, eta=eta, n=terms, x0=x0, hint=hint, prep=prep)
    else:
        hints = kwargs.get('hint', {'default': hint, hint: kwargs['match'], 'order': kwargs['order']})
    if not hints['default']:
        if hint not in allhints and hint != 'default':
            raise ValueError('Hint not recognized: ' + hint)
        elif hint not in hints['ordered_hints'] and hint != 'default':
            raise ValueError(string + str(eq) + ' does not match hint ' + hint)
        elif hints['order'] == 0:
            raise ValueError(str(eq) + ' is not a solvable differential equation in ' + str(func))
        else:
            raise NotImplementedError(dummy + 'solve' + ': Cannot solve ' + str(eq))
    if hint == 'default':
        return _desolve(eq, func, ics=ics, hint=hints['default'], simplify=simplify, prep=prep, x0=x0, classify=False, order=hints['order'], match=hints[hints['default']], xi=xi, eta=eta, n=terms, type=type)
    elif hint in ('all', 'all_Integral', 'best'):
        retdict = {}
        gethints = set(hints) - {'order', 'default', 'ordered_hints'}
        if hint == 'all_Integral':
            for i in hints:
                if i.endswith('_Integral'):
                    gethints.remove(i[:-len('_Integral')])
            for k in ['1st_homogeneous_coeff_best', '1st_power_series', 'lie_group', '2nd_power_series_ordinary', '2nd_power_series_regular']:
                if k in gethints:
                    gethints.remove(k)
        for i in gethints:
            sol = _desolve(eq, func, ics=ics, hint=i, x0=x0, simplify=simplify, prep=prep, classify=False, n=terms, order=hints['order'], match=hints[i], type=type)
            retdict[i] = sol
        retdict['all'] = True
        retdict['eq'] = eq
        return retdict
    elif hint not in allhints:
        raise ValueError('Hint not recognized: ' + hint)
    elif hint not in hints:
        raise ValueError(string + str(eq) + ' does not match hint ' + hint)
    else:
        hints['hint'] = hint
    hints.update({'func': func, 'eq': eq})
    return hints