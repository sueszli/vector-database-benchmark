"""
Singularities
=============

This module implements algorithms for finding singularities for a function
and identifying types of functions.

The differential calculus methods in this module include methods to identify
the following function types in the given ``Interval``:
- Increasing
- Strictly Increasing
- Decreasing
- Strictly Decreasing
- Monotonic

"""
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.trigonometric import sec, csc, cot, tan, cos
from sympy.utilities.misc import filldedent

def singularities(expression, symbol, domain=None):
    if False:
        print('Hello World!')
    "\n    Find singularities of a given function.\n\n    Parameters\n    ==========\n\n    expression : Expr\n        The target function in which singularities need to be found.\n    symbol : Symbol\n        The symbol over the values of which the singularity in\n        expression in being searched for.\n\n    Returns\n    =======\n\n    Set\n        A set of values for ``symbol`` for which ``expression`` has a\n        singularity. An ``EmptySet`` is returned if ``expression`` has no\n        singularities for any given value of ``Symbol``.\n\n    Raises\n    ======\n\n    NotImplementedError\n        Methods for determining the singularities of this function have\n        not been developed.\n\n    Notes\n    =====\n\n    This function does not find non-isolated singularities\n    nor does it find branch points of the expression.\n\n    Currently supported functions are:\n        - univariate continuous (real or complex) functions\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Mathematical_singularity\n\n    Examples\n    ========\n\n    >>> from sympy import singularities, Symbol, log\n    >>> x = Symbol('x', real=True)\n    >>> y = Symbol('y', real=False)\n    >>> singularities(x**2 + x + 1, x)\n    EmptySet\n    >>> singularities(1/(x + 1), x)\n    {-1}\n    >>> singularities(1/(y**2 + 1), y)\n    {-I, I}\n    >>> singularities(1/(y**3 + 1), y)\n    {-1, 1/2 - sqrt(3)*I/2, 1/2 + sqrt(3)*I/2}\n    >>> singularities(log(x), x)\n    {0}\n\n    "
    from sympy.solvers.solveset import solveset
    if domain is None:
        domain = S.Reals if symbol.is_real else S.Complexes
    try:
        sings = S.EmptySet
        for i in expression.rewrite([sec, csc, cot, tan], cos).atoms(Pow):
            if i.exp.is_infinite:
                raise NotImplementedError
            if i.exp.is_negative:
                sings += solveset(i.base, symbol, domain)
        for i in expression.atoms(log):
            sings += solveset(i.args[0], symbol, domain)
        return sings
    except NotImplementedError:
        raise NotImplementedError(filldedent('\n            Methods for determining the singularities\n            of this function have not been developed.'))

def monotonicity_helper(expression, predicate, interval=S.Reals, symbol=None):
    if False:
        return 10
    "\n    Helper function for functions checking function monotonicity.\n\n    Parameters\n    ==========\n\n    expression : Expr\n        The target function which is being checked\n    predicate : function\n        The property being tested for. The function takes in an integer\n        and returns a boolean. The integer input is the derivative and\n        the boolean result should be true if the property is being held,\n        and false otherwise.\n    interval : Set, optional\n        The range of values in which we are testing, defaults to all reals.\n    symbol : Symbol, optional\n        The symbol present in expression which gets varied over the given range.\n\n    It returns a boolean indicating whether the interval in which\n    the function's derivative satisfies given predicate is a superset\n    of the given interval.\n\n    Returns\n    =======\n\n    Boolean\n        True if ``predicate`` is true for all the derivatives when ``symbol``\n        is varied in ``range``, False otherwise.\n\n    "
    from sympy.solvers.solveset import solveset
    expression = sympify(expression)
    free = expression.free_symbols
    if symbol is None:
        if len(free) > 1:
            raise NotImplementedError('The function has not yet been implemented for all multivariate expressions.')
    variable = symbol or (free.pop() if free else Symbol('x'))
    derivative = expression.diff(variable)
    predicate_interval = solveset(predicate(derivative), variable, S.Reals)
    return interval.is_subset(predicate_interval)

def is_increasing(expression, interval=S.Reals, symbol=None):
    if False:
        while True:
            i = 10
    '\n    Return whether the function is increasing in the given interval.\n\n    Parameters\n    ==========\n\n    expression : Expr\n        The target function which is being checked.\n    interval : Set, optional\n        The range of values in which we are testing (defaults to set of\n        all real numbers).\n    symbol : Symbol, optional\n        The symbol present in expression which gets varied over the given range.\n\n    Returns\n    =======\n\n    Boolean\n        True if ``expression`` is increasing (either strictly increasing or\n        constant) in the given ``interval``, False otherwise.\n\n    Examples\n    ========\n\n    >>> from sympy import is_increasing\n    >>> from sympy.abc import x, y\n    >>> from sympy import S, Interval, oo\n    >>> is_increasing(x**3 - 3*x**2 + 4*x, S.Reals)\n    True\n    >>> is_increasing(-x**2, Interval(-oo, 0))\n    True\n    >>> is_increasing(-x**2, Interval(0, oo))\n    False\n    >>> is_increasing(4*x**3 - 6*x**2 - 72*x + 30, Interval(-2, 3))\n    False\n    >>> is_increasing(x**2 + y, Interval(1, 2), x)\n    True\n\n    '
    return monotonicity_helper(expression, lambda x: x >= 0, interval, symbol)

def is_strictly_increasing(expression, interval=S.Reals, symbol=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return whether the function is strictly increasing in the given interval.\n\n    Parameters\n    ==========\n\n    expression : Expr\n        The target function which is being checked.\n    interval : Set, optional\n        The range of values in which we are testing (defaults to set of\n        all real numbers).\n    symbol : Symbol, optional\n        The symbol present in expression which gets varied over the given range.\n\n    Returns\n    =======\n\n    Boolean\n        True if ``expression`` is strictly increasing in the given ``interval``,\n        False otherwise.\n\n    Examples\n    ========\n\n    >>> from sympy import is_strictly_increasing\n    >>> from sympy.abc import x, y\n    >>> from sympy import Interval, oo\n    >>> is_strictly_increasing(4*x**3 - 6*x**2 - 72*x + 30, Interval.Ropen(-oo, -2))\n    True\n    >>> is_strictly_increasing(4*x**3 - 6*x**2 - 72*x + 30, Interval.Lopen(3, oo))\n    True\n    >>> is_strictly_increasing(4*x**3 - 6*x**2 - 72*x + 30, Interval.open(-2, 3))\n    False\n    >>> is_strictly_increasing(-x**2, Interval(0, oo))\n    False\n    >>> is_strictly_increasing(-x**2 + y, Interval(-oo, 0), x)\n    False\n\n    '
    return monotonicity_helper(expression, lambda x: x > 0, interval, symbol)

def is_decreasing(expression, interval=S.Reals, symbol=None):
    if False:
        while True:
            i = 10
    '\n    Return whether the function is decreasing in the given interval.\n\n    Parameters\n    ==========\n\n    expression : Expr\n        The target function which is being checked.\n    interval : Set, optional\n        The range of values in which we are testing (defaults to set of\n        all real numbers).\n    symbol : Symbol, optional\n        The symbol present in expression which gets varied over the given range.\n\n    Returns\n    =======\n\n    Boolean\n        True if ``expression`` is decreasing (either strictly decreasing or\n        constant) in the given ``interval``, False otherwise.\n\n    Examples\n    ========\n\n    >>> from sympy import is_decreasing\n    >>> from sympy.abc import x, y\n    >>> from sympy import S, Interval, oo\n    >>> is_decreasing(1/(x**2 - 3*x), Interval.open(S(3)/2, 3))\n    True\n    >>> is_decreasing(1/(x**2 - 3*x), Interval.open(1.5, 3))\n    True\n    >>> is_decreasing(1/(x**2 - 3*x), Interval.Lopen(3, oo))\n    True\n    >>> is_decreasing(1/(x**2 - 3*x), Interval.Ropen(-oo, S(3)/2))\n    False\n    >>> is_decreasing(1/(x**2 - 3*x), Interval.Ropen(-oo, 1.5))\n    False\n    >>> is_decreasing(-x**2, Interval(-oo, 0))\n    False\n    >>> is_decreasing(-x**2 + y, Interval(-oo, 0), x)\n    False\n\n    '
    return monotonicity_helper(expression, lambda x: x <= 0, interval, symbol)

def is_strictly_decreasing(expression, interval=S.Reals, symbol=None):
    if False:
        print('Hello World!')
    '\n    Return whether the function is strictly decreasing in the given interval.\n\n    Parameters\n    ==========\n\n    expression : Expr\n        The target function which is being checked.\n    interval : Set, optional\n        The range of values in which we are testing (defaults to set of\n        all real numbers).\n    symbol : Symbol, optional\n        The symbol present in expression which gets varied over the given range.\n\n    Returns\n    =======\n\n    Boolean\n        True if ``expression`` is strictly decreasing in the given ``interval``,\n        False otherwise.\n\n    Examples\n    ========\n\n    >>> from sympy import is_strictly_decreasing\n    >>> from sympy.abc import x, y\n    >>> from sympy import S, Interval, oo\n    >>> is_strictly_decreasing(1/(x**2 - 3*x), Interval.Lopen(3, oo))\n    True\n    >>> is_strictly_decreasing(1/(x**2 - 3*x), Interval.Ropen(-oo, S(3)/2))\n    False\n    >>> is_strictly_decreasing(1/(x**2 - 3*x), Interval.Ropen(-oo, 1.5))\n    False\n    >>> is_strictly_decreasing(-x**2, Interval(-oo, 0))\n    False\n    >>> is_strictly_decreasing(-x**2 + y, Interval(-oo, 0), x)\n    False\n\n    '
    return monotonicity_helper(expression, lambda x: x < 0, interval, symbol)

def is_monotonic(expression, interval=S.Reals, symbol=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return whether the function is monotonic in the given interval.\n\n    Parameters\n    ==========\n\n    expression : Expr\n        The target function which is being checked.\n    interval : Set, optional\n        The range of values in which we are testing (defaults to set of\n        all real numbers).\n    symbol : Symbol, optional\n        The symbol present in expression which gets varied over the given range.\n\n    Returns\n    =======\n\n    Boolean\n        True if ``expression`` is monotonic in the given ``interval``,\n        False otherwise.\n\n    Raises\n    ======\n\n    NotImplementedError\n        Monotonicity check has not been implemented for the queried function.\n\n    Examples\n    ========\n\n    >>> from sympy import is_monotonic\n    >>> from sympy.abc import x, y\n    >>> from sympy import S, Interval, oo\n    >>> is_monotonic(1/(x**2 - 3*x), Interval.open(S(3)/2, 3))\n    True\n    >>> is_monotonic(1/(x**2 - 3*x), Interval.open(1.5, 3))\n    True\n    >>> is_monotonic(1/(x**2 - 3*x), Interval.Lopen(3, oo))\n    True\n    >>> is_monotonic(x**3 - 3*x**2 + 4*x, S.Reals)\n    True\n    >>> is_monotonic(-x**2, S.Reals)\n    False\n    >>> is_monotonic(x**2 + y + 1, Interval(1, 2), x)\n    True\n\n    '
    from sympy.solvers.solveset import solveset
    expression = sympify(expression)
    free = expression.free_symbols
    if symbol is None and len(free) > 1:
        raise NotImplementedError('is_monotonic has not yet been implemented for all multivariate expressions.')
    variable = symbol or (free.pop() if free else Symbol('x'))
    turning_points = solveset(expression.diff(variable), variable, interval)
    return interval.intersection(turning_points) is S.EmptySet