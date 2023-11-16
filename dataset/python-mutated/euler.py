"""
This module implements a method to find
Euler-Lagrange Equations for given Lagrangian.
"""
from itertools import combinations_with_replacement
from sympy.core.function import Derivative, Function, diff
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.utilities.iterables import iterable

def euler_equations(L, funcs=(), vars=()):
    if False:
        for i in range(10):
            print('nop')
    "\n    Find the Euler-Lagrange equations [1]_ for a given Lagrangian.\n\n    Parameters\n    ==========\n\n    L : Expr\n        The Lagrangian that should be a function of the functions listed\n        in the second argument and their derivatives.\n\n        For example, in the case of two functions $f(x,y)$, $g(x,y)$ and\n        two independent variables $x$, $y$ the Lagrangian has the form:\n\n            .. math:: L\\left(f(x,y),g(x,y),\\frac{\\partial f(x,y)}{\\partial x},\n                      \\frac{\\partial f(x,y)}{\\partial y},\n                      \\frac{\\partial g(x,y)}{\\partial x},\n                      \\frac{\\partial g(x,y)}{\\partial y},x,y\\right)\n\n        In many cases it is not necessary to provide anything, except the\n        Lagrangian, it will be auto-detected (and an error raised if this\n        cannot be done).\n\n    funcs : Function or an iterable of Functions\n        The functions that the Lagrangian depends on. The Euler equations\n        are differential equations for each of these functions.\n\n    vars : Symbol or an iterable of Symbols\n        The Symbols that are the independent variables of the functions.\n\n    Returns\n    =======\n\n    eqns : list of Eq\n        The list of differential equations, one for each function.\n\n    Examples\n    ========\n\n    >>> from sympy import euler_equations, Symbol, Function\n    >>> x = Function('x')\n    >>> t = Symbol('t')\n    >>> L = (x(t).diff(t))**2/2 - x(t)**2/2\n    >>> euler_equations(L, x(t), t)\n    [Eq(-x(t) - Derivative(x(t), (t, 2)), 0)]\n    >>> u = Function('u')\n    >>> x = Symbol('x')\n    >>> L = (u(t, x).diff(t))**2/2 - (u(t, x).diff(x))**2/2\n    >>> euler_equations(L, u(t, x), [t, x])\n    [Eq(-Derivative(u(t, x), (t, 2)) + Derivative(u(t, x), (x, 2)), 0)]\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Euler%E2%80%93Lagrange_equation\n\n    "
    funcs = tuple(funcs) if iterable(funcs) else (funcs,)
    if not funcs:
        funcs = tuple(L.atoms(Function))
    else:
        for f in funcs:
            if not isinstance(f, Function):
                raise TypeError('Function expected, got: %s' % f)
    vars = tuple(vars) if iterable(vars) else (vars,)
    if not vars:
        vars = funcs[0].args
    else:
        vars = tuple((sympify(var) for var in vars))
    if not all((isinstance(v, Symbol) for v in vars)):
        raise TypeError('Variables are not symbols, got %s' % vars)
    for f in funcs:
        if not vars == f.args:
            raise ValueError('Variables %s do not match args: %s' % (vars, f))
    order = max([len(d.variables) for d in L.atoms(Derivative) if d.expr in funcs] + [0])
    eqns = []
    for f in funcs:
        eq = diff(L, f)
        for i in range(1, order + 1):
            for p in combinations_with_replacement(vars, i):
                eq = eq + S.NegativeOne ** i * diff(L, diff(f, *p), *p)
        new_eq = Eq(eq, 0)
        if isinstance(new_eq, Eq):
            eqns.append(new_eq)
    return eqns