from sympy.core import Function, Pow, sympify, Expr
from sympy.core.relational import Relational
from sympy.core.singleton import S
from sympy.polys import Poly, decompose
from sympy.utilities.misc import func_name
from sympy.functions.elementary.miscellaneous import Min, Max

def decompogen(f, symbol):
    if False:
        return 10
    '\n    Computes General functional decomposition of ``f``.\n    Given an expression ``f``, returns a list ``[f_1, f_2, ..., f_n]``,\n    where::\n              f = f_1 o f_2 o ... f_n = f_1(f_2(... f_n))\n\n    Note: This is a General decomposition function. It also decomposes\n    Polynomials. For only Polynomial decomposition see ``decompose`` in polys.\n\n    Examples\n    ========\n\n    >>> from sympy.abc import x\n    >>> from sympy import decompogen, sqrt, sin, cos\n    >>> decompogen(sin(cos(x)), x)\n    [sin(x), cos(x)]\n    >>> decompogen(sin(x)**2 + sin(x) + 1, x)\n    [x**2 + x + 1, sin(x)]\n    >>> decompogen(sqrt(6*x**2 - 5), x)\n    [sqrt(x), 6*x**2 - 5]\n    >>> decompogen(sin(sqrt(cos(x**2 + 1))), x)\n    [sin(x), sqrt(x), cos(x), x**2 + 1]\n    >>> decompogen(x**4 + 2*x**3 - x - 1, x)\n    [x**2 - x - 1, x**2 + x]\n\n    '
    f = sympify(f)
    if not isinstance(f, Expr) or isinstance(f, Relational):
        raise TypeError('expecting Expr but got: `%s`' % func_name(f))
    if symbol not in f.free_symbols:
        return [f]
    if isinstance(f, (Function, Pow)):
        if f.is_Pow and f.base == S.Exp1:
            arg = f.exp
        else:
            arg = f.args[0]
        if arg == symbol:
            return [f]
        return [f.subs(arg, symbol)] + decompogen(arg, symbol)
    if isinstance(f, (Min, Max)):
        args = list(f.args)
        d0 = None
        for (i, a) in enumerate(args):
            if not a.has_free(symbol):
                continue
            d = decompogen(a, symbol)
            if len(d) == 1:
                d = [symbol] + d
            if d0 is None:
                d0 = d[1:]
            elif d[1:] != d0:
                d = [symbol]
                break
            args[i] = d[0]
        if d[0] == symbol:
            return [f]
        return [f.func(*args)] + d0
    fp = Poly(f)
    gens = list(filter(lambda x: symbol in x.free_symbols, fp.gens))
    if len(gens) == 1 and gens[0] != symbol:
        f1 = f.subs(gens[0], symbol)
        f2 = gens[0]
        return [f1] + decompogen(f2, symbol)
    try:
        return decompose(f)
    except ValueError:
        return [f]

def compogen(g_s, symbol):
    if False:
        while True:
            i = 10
    '\n    Returns the composition of functions.\n    Given a list of functions ``g_s``, returns their composition ``f``,\n    where:\n        f = g_1 o g_2 o .. o g_n\n\n    Note: This is a General composition function. It also composes Polynomials.\n    For only Polynomial composition see ``compose`` in polys.\n\n    Examples\n    ========\n\n    >>> from sympy.solvers.decompogen import compogen\n    >>> from sympy.abc import x\n    >>> from sympy import sqrt, sin, cos\n    >>> compogen([sin(x), cos(x)], x)\n    sin(cos(x))\n    >>> compogen([x**2 + x + 1, sin(x)], x)\n    sin(x)**2 + sin(x) + 1\n    >>> compogen([sqrt(x), 6*x**2 - 5], x)\n    sqrt(6*x**2 - 5)\n    >>> compogen([sin(x), sqrt(x), cos(x), x**2 + 1], x)\n    sin(sqrt(cos(x**2 + 1)))\n    >>> compogen([x**2 - x - 1, x**2 + x], x)\n    -x**2 - x + (x**2 + x)**2 - 1\n    '
    if len(g_s) == 1:
        return g_s[0]
    foo = g_s[0].subs(symbol, g_s[1])
    if len(g_s) == 2:
        return foo
    return compogen([foo] + g_s[2:], symbol)