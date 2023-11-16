"""
This module implements the Residue function and related tools for working
with residues.
"""
from sympy.core.mul import Mul
from sympy.core.singleton import S
from sympy.core.sympify import sympify
from sympy.utilities.timeutils import timethis

@timethis('residue')
def residue(expr, x, x0):
    if False:
        for i in range(10):
            print('nop')
    '\n    Finds the residue of ``expr`` at the point x=x0.\n\n    The residue is defined as the coefficient of ``1/(x-x0)`` in the power series\n    expansion about ``x=x0``.\n\n    Examples\n    ========\n\n    >>> from sympy import Symbol, residue, sin\n    >>> x = Symbol("x")\n    >>> residue(1/x, x, 0)\n    1\n    >>> residue(1/x**2, x, 0)\n    0\n    >>> residue(2/sin(x), x, 0)\n    2\n\n    This function is essential for the Residue Theorem [1].\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Residue_theorem\n    '
    from sympy.series.order import Order
    from sympy.simplify.radsimp import collect
    expr = sympify(expr)
    if x0 != 0:
        expr = expr.subs(x, x + x0)
    for n in (0, 1, 2, 4, 8, 16, 32):
        s = expr.nseries(x, n=n)
        if not s.has(Order) or s.getn() >= 0:
            break
    s = collect(s.removeO(), x)
    if s.is_Add:
        args = s.args
    else:
        args = [s]
    res = S.Zero
    for arg in args:
        (c, m) = arg.as_coeff_mul(x)
        m = Mul(*m)
        if not (m in (S.One, x) or (m.is_Pow and m.exp.is_Integer)):
            raise NotImplementedError('term of unexpected form: %s' % m)
        if m == 1 / x:
            res += c
    return res