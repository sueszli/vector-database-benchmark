""" Functions to support rewriting of SymPy expressions """
from sympy.core.expr import Expr
from sympy.assumptions import ask
from sympy.strategies.tools import subs
from sympy.unify.usympy import rebuild, unify

def rewriterule(source, target, variables=(), condition=None, assume=None):
    if False:
        for i in range(10):
            print('nop')
    ' Rewrite rule.\n\n    Transform expressions that match source into expressions that match target\n    treating all ``variables`` as wilds.\n\n    Examples\n    ========\n\n    >>> from sympy.abc import w, x, y, z\n    >>> from sympy.unify.rewrite import rewriterule\n    >>> from sympy import default_sort_key\n    >>> rl = rewriterule(x + y, x**y, [x, y])\n    >>> sorted(rl(z + 3), key=default_sort_key)\n    [3**z, z**3]\n\n    Use ``condition`` to specify additional requirements.  Inputs are taken in\n    the same order as is found in variables.\n\n    >>> rl = rewriterule(x + y, x**y, [x, y], lambda x, y: x.is_integer)\n    >>> list(rl(z + 3))\n    [3**z]\n\n    Use ``assume`` to specify additional requirements using new assumptions.\n\n    >>> from sympy.assumptions import Q\n    >>> rl = rewriterule(x + y, x**y, [x, y], assume=Q.integer(x))\n    >>> list(rl(z + 3))\n    [3**z]\n\n    Assumptions for the local context are provided at rule runtime\n\n    >>> list(rl(w + z, Q.integer(z)))\n    [z**w]\n    '

    def rewrite_rl(expr, assumptions=True):
        if False:
            while True:
                i = 10
        for match in unify(source, expr, {}, variables=variables):
            if condition and (not condition(*[match.get(var, var) for var in variables])):
                continue
            if assume and (not ask(assume.xreplace(match), assumptions)):
                continue
            expr2 = subs(match)(target)
            if isinstance(expr2, Expr):
                expr2 = rebuild(expr2)
            yield expr2
    return rewrite_rl