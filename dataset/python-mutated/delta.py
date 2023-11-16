"""
This module implements sums and products containing the Kronecker Delta function.

References
==========

.. [1] https://mathworld.wolfram.com/KroneckerDelta.html

"""
from .products import product
from .summations import Sum, summation
from sympy.core import Add, Mul, S, Dummy
from sympy.core.cache import cacheit
from sympy.core.sorting import default_sort_key
from sympy.functions import KroneckerDelta, Piecewise, piecewise_fold
from sympy.polys.polytools import factor
from sympy.sets.sets import Interval
from sympy.solvers.solvers import solve

@cacheit
def _expand_delta(expr, index):
    if False:
        for i in range(10):
            print('nop')
    '\n    Expand the first Add containing a simple KroneckerDelta.\n    '
    if not expr.is_Mul:
        return expr
    delta = None
    func = Add
    terms = [S.One]
    for h in expr.args:
        if delta is None and h.is_Add and _has_simple_delta(h, index):
            delta = True
            func = h.func
            terms = [terms[0] * t for t in h.args]
        else:
            terms = [t * h for t in terms]
    return func(*terms)

@cacheit
def _extract_delta(expr, index):
    if False:
        print('Hello World!')
    '\n    Extract a simple KroneckerDelta from the expression.\n\n    Explanation\n    ===========\n\n    Returns the tuple ``(delta, newexpr)`` where:\n\n      - ``delta`` is a simple KroneckerDelta expression if one was found,\n        or ``None`` if no simple KroneckerDelta expression was found.\n\n      - ``newexpr`` is a Mul containing the remaining terms; ``expr`` is\n        returned unchanged if no simple KroneckerDelta expression was found.\n\n    Examples\n    ========\n\n    >>> from sympy import KroneckerDelta\n    >>> from sympy.concrete.delta import _extract_delta\n    >>> from sympy.abc import x, y, i, j, k\n    >>> _extract_delta(4*x*y*KroneckerDelta(i, j), i)\n    (KroneckerDelta(i, j), 4*x*y)\n    >>> _extract_delta(4*x*y*KroneckerDelta(i, j), k)\n    (None, 4*x*y*KroneckerDelta(i, j))\n\n    See Also\n    ========\n\n    sympy.functions.special.tensor_functions.KroneckerDelta\n    deltaproduct\n    deltasummation\n    '
    if not _has_simple_delta(expr, index):
        return (None, expr)
    if isinstance(expr, KroneckerDelta):
        return (expr, S.One)
    if not expr.is_Mul:
        raise ValueError('Incorrect expr')
    delta = None
    terms = []
    for arg in expr.args:
        if delta is None and _is_simple_delta(arg, index):
            delta = arg
        else:
            terms.append(arg)
    return (delta, expr.func(*terms))

@cacheit
def _has_simple_delta(expr, index):
    if False:
        return 10
    '\n    Returns True if ``expr`` is an expression that contains a KroneckerDelta\n    that is simple in the index ``index``, meaning that this KroneckerDelta\n    is nonzero for a single value of the index ``index``.\n    '
    if expr.has(KroneckerDelta):
        if _is_simple_delta(expr, index):
            return True
        if expr.is_Add or expr.is_Mul:
            for arg in expr.args:
                if _has_simple_delta(arg, index):
                    return True
    return False

@cacheit
def _is_simple_delta(delta, index):
    if False:
        i = 10
        return i + 15
    '\n    Returns True if ``delta`` is a KroneckerDelta and is nonzero for a single\n    value of the index ``index``.\n    '
    if isinstance(delta, KroneckerDelta) and delta.has(index):
        p = (delta.args[0] - delta.args[1]).as_poly(index)
        if p:
            return p.degree() == 1
    return False

@cacheit
def _remove_multiple_delta(expr):
    if False:
        while True:
            i = 10
    "\n    Evaluate products of KroneckerDelta's.\n    "
    if expr.is_Add:
        return expr.func(*list(map(_remove_multiple_delta, expr.args)))
    if not expr.is_Mul:
        return expr
    eqs = []
    newargs = []
    for arg in expr.args:
        if isinstance(arg, KroneckerDelta):
            eqs.append(arg.args[0] - arg.args[1])
        else:
            newargs.append(arg)
    if not eqs:
        return expr
    solns = solve(eqs, dict=True)
    if len(solns) == 0:
        return S.Zero
    elif len(solns) == 1:
        for key in solns[0].keys():
            newargs.append(KroneckerDelta(key, solns[0][key]))
        expr2 = expr.func(*newargs)
        if expr != expr2:
            return _remove_multiple_delta(expr2)
    return expr

@cacheit
def _simplify_delta(expr):
    if False:
        return 10
    "\n    Rewrite a KroneckerDelta's indices in its simplest form.\n    "
    if isinstance(expr, KroneckerDelta):
        try:
            slns = solve(expr.args[0] - expr.args[1], dict=True)
            if slns and len(slns) == 1:
                return Mul(*[KroneckerDelta(*(key, value)) for (key, value) in slns[0].items()])
        except NotImplementedError:
            pass
    return expr

@cacheit
def deltaproduct(f, limit):
    if False:
        for i in range(10):
            print('nop')
    '\n    Handle products containing a KroneckerDelta.\n\n    See Also\n    ========\n\n    deltasummation\n    sympy.functions.special.tensor_functions.KroneckerDelta\n    sympy.concrete.products.product\n    '
    if (limit[2] - limit[1] < 0) == True:
        return S.One
    if not f.has(KroneckerDelta):
        return product(f, limit)
    if f.is_Add:
        delta = None
        terms = []
        for arg in sorted(f.args, key=default_sort_key):
            if delta is None and _has_simple_delta(arg, limit[0]):
                delta = arg
            else:
                terms.append(arg)
        newexpr = f.func(*terms)
        k = Dummy('kprime', integer=True)
        if isinstance(limit[1], int) and isinstance(limit[2], int):
            result = deltaproduct(newexpr, limit) + sum([deltaproduct(newexpr, (limit[0], limit[1], ik - 1)) * delta.subs(limit[0], ik) * deltaproduct(newexpr, (limit[0], ik + 1, limit[2])) for ik in range(int(limit[1]), int(limit[2] + 1))])
        else:
            result = deltaproduct(newexpr, limit) + deltasummation(deltaproduct(newexpr, (limit[0], limit[1], k - 1)) * delta.subs(limit[0], k) * deltaproduct(newexpr, (limit[0], k + 1, limit[2])), (k, limit[1], limit[2]), no_piecewise=_has_simple_delta(newexpr, limit[0]))
        return _remove_multiple_delta(result)
    (delta, _) = _extract_delta(f, limit[0])
    if not delta:
        g = _expand_delta(f, limit[0])
        if f != g:
            try:
                return factor(deltaproduct(g, limit))
            except AssertionError:
                return deltaproduct(g, limit)
        return product(f, limit)
    return _remove_multiple_delta(f.subs(limit[0], limit[1]) * KroneckerDelta(limit[2], limit[1])) + S.One * _simplify_delta(KroneckerDelta(limit[2], limit[1] - 1))

@cacheit
def deltasummation(f, limit, no_piecewise=False):
    if False:
        i = 10
        return i + 15
    "\n    Handle summations containing a KroneckerDelta.\n\n    Explanation\n    ===========\n\n    The idea for summation is the following:\n\n    - If we are dealing with a KroneckerDelta expression, i.e. KroneckerDelta(g(x), j),\n      we try to simplify it.\n\n      If we could simplify it, then we sum the resulting expression.\n      We already know we can sum a simplified expression, because only\n      simple KroneckerDelta expressions are involved.\n\n      If we could not simplify it, there are two cases:\n\n      1) The expression is a simple expression: we return the summation,\n         taking care if we are dealing with a Derivative or with a proper\n         KroneckerDelta.\n\n      2) The expression is not simple (i.e. KroneckerDelta(cos(x))): we can do\n         nothing at all.\n\n    - If the expr is a multiplication expr having a KroneckerDelta term:\n\n      First we expand it.\n\n      If the expansion did work, then we try to sum the expansion.\n\n      If not, we try to extract a simple KroneckerDelta term, then we have two\n      cases:\n\n      1) We have a simple KroneckerDelta term, so we return the summation.\n\n      2) We did not have a simple term, but we do have an expression with\n         simplified KroneckerDelta terms, so we sum this expression.\n\n    Examples\n    ========\n\n    >>> from sympy import oo, symbols\n    >>> from sympy.abc import k\n    >>> i, j = symbols('i, j', integer=True, finite=True)\n    >>> from sympy.concrete.delta import deltasummation\n    >>> from sympy import KroneckerDelta\n    >>> deltasummation(KroneckerDelta(i, k), (k, -oo, oo))\n    1\n    >>> deltasummation(KroneckerDelta(i, k), (k, 0, oo))\n    Piecewise((1, i >= 0), (0, True))\n    >>> deltasummation(KroneckerDelta(i, k), (k, 1, 3))\n    Piecewise((1, (i >= 1) & (i <= 3)), (0, True))\n    >>> deltasummation(k*KroneckerDelta(i, j)*KroneckerDelta(j, k), (k, -oo, oo))\n    j*KroneckerDelta(i, j)\n    >>> deltasummation(j*KroneckerDelta(i, j), (j, -oo, oo))\n    i\n    >>> deltasummation(i*KroneckerDelta(i, j), (i, -oo, oo))\n    j\n\n    See Also\n    ========\n\n    deltaproduct\n    sympy.functions.special.tensor_functions.KroneckerDelta\n    sympy.concrete.sums.summation\n    "
    if (limit[2] - limit[1] < 0) == True:
        return S.Zero
    if not f.has(KroneckerDelta):
        return summation(f, limit)
    x = limit[0]
    g = _expand_delta(f, x)
    if g.is_Add:
        return piecewise_fold(g.func(*[deltasummation(h, limit, no_piecewise) for h in g.args]))
    (delta, expr) = _extract_delta(g, x)
    if delta is not None and delta.delta_range is not None:
        (dinf, dsup) = delta.delta_range
        if (limit[1] - dinf <= 0) == True and (limit[2] - dsup >= 0) == True:
            no_piecewise = True
    if not delta:
        return summation(f, limit)
    solns = solve(delta.args[0] - delta.args[1], x)
    if len(solns) == 0:
        return S.Zero
    elif len(solns) != 1:
        return Sum(f, limit)
    value = solns[0]
    if no_piecewise:
        return expr.subs(x, value)
    return Piecewise((expr.subs(x, value), Interval(*limit[1:3]).as_relational(value)), (S.Zero, True))