"""Module with functions operating on IndexedBase, Indexed and Idx objects

    - Check shape conformance
    - Determine indices in resulting expression

    etc.

    Methods in this module could be implemented by calling methods on Expr
    objects instead.  When things stabilize this could be a useful
    refactoring.
"""
from functools import reduce
from sympy.core.function import Function
from sympy.functions import exp, Piecewise
from sympy.tensor.indexed import Idx, Indexed
from sympy.utilities import sift
from collections import OrderedDict

class IndexConformanceException(Exception):
    pass

def _unique_and_repeated(inds):
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns the unique and repeated indices. Also note, from the examples given below\n    that the order of indices is maintained as given in the input.\n\n    Examples\n    ========\n\n    >>> from sympy.tensor.index_methods import _unique_and_repeated\n    >>> _unique_and_repeated([2, 3, 1, 3, 0, 4, 0])\n    ([2, 1, 4], [3, 0])\n    '
    uniq = OrderedDict()
    for i in inds:
        if i in uniq:
            uniq[i] = 0
        else:
            uniq[i] = 1
    return sift(uniq, lambda x: uniq[x], binary=True)

def _remove_repeated(inds):
    if False:
        for i in range(10):
            print('nop')
    '\n    Removes repeated objects from sequences\n\n    Returns a set of the unique objects and a tuple of all that have been\n    removed.\n\n    Examples\n    ========\n\n    >>> from sympy.tensor.index_methods import _remove_repeated\n    >>> l1 = [1, 2, 3, 2]\n    >>> _remove_repeated(l1)\n    ({1, 3}, (2,))\n\n    '
    (u, r) = _unique_and_repeated(inds)
    return (set(u), tuple(r))

def _get_indices_Mul(expr, return_dummies=False):
    if False:
        print('Hello World!')
    "Determine the outer indices of a Mul object.\n\n    Examples\n    ========\n\n    >>> from sympy.tensor.index_methods import _get_indices_Mul\n    >>> from sympy.tensor.indexed import IndexedBase, Idx\n    >>> i, j, k = map(Idx, ['i', 'j', 'k'])\n    >>> x = IndexedBase('x')\n    >>> y = IndexedBase('y')\n    >>> _get_indices_Mul(x[i, k]*y[j, k])\n    ({i, j}, {})\n    >>> _get_indices_Mul(x[i, k]*y[j, k], return_dummies=True)\n    ({i, j}, {}, (k,))\n\n    "
    inds = list(map(get_indices, expr.args))
    (inds, syms) = list(zip(*inds))
    inds = list(map(list, inds))
    inds = list(reduce(lambda x, y: x + y, inds))
    (inds, dummies) = _remove_repeated(inds)
    symmetry = {}
    for s in syms:
        for pair in s:
            if pair in symmetry:
                symmetry[pair] *= s[pair]
            else:
                symmetry[pair] = s[pair]
    if return_dummies:
        return (inds, symmetry, dummies)
    else:
        return (inds, symmetry)

def _get_indices_Pow(expr):
    if False:
        for i in range(10):
            print('nop')
    "Determine outer indices of a power or an exponential.\n\n    A power is considered a universal function, so that the indices of a Pow is\n    just the collection of indices present in the expression.  This may be\n    viewed as a bit inconsistent in the special case:\n\n        x[i]**2 = x[i]*x[i]                                                      (1)\n\n    The above expression could have been interpreted as the contraction of x[i]\n    with itself, but we choose instead to interpret it as a function\n\n        lambda y: y**2\n\n    applied to each element of x (a universal function in numpy terms).  In\n    order to allow an interpretation of (1) as a contraction, we need\n    contravariant and covariant Idx subclasses.  (FIXME: this is not yet\n    implemented)\n\n    Expressions in the base or exponent are subject to contraction as usual,\n    but an index that is present in the exponent, will not be considered\n    contractable with its own base.  Note however, that indices in the same\n    exponent can be contracted with each other.\n\n    Examples\n    ========\n\n    >>> from sympy.tensor.index_methods import _get_indices_Pow\n    >>> from sympy import Pow, exp, IndexedBase, Idx\n    >>> A = IndexedBase('A')\n    >>> x = IndexedBase('x')\n    >>> i, j, k = map(Idx, ['i', 'j', 'k'])\n    >>> _get_indices_Pow(exp(A[i, j]*x[j]))\n    ({i}, {})\n    >>> _get_indices_Pow(Pow(x[i], x[i]))\n    ({i}, {})\n    >>> _get_indices_Pow(Pow(A[i, j]*x[j], x[i]))\n    ({i}, {})\n\n    "
    (base, exp) = expr.as_base_exp()
    (binds, bsyms) = get_indices(base)
    (einds, esyms) = get_indices(exp)
    inds = binds | einds
    symmetries = {}
    return (inds, symmetries)

def _get_indices_Add(expr):
    if False:
        while True:
            i = 10
    "Determine outer indices of an Add object.\n\n    In a sum, each term must have the same set of outer indices.  A valid\n    expression could be\n\n        x(i)*y(j) - x(j)*y(i)\n\n    But we do not allow expressions like:\n\n        x(i)*y(j) - z(j)*z(j)\n\n    FIXME: Add support for Numpy broadcasting\n\n    Examples\n    ========\n\n    >>> from sympy.tensor.index_methods import _get_indices_Add\n    >>> from sympy.tensor.indexed import IndexedBase, Idx\n    >>> i, j, k = map(Idx, ['i', 'j', 'k'])\n    >>> x = IndexedBase('x')\n    >>> y = IndexedBase('y')\n    >>> _get_indices_Add(x[i] + x[k]*y[i, k])\n    ({i}, {})\n\n    "
    inds = list(map(get_indices, expr.args))
    (inds, syms) = list(zip(*inds))
    non_scalars = [x for x in inds if x != set()]
    if not non_scalars:
        return (set(), {})
    if not all((x == non_scalars[0] for x in non_scalars[1:])):
        raise IndexConformanceException('Indices are not consistent: %s' % expr)
    if not reduce(lambda x, y: x != y or y, syms):
        symmetries = syms[0]
    else:
        symmetries = {}
    return (non_scalars[0], symmetries)

def get_indices(expr):
    if False:
        for i in range(10):
            print('nop')
    "Determine the outer indices of expression ``expr``\n\n    By *outer* we mean indices that are not summation indices.  Returns a set\n    and a dict.  The set contains outer indices and the dict contains\n    information about index symmetries.\n\n    Examples\n    ========\n\n    >>> from sympy.tensor.index_methods import get_indices\n    >>> from sympy import symbols\n    >>> from sympy.tensor import IndexedBase\n    >>> x, y, A = map(IndexedBase, ['x', 'y', 'A'])\n    >>> i, j, a, z = symbols('i j a z', integer=True)\n\n    The indices of the total expression is determined, Repeated indices imply a\n    summation, for instance the trace of a matrix A:\n\n    >>> get_indices(A[i, i])\n    (set(), {})\n\n    In the case of many terms, the terms are required to have identical\n    outer indices.  Else an IndexConformanceException is raised.\n\n    >>> get_indices(x[i] + A[i, j]*y[j])\n    ({i}, {})\n\n    :Exceptions:\n\n    An IndexConformanceException means that the terms ar not compatible, e.g.\n\n    >>> get_indices(x[i] + y[j])                #doctest: +SKIP\n            (...)\n    IndexConformanceException: Indices are not consistent: x(i) + y(j)\n\n    .. warning::\n       The concept of *outer* indices applies recursively, starting on the deepest\n       level.  This implies that dummies inside parenthesis are assumed to be\n       summed first, so that the following expression is handled gracefully:\n\n       >>> get_indices((x[i] + A[i, j]*y[j])*x[j])\n       ({i, j}, {})\n\n       This is correct and may appear convenient, but you need to be careful\n       with this as SymPy will happily .expand() the product, if requested.  The\n       resulting expression would mix the outer ``j`` with the dummies inside\n       the parenthesis, which makes it a different expression.  To be on the\n       safe side, it is best to avoid such ambiguities by using unique indices\n       for all contractions that should be held separate.\n\n    "
    if isinstance(expr, Indexed):
        c = expr.indices
        (inds, dummies) = _remove_repeated(c)
        return (inds, {})
    elif expr is None:
        return (set(), {})
    elif isinstance(expr, Idx):
        return ({expr}, {})
    elif expr.is_Atom:
        return (set(), {})
    else:
        if expr.is_Mul:
            return _get_indices_Mul(expr)
        elif expr.is_Add:
            return _get_indices_Add(expr)
        elif expr.is_Pow or isinstance(expr, exp):
            return _get_indices_Pow(expr)
        elif isinstance(expr, Piecewise):
            return (set(), {})
        elif isinstance(expr, Function):
            ind0 = set()
            for arg in expr.args:
                (ind, sym) = get_indices(arg)
                ind0 |= ind
            return (ind0, sym)
        elif not expr.has(Indexed):
            return (set(), {})
        raise NotImplementedError('FIXME: No specialized handling of type %s' % type(expr))

def get_contraction_structure(expr):
    if False:
        for i in range(10):
            print('nop')
    'Determine dummy indices of ``expr`` and describe its structure\n\n    By *dummy* we mean indices that are summation indices.\n\n    The structure of the expression is determined and described as follows:\n\n    1) A conforming summation of Indexed objects is described with a dict where\n       the keys are summation indices and the corresponding values are sets\n       containing all terms for which the summation applies.  All Add objects\n       in the SymPy expression tree are described like this.\n\n    2) For all nodes in the SymPy expression tree that are *not* of type Add, the\n       following applies:\n\n       If a node discovers contractions in one of its arguments, the node\n       itself will be stored as a key in the dict.  For that key, the\n       corresponding value is a list of dicts, each of which is the result of a\n       recursive call to get_contraction_structure().  The list contains only\n       dicts for the non-trivial deeper contractions, omitting dicts with None\n       as the one and only key.\n\n    .. Note:: The presence of expressions among the dictionary keys indicates\n       multiple levels of index contractions.  A nested dict displays nested\n       contractions and may itself contain dicts from a deeper level.  In\n       practical calculations the summation in the deepest nested level must be\n       calculated first so that the outer expression can access the resulting\n       indexed object.\n\n    Examples\n    ========\n\n    >>> from sympy.tensor.index_methods import get_contraction_structure\n    >>> from sympy import default_sort_key\n    >>> from sympy.tensor import IndexedBase, Idx\n    >>> x, y, A = map(IndexedBase, [\'x\', \'y\', \'A\'])\n    >>> i, j, k, l = map(Idx, [\'i\', \'j\', \'k\', \'l\'])\n    >>> get_contraction_structure(x[i]*y[i] + A[j, j])\n    {(i,): {x[i]*y[i]}, (j,): {A[j, j]}}\n    >>> get_contraction_structure(x[i]*y[j])\n    {None: {x[i]*y[j]}}\n\n    A multiplication of contracted factors results in nested dicts representing\n    the internal contractions.\n\n    >>> d = get_contraction_structure(x[i, i]*y[j, j])\n    >>> sorted(d.keys(), key=default_sort_key)\n    [None, x[i, i]*y[j, j]]\n\n    In this case, the product has no contractions:\n\n    >>> d[None]\n    {x[i, i]*y[j, j]}\n\n    Factors are contracted "first":\n\n    >>> sorted(d[x[i, i]*y[j, j]], key=default_sort_key)\n    [{(i,): {x[i, i]}}, {(j,): {y[j, j]}}]\n\n    A parenthesized Add object is also returned as a nested dictionary.  The\n    term containing the parenthesis is a Mul with a contraction among the\n    arguments, so it will be found as a key in the result.  It stores the\n    dictionary resulting from a recursive call on the Add expression.\n\n    >>> d = get_contraction_structure(x[i]*(y[i] + A[i, j]*x[j]))\n    >>> sorted(d.keys(), key=default_sort_key)\n    [(A[i, j]*x[j] + y[i])*x[i], (i,)]\n    >>> d[(i,)]\n    {(A[i, j]*x[j] + y[i])*x[i]}\n    >>> d[x[i]*(A[i, j]*x[j] + y[i])]\n    [{None: {y[i]}, (j,): {A[i, j]*x[j]}}]\n\n    Powers with contractions in either base or exponent will also be found as\n    keys in the dictionary, mapping to a list of results from recursive calls:\n\n    >>> d = get_contraction_structure(A[j, j]**A[i, i])\n    >>> d[None]\n    {A[j, j]**A[i, i]}\n    >>> nested_contractions = d[A[j, j]**A[i, i]]\n    >>> nested_contractions[0]\n    {(j,): {A[j, j]}}\n    >>> nested_contractions[1]\n    {(i,): {A[i, i]}}\n\n    The description of the contraction structure may appear complicated when\n    represented with a string in the above examples, but it is easy to iterate\n    over:\n\n    >>> from sympy import Expr\n    >>> for key in d:\n    ...     if isinstance(key, Expr):\n    ...         continue\n    ...     for term in d[key]:\n    ...         if term in d:\n    ...             # treat deepest contraction first\n    ...             pass\n    ...     # treat outermost contactions here\n\n    '
    if isinstance(expr, Indexed):
        (junk, key) = _remove_repeated(expr.indices)
        return {key or None: {expr}}
    elif expr.is_Atom:
        return {None: {expr}}
    elif expr.is_Mul:
        (junk, junk, key) = _get_indices_Mul(expr, return_dummies=True)
        result = {key or None: {expr}}
        nested = []
        for fac in expr.args:
            facd = get_contraction_structure(fac)
            if not (None in facd and len(facd) == 1):
                nested.append(facd)
        if nested:
            result[expr] = nested
        return result
    elif expr.is_Pow or isinstance(expr, exp):
        (b, e) = expr.as_base_exp()
        dbase = get_contraction_structure(b)
        dexp = get_contraction_structure(e)
        dicts = []
        for d in (dbase, dexp):
            if not (None in d and len(d) == 1):
                dicts.append(d)
        result = {None: {expr}}
        if dicts:
            result[expr] = dicts
        return result
    elif expr.is_Add:
        result = {}
        for term in expr.args:
            d = get_contraction_structure(term)
            for key in d:
                if key in result:
                    result[key] |= d[key]
                else:
                    result[key] = d[key]
        return result
    elif isinstance(expr, Piecewise):
        return {None: expr}
    elif isinstance(expr, Function):
        deeplist = []
        for arg in expr.args:
            deep = get_contraction_structure(arg)
            if not (None in deep and len(deep) == 1):
                deeplist.append(deep)
        d = {None: {expr}}
        if deeplist:
            d[expr] = deeplist
        return d
    elif not expr.has(Indexed):
        return {None: {expr}}
    raise NotImplementedError('FIXME: No specialized handling of type %s' % type(expr))