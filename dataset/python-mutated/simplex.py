"""Tools for optimizing a linear function for a given simplex.

For the linear objective function ``f`` with linear constraints
expressed using `Le`, `Ge` or `Eq` can be found with ``lpmin`` or
``lpmax``. The symbols are **unbounded** unless specifically
constrained.

As an alternative, the matrices describing the objective and the
constraints, and an optional list of bounds can be passed to
``linprog`` which will solve for the minimization of ``C*x``
under constraints ``A*x <= b`` and/or ``Aeq*x = beq``, and
individual bounds for variables given as ``(lo, hi)``. The values
returned are **nonnegative** unless bounds are provided that
indicate otherwise.

Errors that might be raised are UnboundedLPError when there is no
finite solution for the system or InfeasibleLPError when the
constraints represent impossible conditions (i.e. a non-existant
 simplex).

Here is a simple 1-D system: minimize `x` given that ``x >= 1``.

    >>> from sympy.solvers.simplex import lpmin, linprog
    >>> from sympy.abc import x

    The function and a list with the constraint is passed directly
    to `lpmin`:

    >>> lpmin(x, [x >= 1])
    (1, {x: 1})

    For `linprog` the matrix for the objective is `[1]` and the
    uivariate constraint can be passed as a bound with None acting
    as infinity:

    >>> linprog([1], bounds=(1, None))
    (1, [1])

    Or the matrices, corresponding to ``x >= 1`` expressed as
    ``-x <= -1`` as required by the routine, can be passed:

    >>> linprog([1], [-1], [-1])
    (1, [1])

    If there is no limit for the objective, an error is raised.
    In this case there is a valid region of interest (simplex)
    but no limit to how small ``x`` can be:

    >>> lpmin(x, [])
    Traceback (most recent call last):
    ...
    sympy.solvers.simplex.UnboundedLPError:
    Objective function can assume arbitrarily large values!

    An error is raised if there is no possible solution:

    >>> lpmin(x,[x<=1,x>=2])
    Traceback (most recent call last):
    ...
    sympy.solvers.simplex.InfeasibleLPError:
    Inconsistent/False constraint
"""
from sympy.core import sympify
from sympy.core.exprtools import factor_terms
from sympy.core.relational import Le, Ge, Eq
from sympy.core.singleton import S
from sympy.core.symbol import Dummy
from sympy.core.sorting import ordered
from sympy.functions.elementary.complexes import sign
from sympy.matrices.dense import Matrix, zeros
from sympy.solvers.solveset import linear_eq_to_matrix
from sympy.utilities.iterables import numbered_symbols
from sympy.utilities.misc import filldedent

class UnboundedLPError(Exception):
    """
    A linear programing problem is said to be unbounded if its objective
    function can assume arbitrarily large values.

    Example
    =======

    Suppose you want to maximize
        2x
    subject to
        x >= 0

    There's no upper limit that 2x can take.
    """
    pass

class InfeasibleLPError(Exception):
    """
    A linear programing problem is considered infeasible if its
    constraint set is empty. That is, if the set of all vectors
    satisfying the contraints is empty, then the problem is infeasible.

    Example
    =======

    Suppose you want to maximize
        x
    subject to
        x >= 10
        x <= 9

    No x can satisfy those constraints.
    """
    pass

def _pivot(M, i, j):
    if False:
        for i in range(10):
            print('nop')
    "\n    The pivot element `M[i, j]` is inverted and the rest of the matrix\n    modified and returned as a new matrix; original is left unmodified.\n\n    Example\n    =======\n\n    >>> from sympy.matrices.dense import Matrix\n    >>> from sympy.solvers.simplex import _pivot\n    >>> from sympy import var\n    >>> Matrix(3, 3, var('a:i'))\n    Matrix([\n    [a, b, c],\n    [d, e, f],\n    [g, h, i]])\n    >>> _pivot(_, 1, 0)\n    Matrix([\n    [-a/d, -a*e/d + b, -a*f/d + c],\n    [ 1/d,        e/d,        f/d],\n    [-g/d,  h - e*g/d,  i - f*g/d]])\n    "
    (Mi, Mj, Mij) = (M[i, :], M[:, j], M[i, j])
    if Mij == 0:
        raise ZeroDivisionError('Tried to pivot about zero-valued entry.')
    A = M - Mj * (Mi / Mij)
    A[i, :] = Mi / Mij
    A[:, j] = -Mj / Mij
    A[i, j] = 1 / Mij
    return A

def _choose_pivot_row(A, B, candidate_rows, pivot_col, Y):
    if False:
        for i in range(10):
            print('nop')
    first_row = candidate_rows[0]
    min_ratio = B[first_row] / A[first_row, pivot_col]
    min_rows = [first_row]
    for i in candidate_rows[1:]:
        ratio = B[i] / A[i, pivot_col]
        if ratio < min_ratio:
            min_ratio = ratio
            min_rows = [i]
        elif ratio == min_ratio:
            min_rows.append(i)
    (_, row) = min(((Y[i], i) for i in min_rows))
    return row

def _simplex(A, B, C, D=None, dual=False):
    if False:
        i = 10
        return i + 15
    'Return ``(o, x, y)`` obtained from the two-phase simplex method\n    using Bland\'s rule: ``o`` is the minimum value of primal,\n    ``Cx - D``, under constraints ``Ax <= B`` (with ``x >= 0``) and\n    the maximum of the dual, ``y^{T}B - D``, under constraints\n    ``A^{T}*y >= C^{T}`` (with ``y >= 0``). To compute the dual of\n    the system, pass `dual=True` and ``(o, y, x)`` will be returned.\n\n    Note: the nonnegative constraints for ``x`` and ``y`` supercede\n    any values of ``A`` and ``B`` that are inconsistent with that\n    assumption, so if a constraint of ``x >= -1`` is represented\n    in ``A`` and ``B``, no value will be obtained that is negative; if\n    a constraint of ``x <= -1`` is represented, an error will be\n    raised since no solution is possible.\n\n    This routine relies on the ability of determining whether an\n    expression is 0 or not. This is guaranteed if the input contains\n    only Float or Rational entries. It will raise a TypeError if\n    a relationship does not evaluate to True or False.\n\n    Examples\n    ========\n\n    >>> from sympy.solvers.simplex import _simplex\n    >>> from sympy import Matrix\n\n    Consider the simple minimization of ``f = x + y + 1`` under the\n    constraint that ``y + 2*x >= 4``. This is the "standard form" of\n    a minimization.\n\n    In the nonnegative quadrant, this inequality describes a area above\n    a triangle with vertices at (0, 4), (0, 0) and (2, 0). The minimum\n    of ``f`` occurs at (2, 0). Define A, B, C, D for the standard\n    minimization:\n\n    >>> A = Matrix([[2, 1]])\n    >>> B = Matrix([4])\n    >>> C = Matrix([[1, 1]])\n    >>> D = Matrix([-1])\n\n    Confirm that this is the system of interest:\n\n    >>> from sympy.abc import x, y\n    >>> X = Matrix([x, y])\n    >>> (C*X - D)[0]\n    x + y + 1\n    >>> [i >= j for i, j in zip(A*X, B)]\n    [2*x + y >= 4]\n\n    Since `_simplex` will do a minimization for constraints given as\n    ``A*x <= B``, the signs of ``A`` and ``B`` must be negated since\n    the currently correspond to a greater-than inequality:\n\n    >>> _simplex(-A, -B, C, D)\n    (3, [2, 0], [1/2])\n\n    The dual of minimizing ``f`` is maximizing ``F = c*y - d`` for\n    ``a*y <= b`` where ``a``, ``b``, ``c``, ``d`` are derived from the\n    transpose of the matrix representation of the standard minimization:\n\n    >>> tr = lambda a, b, c, d: [i.T for i in (a, c, b, d)]\n    >>> a, b, c, d = tr(A, B, C, D)\n\n    This time ``a*x <= b`` is the expected inequality for the `_simplex`\n    method, but to maximize ``F``, the sign of ``c`` and ``d`` must be\n    changed (so that minimizing the negative will give the negative of\n    the maximum of ``F``):\n\n    >>> _simplex(a, b, -c, -d)\n    (-3, [1/2], [2, 0])\n\n    The negative of ``F`` and the min of ``f`` are the same. The dual\n    point `[1/2]` is the value of ``y`` that minimized ``F = c*y - d``\n    under constraints a*x <= b``:\n\n    >>> y = Matrix([\'y\'])\n    >>> (c*y - d)[0]\n    4*y + 1\n    >>> [i <= j for i, j in zip(a*y,b)]\n    [2*y <= 1, y <= 1]\n\n    In this 1-dimensional dual system, the more restrictive contraint is\n    the first which limits ``y`` between 0 and 1/2 and the maximum of\n    ``F`` is attained at the nonzero value, hence is ``4*(1/2) + 1 = 3``.\n\n    In this case the values for ``x`` and ``y`` were the same when the\n    dual representation was solved. This is not always the case (though\n    the value of the function will be the same).\n\n    >>> l = [[1, 1], [-1, 1], [0, 1], [-1, 0]], [5, 1, 2, -1], [[1, 1]], [-1]\n    >>> A, B, C, D = [Matrix(i) for i in l]\n    >>> _simplex(A, B, -C, -D)\n    (-6, [3, 2], [1, 0, 0, 0])\n    >>> _simplex(A, B, -C, -D, dual=True)  # [5, 0] != [3, 2]\n    (-6, [1, 0, 0, 0], [5, 0])\n\n    In both cases the function has the same value:\n\n    >>> Matrix(C)*Matrix([3, 2]) == Matrix(C)*Matrix([5, 0])\n    True\n\n    See Also\n    ========\n    _lp - poses min/max problem in form compatible with _simplex\n    lpmin - minimization which calls _lp\n    lpmax - maximimzation which calls _lp\n\n    References\n    ==========\n\n    .. [1] Thomas S. Ferguson, LINEAR PROGRAMMING: A Concise Introduction\n           web.tecnico.ulisboa.pt/mcasquilho/acad/or/ftp/FergusonUCLA_lp.pdf\n\n    '
    (A, B, C, D) = [Matrix(i) for i in (A, B, C, D or [0])]
    if dual:
        (_o, d, p) = _simplex(-A.T, C.T, B.T, -D)
        return (-_o, d, p)
    if A and B:
        M = Matrix([[A, B], [C, D]])
    else:
        if A or B:
            raise ValueError('must give A and B')
        M = Matrix([[C, D]])
    n = M.cols - 1
    m = M.rows - 1
    if not all((i.is_Float or i.is_Rational for i in M)):
        raise TypeError(filldedent('\n            Only rationals and floats are allowed.\n            '))
    X = [(False, j) for j in range(n)]
    Y = [(True, i) for i in range(m)]
    last = None
    while True:
        B = M[:-1, -1]
        A = M[:-1, :-1]
        if all((B[i] >= 0 for i in range(B.rows))):
            break
        for k in range(B.rows):
            if B[k] < 0:
                break
        else:
            pass
        piv_cols = [_ for _ in range(A.cols) if A[k, _] < 0]
        if not piv_cols:
            raise InfeasibleLPError(filldedent('\n                The constraint set is empty!'))
        (_, c) = min(((X[i], i) for i in piv_cols))
        piv_rows = [_ for _ in range(A.rows) if A[_, c] > 0 and B[_] > 0]
        piv_rows.append(k)
        r = _choose_pivot_row(A, B, piv_rows, c, Y)
        if (r, c) == last:
            last = True
            break
        last = (r, c)
        M = _pivot(M, r, c)
        (X[c], Y[r]) = (Y[r], X[c])
    while True:
        B = M[:-1, -1]
        A = M[:-1, :-1]
        C = M[-1, :-1]
        piv_cols = []
        piv_cols = [_ for _ in range(n) if C[_] < 0]
        if not piv_cols:
            break
        (_, c) = min(((X[i], i) for i in piv_cols))
        piv_rows = [_ for _ in range(m) if A[_, c] > 0]
        if not piv_rows:
            raise UnboundedLPError(filldedent('\n                Objective function can assume\n                arbitrarily large values!'))
        r = _choose_pivot_row(A, B, piv_rows, c, Y)
        M = _pivot(M, r, c)
        (X[c], Y[r]) = (Y[r], X[c])
    argmax = [None] * n
    argmin_dual = [None] * m
    for (i, (v, n)) in enumerate(X):
        if v == False:
            argmax[n] = 0
        else:
            argmin_dual[n] = M[-1, i]
    for (i, (v, n)) in enumerate(Y):
        if v == True:
            argmin_dual[n] = 0
        else:
            argmax[n] = M[i, -1]
    if last and (not all((i >= 0 for i in argmax + argmin_dual))):
        raise InfeasibleLPError(filldedent('\n            Oscillating system led to invalid solution.\n            If you believe there was a valid solution, please\n            report this as a bug.'))
    return (-M[-1, -1], argmax, argmin_dual)

def _abcd(M, list=False):
    if False:
        print('Hello World!')
    'return parts of M as matrices or lists\n\n    Examples\n    ========\n\n    >>> from sympy import Matrix\n    >>> from sympy.solvers.simplex import _abcd\n\n    >>> m = Matrix(3, 3, range(9)); m\n    Matrix([\n    [0, 1, 2],\n    [3, 4, 5],\n    [6, 7, 8]])\n    >>> a, b, c, d = _abcd(m)\n    >>> a\n    Matrix([\n    [0, 1],\n    [3, 4]])\n    >>> b\n    Matrix([\n    [2],\n    [5]])\n    >>> c\n    Matrix([[6, 7]])\n    >>> d\n    Matrix([[8]])\n\n    The matrices can be returned as compact lists, too:\n\n    >>> L = a, b, c, d = _abcd(m, list=True); L\n    ([[0, 1], [3, 4]], [2, 5], [[6, 7]], [8])\n    '

    def aslist(i):
        if False:
            for i in range(10):
                print('nop')
        l = i.tolist()
        if len(l[0]) == 1:
            return [i[0] for i in l]
        return l
    m = (M[:-1, :-1], M[:-1, -1], M[-1, :-1], M[-1:, -1:])
    if not list:
        return m
    return tuple([aslist(i) for i in m])

def _m(a, b, c, d=None):
    if False:
        while True:
            i = 10
    'return Matrix([[a, b], [c, d]]) from matrices\n    in Matrix or list form.\n\n    Examples\n    ========\n\n    >>> from sympy import Matrix\n    >>> from sympy.solvers.simplex import _abcd, _m\n    >>> m = Matrix(3, 3, range(9))\n    >>> L = _abcd(m, list=True); L\n    ([[0, 1], [3, 4]], [2, 5], [[6, 7]], [8])\n    >>> _abcd(m)\n    (Matrix([\n    [0, 1],\n    [3, 4]]), Matrix([\n    [2],\n    [5]]), Matrix([[6, 7]]), Matrix([[8]]))\n    >>> assert m == _m(*L) == _m(*_)\n    '
    (a, b, c, d) = [Matrix(i) for i in (a, b, c, d or [0])]
    return Matrix([[a, b], [c, d]])

def _primal_dual(M, factor=True):
    if False:
        i = 10
        return i + 15
    'return primal and dual function and constraints\n    assuming that ``M = Matrix([[A, b], [c, d]])`` and the\n    function ``c*x - d`` is being minimized with ``Ax >= b``\n    for nonnegative values of ``x``. The dual and its\n    constraints will be for maximizing `b.T*y - d` subject\n    to ``A.T*y <= c.T``.\n\n    Examples\n    ========\n\n    >>> from sympy.solvers.simplex import _primal_dual, lpmin, lpmax\n    >>> from sympy import Matrix\n\n    The following matrix represents the primal task of\n    minimizing x + y + 7 for y >= x + 1 and y >= -2*x + 3.\n    The dual task seeks to maximize x + 3*y + 7 with\n    2*y - x <= 1 and and x + y <= 1:\n\n    >>> M = Matrix([\n    ...     [-1, 1,  1],\n    ...     [ 2, 1,  3],\n    ...     [ 1, 1, -7]])\n    >>> p, d = _primal_dual(M)\n\n    The minimum of the primal and maximum of the dual are the same\n    (though they occur at different points):\n\n    >>> lpmin(*p)\n    (28/3, {x1: 2/3, x2: 5/3})\n    >>> lpmax(*d)\n    (28/3, {y1: 1/3, y2: 2/3})\n\n    If the equivalent (but canonical) inequalities are\n    desired, leave `factor=True`, otherwise the unmodified\n    inequalities for M will be returned.\n\n    >>> m = Matrix([\n    ... [-3, -2,  4, -2],\n    ... [ 2,  0,  0, -2],\n    ... [ 0,  1, -3,  0]])\n\n    >>> _primal_dual(m, False)  # last condition is 2*x1 >= -2\n    ((x2 - 3*x3,\n        [-3*x1 - 2*x2 + 4*x3 >= -2, 2*x1 >= -2]),\n    (-2*y1 - 2*y2,\n        [-3*y1 + 2*y2 <= 0, -2*y1 <= 1, 4*y1 <= -3]))\n\n    >>> _primal_dual(m)  # condition now x1 >= -1\n    ((x2 - 3*x3,\n        [-3*x1 - 2*x2 + 4*x3 >= -2, x1 >= -1]),\n    (-2*y1 - 2*y2,\n        [-3*y1 + 2*y2 <= 0, -2*y1 <= 1, 4*y1 <= -3]))\n\n    If you pass the transpose of the matrix, the primal will be\n    identified as the standard minimization problem and the\n    dual as the standard maximization:\n\n    >>> _primal_dual(m.T)\n    ((-2*x1 - 2*x2,\n        [-3*x1 + 2*x2 >= 0, -2*x1 >= 1, 4*x1 >= -3]),\n    (y2 - 3*y3,\n        [-3*y1 - 2*y2 + 4*y3 <= -2, y1 <= -1]))\n\n    A matrix must have some size or else None will be returned for\n    the functions:\n\n    >>> _primal_dual(Matrix([[1, 2]]))\n    ((x1 - 2, []), (-2, []))\n\n    >>> _primal_dual(Matrix([]))\n    ((None, []), (None, []))\n\n    References\n    ==========\n\n    .. [1] David Galvin, Relations between Primal and Dual\n           www3.nd.edu/~dgalvin1/30210/30210_F07/presentations/dual_opt.pdf\n    '
    if not M:
        return ((None, []), (None, []))
    if not hasattr(M, 'shape'):
        if len(M) not in (3, 4):
            raise ValueError('expecting Matrix or 3 or 4 lists')
        M = _m(*M)
    (m, n) = [i - 1 for i in M.shape]
    (A, b, c, d) = _abcd(M)
    d = d[0]
    _ = lambda x: numbered_symbols(x, start=1)
    x = Matrix([i for (i, j) in zip(_('x'), range(n))])
    yT = Matrix([i for (i, j) in zip(_('y'), range(m))]).T

    def ineq(L, r, op):
        if False:
            i = 10
            return i + 15
        rv = []
        for r in (op(i, j) for (i, j) in zip(L, r)):
            if r == True:
                continue
            elif r == False:
                return [False]
            if factor:
                f = factor_terms(r)
                if f.lhs.is_Mul and f.rhs % f.lhs.args[0] == 0:
                    assert len(f.lhs.args) == 2, f.lhs
                    k = f.lhs.args[0]
                    r = r.func(sign(k) * f.lhs.args[1], f.rhs // abs(k))
            rv.append(r)
        return rv
    eq = lambda x, d: x[0] - d if x else -d
    F = eq(c * x, d)
    f = eq(yT * b, d)
    return ((F, ineq(A * x, b, Ge)), (f, ineq(yT * A, c, Le)))

def _rel_as_nonpos(constr, syms):
    if False:
        i = 10
        return i + 15
    'return `(np, d, aux)` where `np` is a list of nonpositive\n    expressions that represent the given constraints (possibly\n    rewritten in terms of auxilliary variables) expressible with\n    nonnegative symbols, and `d` is a dictionary mapping a given\n    symbols to an expression with an auxilliary variable. In some\n    cases a symbol will be used as part of the change of variables,\n    e.g. x: x - z1 instead of x: z1 - z2.\n\n    If any constraint is False/empty, return None. All variables in\n    ``constr`` are assumed to be unbounded unless explicitly indicated\n    otherwise with a univariate constraint, e.g. ``x >= 0`` will\n    restrict ``x`` to nonnegative values.\n\n    The ``syms`` must be included so all symbols can be given an\n    unbounded assumption if they are not otherwise bound with\n    univariate conditions like ``x <= 3``.\n\n    Examples\n    ========\n\n    >>> from sympy.solvers.simplex import _rel_as_nonpos\n    >>> from sympy.abc import x, y\n    >>> _rel_as_nonpos([x >= y, x >= 0, y >= 0], (x, y))\n    ([-x + y], {}, [])\n    >>> _rel_as_nonpos([x >= 3, x <= 5], [x])\n    ([_z1 - 2], {x: _z1 + 3}, [_z1])\n    >>> _rel_as_nonpos([x <= 5], [x])\n    ([], {x: 5 - _z1}, [_z1])\n    >>> _rel_as_nonpos([x >= 1], [x])\n    ([], {x: _z1 + 1}, [_z1])\n    '
    r = {}
    np = []
    aux = []
    ui = numbered_symbols('z', start=1, cls=Dummy)
    univariate = {}
    unbound = []
    syms = set(syms)
    for i in constr:
        if i == True:
            continue
        if i == False:
            return
        if i.has(S.Infinity, S.NegativeInfinity):
            raise ValueError('only finite bounds are permitted')
        if isinstance(i, (Le, Ge)):
            i = i.lts - i.gts
            freei = i.free_symbols
            if freei - syms:
                raise ValueError('unexpected symbol(s) in constraint: %s' % (freei - syms))
            if len(freei) > 1:
                np.append(i)
            elif freei:
                x = freei.pop()
                if x in unbound:
                    continue
                ivl = Le(i, 0, evaluate=False).as_set()
                if x not in univariate:
                    univariate[x] = ivl
                else:
                    univariate[x] &= ivl
            elif i:
                return False
        else:
            raise TypeError(filldedent('\n                only equalities like Eq(x, y) or non-strict\n                inequalities like x >= y are allowed in lp, not %s' % i))
    for x in syms:
        i = univariate.get(x, True)
        if not i:
            return None
        if i == True:
            unbound.append(x)
            continue
        (a, b) = (i.inf, i.sup)
        if a.is_infinite:
            u = next(ui)
            r[x] = b - u
            aux.append(u)
        elif b.is_infinite:
            if a:
                u = next(ui)
                r[x] = a + u
                aux.append(u)
            else:
                pass
        else:
            u = next(ui)
            aux.append(u)
            r[x] = u + a
            np.append(u - (b - a))
    for x in unbound:
        u = next(ui)
        r[x] = u - x
        aux.append(u)
    return (np, r, aux)

def _lp_matrices(objective, constraints):
    if False:
        return 10
    'return A, B, C, D, r, x+X, X for maximizing\n    objective = Cx - D with constraints Ax <= B, introducing\n    introducing auxilliary variables, X, as necessary to make\n    replacements of symbols as given in r, {xi: expression with Xj},\n    so all variables in x+X will take on nonnegative values.\n\n    Every univariate condition creates a semi-infinite\n    condition, e.g. a single ``x <= 3`` creates the\n    interval ``[-oo, 3]`` while ``x <= 3`` and ``x >= 2``\n    create an interval ``[2, 3]``. Variables not in a univariate\n    expression will take on nonnegative values.\n    '
    F = sympify(objective)
    np = [sympify(i) for i in constraints]
    syms = set.union(*[i.free_symbols for i in [F] + np], set())
    for i in range(len(np)):
        if isinstance(np[i], Eq):
            np[i] = np[i].lhs - np[i].rhs <= 0
            np.append(-np[i].lhs <= 0)
    _ = _rel_as_nonpos(np, syms)
    if _ is None:
        raise InfeasibleLPError(filldedent('\n            Inconsistent/False constraint'))
    (np, r, aux) = _
    F = F.xreplace(r)
    np = [i.xreplace(r) for i in np]
    xx = list(ordered(syms)) + aux
    (A, B) = linear_eq_to_matrix(np, xx)
    (C, D) = linear_eq_to_matrix([F], xx)
    return (A, B, C, D, r, xx, aux)

def _lp(min_max, f, constr):
    if False:
        print('Hello World!')
    "Return the optimization (min or max) of ``f`` with the given\n    constraints. All variables are unbounded unless constrained.\n\n    If `min_max` is 'max' then the results corresponding to the\n    maximization of ``f`` will be returned, else the minimization.\n    The constraints can be given as Le, Ge or Eq expressions.\n\n    Examples\n    ========\n\n    >>> from sympy.solvers.simplex import _lp as lp\n    >>> from sympy import Eq\n    >>> from sympy.abc import x, y, z\n    >>> f = x + y - 2*z\n    >>> c = [7*x + 4*y - 7*z <= 3, 3*x - y + 10*z <= 6]\n    >>> c += [i >= 0 for i in (x, y, z)]\n    >>> lp(min, f, c)\n    (-6/5, {x: 0, y: 0, z: 3/5})\n\n    By passing max, the maximum value for f under the constraints\n    is returned (if possible):\n\n    >>> lp(max, f, c)\n    (3/4, {x: 0, y: 3/4, z: 0})\n\n    Constraints that are equalities will require that the solution\n    also satisfy them:\n\n    >>> lp(max, f, c + [Eq(y - 9*x, 1)])\n    (5/7, {x: 0, y: 1, z: 1/7})\n\n    All symbols are reported, even if they are not in the objective\n    function:\n\n    >>> lp(min, x, [y + x >= 3, x >= 0])\n    (0, {x: 0, y: 3})\n    "
    (A, B, C, D, r, xx, aux) = _lp_matrices(f, constr)
    how = str(min_max).lower()
    if 'max' in how:
        (_o, p, d) = _simplex(A, B, -C, -D)
        o = -_o
    elif 'min' in how:
        (o, p, d) = _simplex(A, B, C, D)
    else:
        raise ValueError('expecting min or max')
    p = dict(zip(xx, p))
    if r:
        r = {k: v.xreplace(p) for (k, v) in r.items()}
        p.update(r)
        p = {k: p[k] for k in ordered(p) if k not in aux}
    return (o, p)

def lpmin(f, constr):
    if False:
        for i in range(10):
            print('nop')
    'return minimum of linear equation ``f`` under\n    linear constraints expressed using Ge, Le or Eq.\n\n    All variables are unbounded unless constrained.\n\n    Examples\n    ========\n\n    >>> from sympy.solvers.simplex import lpmin\n    >>> from sympy import Eq\n    >>> from sympy.abc import x, y\n    >>> lpmin(x, [2*x - 3*y >= -1, Eq(x + 3*y, 2), x <= 2*y])\n    (1/3, {x: 1/3, y: 5/9})\n\n    Negative values for variables are permitted unless explicitly\n    exluding, so minimizing ``x`` for ``x <= 3`` is an\n    unbounded problem while the following has a bounded solution:\n\n    >>> lpmin(x, [x >= 0, x <= 3])\n    (0, {x: 0})\n\n    Without indicating that ``x`` is nonnegative, there\n    is no minimum for this objective:\n\n    >>> lpmin(x, [x <= 3])\n    Traceback (most recent call last):\n    ...\n    sympy.solvers.simplex.UnboundedLPError:\n    Objective function can assume arbitrarily large values!\n\n    See Also\n    ========\n    linprog, lpmax\n    '
    return _lp(min, f, constr)

def lpmax(f, constr):
    if False:
        return 10
    'return maximum of linear equation ``f`` under\n    linear constraints expressed using Ge, Le or Eq.\n\n    All variables are unbounded unless constrained.\n\n    Examples\n    ========\n\n    >>> from sympy.solvers.simplex import lpmax\n    >>> from sympy import Eq\n    >>> from sympy.abc import x, y\n    >>> lpmax(x, [2*x - 3*y >= -1, Eq(x+ 3*y,2), x <= 2*y])\n    (4/5, {x: 4/5, y: 2/5})\n\n    Negative values for variables are permitted unless explicitly\n    exluding:\n\n    >>> lpmax(x, [x <= -1])\n    (-1, {x: -1})\n\n    If a non-negative constraint is added for x, there is no\n    possible solution:\n\n    >>> lpmax(x, [x <= -1, x >= 0])\n    Traceback (most recent call last):\n    ...\n    sympy.solvers.simplex.InfeasibleLPError: inconsistent/False constraint\n\n    See Also\n    ========\n    linprog, lpmin\n    '
    return _lp(max, f, constr)

def _handle_bounds(bounds):
    if False:
        print('Hello World!')
    unbound = []
    R = [0] * len(bounds)

    def n():
        if False:
            i = 10
            return i + 15
        return len(R) - 1

    def Arow(inc=1):
        if False:
            for i in range(10):
                print('nop')
        R.extend([0] * inc)
        return R[:]
    row = []
    for (x, (a, b)) in enumerate(bounds):
        if a is None and b is None:
            unbound.append(x)
        elif a is None:
            A = Arow()
            A[x] = 1
            A[n()] = 1
            B = [b]
            row.append((A, B))
            A = [0] * len(A)
            A[x] = -1
            A[n()] = -1
            B = [-b]
            row.append((A, B))
        elif b is None:
            if a:
                A = Arow()
                A[x] = 1
                A[n()] = -1
                B = [a]
                row.append((A, B))
                A = [0] * len(A)
                A[x] = -1
                A[n()] = 1
                B = [-a]
                row.append((A, B))
            else:
                pass
        else:
            A = Arow()
            A[x] = 1
            A[n()] = -1
            B = [a]
            row.append((A, B))
            A = [0] * len(A)
            A[x] = -1
            A[n()] = 1
            B = [-a]
            row.append((A, B))
            A = [0] * len(A)
            A[x] = 0
            A[n()] = 1
            B = [b - a]
            row.append((A, B))
    for x in unbound:
        A = Arow(2)
        B = [0]
        A[x] = 1
        A[n()] = 1
        A[n() - 1] = -1
        row.append((A, B))
        A = [0] * len(A)
        A[x] = -1
        A[n()] = -1
        A[n() - 1] = 1
        row.append((A, B))
    return (Matrix([r + [0] * (len(R) - len(r)) for (r, _) in row]), Matrix([i[1] for i in row]))

def linprog(c, A=None, b=None, A_eq=None, b_eq=None, bounds=None):
    if False:
        while True:
            i = 10
    "Return the minimization of ``c*x`` with the given\n    constraints ``A*x <= b`` and ``A_eq*x = b_eq``. Unless bounds\n    are given, variables will have nonnegative values in the solution.\n\n    If ``A`` is not given, then the dimension of the system will\n    be determined by the length of ``C``.\n\n    By default, all variables will be nonnegative. If ``bounds``\n    is given as a single tuple, ``(lo, hi)``, then all variables\n    will be constrained to be between ``lo`` and ``hi``. Use\n    None for a ``lo`` or ``hi`` if it is unconstrained in the\n    negative or positive direction, respectively, e.g.\n    ``(None, 0)`` indicates nonpositive values. To set\n    individual ranges, pass a list with length equal to the\n    number of columns in ``A``, each element being a tuple; if\n    only a few variables take on non-default values they can be\n    passed as a dictionary with keys giving the corresponding\n    column to which the variable is assigned, e.g. ``bounds={2:\n    (1, 4)}`` would limit the 3rd variable to have a value in\n    range ``[1, 4]``.\n\n    Examples\n    ========\n\n    >>> from sympy.solvers.simplex import linprog\n    >>> from sympy import symbols, Eq, linear_eq_to_matrix as M, Matrix\n    >>> x = x1, x2, x3, x4 = symbols('x1:5')\n    >>> X = Matrix(x)\n    >>> c, d = M(5*x2 + x3 + 4*x4 - x1, x)\n    >>> a, b = M([5*x2 + 2*x3 + 5*x4 - (x1 + 5)], x)\n    >>> aeq, beq = M([Eq(3*x2 + x4, 2), Eq(-x1 + x3 + 2*x4, 1)], x)\n    >>> constr = [i <= j for i,j in zip(a*X, b)]\n    >>> constr += [Eq(i, j) for i,j in zip(aeq*X, beq)]\n    >>> linprog(c, a, b, aeq, beq)\n    (9/2, [0, 1/2, 0, 1/2])\n    >>> assert all(i.subs(dict(zip(x, _[1]))) for i in constr)\n\n    See Also\n    ========\n    lpmin, lpmax\n    "
    C = Matrix(c)
    if C.rows != 1 and C.cols == 1:
        C = C.T
    if C.rows != 1:
        raise ValueError('C must be a single row.')
    if not A:
        if b:
            raise ValueError('A and b must both be given')
        (A, b) = (zeros(0, C.cols), zeros(C.cols, 1))
    else:
        (A, b) = [Matrix(i) for i in (A, b)]
    if A.cols != C.cols:
        raise ValueError('number of columns in A and C must match')
    if A_eq is None:
        if not b_eq is None:
            raise ValueError('A_eq and b_eq must both be given')
    else:
        (A_eq, b_eq) = [Matrix(i) for i in (A_eq, b_eq)]
        A = A.col_join(A_eq)
        A = A.col_join(-A_eq)
        b = b.col_join(b_eq)
        b = b.col_join(-b_eq)
    if not (bounds is None or bounds == {} or bounds == (0, None)):
        if type(bounds) is tuple and len(bounds) == 2:
            bounds = [bounds] * A.cols
        elif len(bounds) == A.cols and all((type(i) is tuple and len(i) == 2 for i in bounds)):
            pass
        elif type(bounds) is dict and all((type(i) is tuple and len(i) == 2 for i in bounds.values())):
            db = bounds
            bounds = [(0, None)] * A.cols
            while db:
                (i, j) = db.popitem()
                bounds[i] = j
        else:
            raise ValueError('unexpected bounds %s' % bounds)
        (A_, b_) = _handle_bounds(bounds)
        aux = A_.cols - A.cols
        if A:
            A = Matrix([[A, zeros(A.rows, aux)], [A_]])
            b = b.col_join(b_)
        else:
            A = A_
            b = b_
        C = C.row_join(zeros(1, aux))
    else:
        aux = -A.cols
    (o, p, d) = _simplex(A, b, C)
    return (o, p[:-aux])

def show_linprog(c, A=None, b=None, A_eq=None, b_eq=None, bounds=None):
    if False:
        return 10
    from sympy import symbols
    C = Matrix(c)
    if C.rows != 1 and C.cols == 1:
        C = C.T
    if C.rows != 1:
        raise ValueError('C must be a single row.')
    if not A:
        if b:
            raise ValueError('A and b must both be given')
        (A, b) = (zeros(0, C.cols), zeros(C.cols, 1))
    else:
        (A, b) = [Matrix(i) for i in (A, b)]
    if A.cols != C.cols:
        raise ValueError('number of columns in A and C must match')
    if A_eq is None:
        if not b_eq is None:
            raise ValueError('A_eq and b_eq must both be given')
    else:
        (A_eq, b_eq) = [Matrix(i) for i in (A_eq, b_eq)]
    if not (bounds is None or bounds == {} or bounds == (0, None)):
        if type(bounds) is tuple and len(bounds) == 2:
            bounds = [bounds] * A.cols
        elif len(bounds) == A.cols and all((type(i) is tuple and len(i) == 2 for i in bounds)):
            pass
        elif type(bounds) is dict and all((type(i) is tuple and len(i) == 2 for i in bounds.values())):
            db = bounds
            bounds = [(0, None)] * A.cols
            while db:
                (i, j) = db.popitem()
                bounds[i] = j
        else:
            raise ValueError('unexpected bounds %s' % bounds)
    x = Matrix(symbols('x1:%s' % (A.cols + 1)))
    (f, c) = ((C * x)[0], [i <= j for (i, j) in zip(A * x, b)] + [Eq(i, j) for (i, j) in zip(A_eq * x, b_eq)])
    for (i, (lo, hi)) in enumerate(bounds):
        if lo is None and hi is None:
            continue
        if lo is None:
            c.append(x[i] <= hi)
        elif hi is None:
            c.append(x[i] >= lo)
        else:
            c.append(x[i] >= lo)
            c.append(x[i] <= hi)
    return (f, c)