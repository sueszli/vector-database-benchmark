from sympy.core import Add, Mul, S
from sympy.core.containers import Tuple
from sympy.core.exprtools import factor_terms
from sympy.core.numbers import I
from sympy.core.relational import Eq, Equality
from sympy.core.sorting import default_sort_key, ordered
from sympy.core.symbol import Dummy, Symbol
from sympy.core.function import expand_mul, expand, Derivative, AppliedUndef, Function, Subs
from sympy.functions import exp, im, cos, sin, re, Piecewise, piecewise_fold, sqrt, log
from sympy.functions.combinatorial.factorials import factorial
from sympy.matrices import zeros, Matrix, NonSquareMatrixError, MatrixBase, eye
from sympy.polys import Poly, together
from sympy.simplify import collect, radsimp, signsimp
from sympy.simplify.powsimp import powdenest, powsimp
from sympy.simplify.ratsimp import ratsimp
from sympy.simplify.simplify import simplify
from sympy.sets.sets import FiniteSet
from sympy.solvers.deutils import ode_order
from sympy.solvers.solveset import NonlinearError, solveset
from sympy.utilities.iterables import connected_components, iterable, strongly_connected_components
from sympy.utilities.misc import filldedent
from sympy.integrals.integrals import Integral, integrate

def _get_func_order(eqs, funcs):
    if False:
        while True:
            i = 10
    return {func: max((ode_order(eq, func) for eq in eqs)) for func in funcs}

class ODEOrderError(ValueError):
    """Raised by linear_ode_to_matrix if the system has the wrong order"""
    pass

class ODENonlinearError(NonlinearError):
    """Raised by linear_ode_to_matrix if the system is nonlinear"""
    pass

def _simpsol(soleq):
    if False:
        i = 10
        return i + 15
    lhs = soleq.lhs
    sol = soleq.rhs
    sol = powsimp(sol)
    gens = list(sol.atoms(exp))
    p = Poly(sol, *gens, expand=False)
    gens = [factor_terms(g) for g in gens]
    if not gens:
        gens = p.gens
    syms = [Symbol('C1'), Symbol('C2')]
    terms = []
    for (coeff, monom) in zip(p.coeffs(), p.monoms()):
        coeff = piecewise_fold(coeff)
        if isinstance(coeff, Piecewise):
            coeff = Piecewise(*((ratsimp(coef).collect(syms), cond) for (coef, cond) in coeff.args))
        else:
            coeff = ratsimp(coeff).collect(syms)
        monom = Mul(*(g ** i for (g, i) in zip(gens, monom)))
        terms.append(coeff * monom)
    return Eq(lhs, Add(*terms))

def _solsimp(e, t):
    if False:
        return 10
    (no_t, has_t) = powsimp(expand_mul(e)).as_independent(t)
    no_t = ratsimp(no_t)
    has_t = has_t.replace(exp, lambda a: exp(factor_terms(a)))
    return no_t + has_t

def simpsol(sol, wrt1, wrt2, doit=True):
    if False:
        return 10
    'Simplify solutions from dsolve_system.'

    def simprhs(rhs, rep, wrt1, wrt2):
        if False:
            i = 10
            return i + 15
        'Simplify the rhs of an ODE solution'
        if rep:
            rhs = rhs.subs(rep)
        rhs = factor_terms(rhs)
        rhs = simp_coeff_dep(rhs, wrt1, wrt2)
        rhs = signsimp(rhs)
        return rhs

    def simp_coeff_dep(expr, wrt1, wrt2=None):
        if False:
            for i in range(10):
                print('nop')
        'Split rhs into terms, split terms into dep and coeff and collect on dep'
        add_dep_terms = lambda e: e.is_Add and e.has(*wrt1)
        expandable = lambda e: e.is_Mul and any(map(add_dep_terms, e.args))
        expand_func = lambda e: expand_mul(e, deep=False)
        expand_mul_mod = lambda e: e.replace(expandable, expand_func)
        terms = Add.make_args(expand_mul_mod(expr))
        dc = {}
        for term in terms:
            (coeff, dep) = term.as_independent(*wrt1, as_Add=False)
            dep = simpdep(dep, wrt1)
            if dep is not S.One:
                dep2 = factor_terms(dep)
                if not dep2.has(*wrt1):
                    coeff *= dep2
                    dep = S.One
            if dep not in dc:
                dc[dep] = coeff
            else:
                dc[dep] += coeff
        termpairs = ((simpcoeff(c, wrt2), d) for (d, c) in dc.items())
        if wrt2 is not None:
            termpairs = ((simp_coeff_dep(c, wrt2), d) for (c, d) in termpairs)
        return Add(*(c * d for (c, d) in termpairs))

    def simpdep(term, wrt1):
        if False:
            i = 10
            return i + 15
        'Normalise factors involving t with powsimp and recombine exp'

        def canonicalise(a):
            if False:
                return 10
            a = factor_terms(a)
            (num, den) = a.as_numer_denom()
            num = expand_mul(num)
            num = collect(num, wrt1)
            return num / den
        term = powsimp(term)
        rep = {e: exp(canonicalise(e.args[0])) for e in term.atoms(exp)}
        term = term.subs(rep)
        return term

    def simpcoeff(coeff, wrt2):
        if False:
            while True:
                i = 10
        'Bring to a common fraction and cancel with ratsimp'
        coeff = together(coeff)
        if coeff.is_polynomial():
            coeff = ratsimp(radsimp(coeff))
        if wrt2 is not None:
            syms = list(wrt2) + list(ordered(coeff.free_symbols - set(wrt2)))
        else:
            syms = list(ordered(coeff.free_symbols))
        coeff = collect(coeff, syms)
        coeff = together(coeff)
        return coeff
    if doit:
        integrals = set().union(*(s.atoms(Integral) for s in sol))
        rep = {i: factor_terms(i).doit() for i in integrals}
    else:
        rep = {}
    sol = [Eq(s.lhs, simprhs(s.rhs, rep, wrt1, wrt2)) for s in sol]
    return sol

def linodesolve_type(A, t, b=None):
    if False:
        i = 10
        return i + 15
    '\n    Helper function that determines the type of the system of ODEs for solving with :obj:`sympy.solvers.ode.systems.linodesolve()`\n\n    Explanation\n    ===========\n\n    This function takes in the coefficient matrix and/or the non-homogeneous term\n    and returns the type of the equation that can be solved by :obj:`sympy.solvers.ode.systems.linodesolve()`.\n\n    If the system is constant coefficient homogeneous, then "type1" is returned\n\n    If the system is constant coefficient non-homogeneous, then "type2" is returned\n\n    If the system is non-constant coefficient homogeneous, then "type3" is returned\n\n    If the system is non-constant coefficient non-homogeneous, then "type4" is returned\n\n    If the system has a non-constant coefficient matrix which can be factorized into constant\n    coefficient matrix, then "type5" or "type6" is returned for when the system is homogeneous or\n    non-homogeneous respectively.\n\n    Note that, if the system of ODEs is of "type3" or "type4", then along with the type,\n    the commutative antiderivative of the coefficient matrix is also returned.\n\n    If the system cannot be solved by :obj:`sympy.solvers.ode.systems.linodesolve()`, then\n    NotImplementedError is raised.\n\n    Parameters\n    ==========\n\n    A : Matrix\n        Coefficient matrix of the system of ODEs\n    b : Matrix or None\n        Non-homogeneous term of the system. The default value is None.\n        If this argument is None, then the system is assumed to be homogeneous.\n\n    Examples\n    ========\n\n    >>> from sympy import symbols, Matrix\n    >>> from sympy.solvers.ode.systems import linodesolve_type\n    >>> t = symbols("t")\n    >>> A = Matrix([[1, 1], [2, 3]])\n    >>> b = Matrix([t, 1])\n\n    >>> linodesolve_type(A, t)\n    {\'antiderivative\': None, \'type_of_equation\': \'type1\'}\n\n    >>> linodesolve_type(A, t, b=b)\n    {\'antiderivative\': None, \'type_of_equation\': \'type2\'}\n\n    >>> A_t = Matrix([[1, t], [-t, 1]])\n\n    >>> linodesolve_type(A_t, t)\n    {\'antiderivative\': Matrix([\n    [      t, t**2/2],\n    [-t**2/2,      t]]), \'type_of_equation\': \'type3\'}\n\n    >>> linodesolve_type(A_t, t, b=b)\n    {\'antiderivative\': Matrix([\n    [      t, t**2/2],\n    [-t**2/2,      t]]), \'type_of_equation\': \'type4\'}\n\n    >>> A_non_commutative = Matrix([[1, t], [t, -1]])\n    >>> linodesolve_type(A_non_commutative, t)\n    Traceback (most recent call last):\n    ...\n    NotImplementedError:\n    The system does not have a commutative antiderivative, it cannot be\n    solved by linodesolve.\n\n    Returns\n    =======\n\n    Dict\n\n    Raises\n    ======\n\n    NotImplementedError\n        When the coefficient matrix does not have a commutative antiderivative\n\n    See Also\n    ========\n\n    linodesolve: Function for which linodesolve_type gets the information\n\n    '
    match = {}
    is_non_constant = not _matrix_is_constant(A, t)
    is_non_homogeneous = not (b is None or b.is_zero_matrix)
    type = 'type{}'.format(int('{}{}'.format(int(is_non_constant), int(is_non_homogeneous)), 2) + 1)
    B = None
    match.update({'type_of_equation': type, 'antiderivative': B})
    if is_non_constant:
        (B, is_commuting) = _is_commutative_anti_derivative(A, t)
        if not is_commuting:
            raise NotImplementedError(filldedent('\n                The system does not have a commutative antiderivative, it cannot be solved\n                by linodesolve.\n            '))
        match['antiderivative'] = B
        match.update(_first_order_type5_6_subs(A, t, b=b))
    return match

def _first_order_type5_6_subs(A, t, b=None):
    if False:
        return 10
    match = {}
    factor_terms = _factor_matrix(A, t)
    is_homogeneous = b is None or b.is_zero_matrix
    if factor_terms is not None:
        t_ = Symbol('{}_'.format(t))
        F_t = integrate(factor_terms[0], t)
        inverse = solveset(Eq(t_, F_t), t)
        if isinstance(inverse, FiniteSet) and (not inverse.has(Piecewise)) and (len(inverse) == 1):
            A = factor_terms[1]
            if not is_homogeneous:
                b = b / factor_terms[0]
                b = b.subs(t, list(inverse)[0])
            type = 'type{}'.format(5 + (not is_homogeneous))
            match.update({'func_coeff': A, 'tau': F_t, 't_': t_, 'type_of_equation': type, 'rhs': b})
    return match

def linear_ode_to_matrix(eqs, funcs, t, order):
    if False:
        return 10
    "\n    Convert a linear system of ODEs to matrix form\n\n    Explanation\n    ===========\n\n    Express a system of linear ordinary differential equations as a single\n    matrix differential equation [1]. For example the system $x' = x + y + 1$\n    and $y' = x - y$ can be represented as\n\n    .. math:: A_1 X' = A_0 X + b\n\n    where $A_1$ and $A_0$ are $2 \\times 2$ matrices and $b$, $X$ and $X'$ are\n    $2 \\times 1$ matrices with $X = [x, y]^T$.\n\n    Higher-order systems are represented with additional matrices e.g. a\n    second-order system would look like\n\n    .. math:: A_2 X'' =  A_1 X' + A_0 X  + b\n\n    Examples\n    ========\n\n    >>> from sympy import Function, Symbol, Matrix, Eq\n    >>> from sympy.solvers.ode.systems import linear_ode_to_matrix\n    >>> t = Symbol('t')\n    >>> x = Function('x')\n    >>> y = Function('y')\n\n    We can create a system of linear ODEs like\n\n    >>> eqs = [\n    ...     Eq(x(t).diff(t), x(t) + y(t) + 1),\n    ...     Eq(y(t).diff(t), x(t) - y(t)),\n    ... ]\n    >>> funcs = [x(t), y(t)]\n    >>> order = 1 # 1st order system\n\n    Now ``linear_ode_to_matrix`` can represent this as a matrix\n    differential equation.\n\n    >>> (A1, A0), b = linear_ode_to_matrix(eqs, funcs, t, order)\n    >>> A1\n    Matrix([\n    [1, 0],\n    [0, 1]])\n    >>> A0\n    Matrix([\n    [1, 1],\n    [1,  -1]])\n    >>> b\n    Matrix([\n    [1],\n    [0]])\n\n    The original equations can be recovered from these matrices:\n\n    >>> eqs_mat = Matrix([eq.lhs - eq.rhs for eq in eqs])\n    >>> X = Matrix(funcs)\n    >>> A1 * X.diff(t) - A0 * X - b == eqs_mat\n    True\n\n    If the system of equations has a maximum order greater than the\n    order of the system specified, a ODEOrderError exception is raised.\n\n    >>> eqs = [Eq(x(t).diff(t, 2), x(t).diff(t) + x(t)), Eq(y(t).diff(t), y(t) + x(t))]\n    >>> linear_ode_to_matrix(eqs, funcs, t, 1)\n    Traceback (most recent call last):\n    ...\n    ODEOrderError: Cannot represent system in 1-order form\n\n    If the system of equations is nonlinear, then ODENonlinearError is\n    raised.\n\n    >>> eqs = [Eq(x(t).diff(t), x(t) + y(t)), Eq(y(t).diff(t), y(t)**2 + x(t))]\n    >>> linear_ode_to_matrix(eqs, funcs, t, 1)\n    Traceback (most recent call last):\n    ...\n    ODENonlinearError: The system of ODEs is nonlinear.\n\n    Parameters\n    ==========\n\n    eqs : list of SymPy expressions or equalities\n        The equations as expressions (assumed equal to zero).\n    funcs : list of applied functions\n        The dependent variables of the system of ODEs.\n    t : symbol\n        The independent variable.\n    order : int\n        The order of the system of ODEs.\n\n    Returns\n    =======\n\n    The tuple ``(As, b)`` where ``As`` is a tuple of matrices and ``b`` is the\n    the matrix representing the rhs of the matrix equation.\n\n    Raises\n    ======\n\n    ODEOrderError\n        When the system of ODEs have an order greater than what was specified\n    ODENonlinearError\n        When the system of ODEs is nonlinear\n\n    See Also\n    ========\n\n    linear_eq_to_matrix: for systems of linear algebraic equations.\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Matrix_differential_equation\n\n    "
    from sympy.solvers.solveset import linear_eq_to_matrix
    if any((ode_order(eq, func) > order for eq in eqs for func in funcs)):
        msg = 'Cannot represent system in {}-order form'
        raise ODEOrderError(msg.format(order))
    As = []
    for o in range(order, -1, -1):
        syms = [func.diff(t, o) for func in funcs]
        try:
            (Ai, b) = linear_eq_to_matrix(eqs, syms)
        except NonlinearError:
            raise ODENonlinearError('The system of ODEs is nonlinear.')
        Ai = Ai.applyfunc(expand_mul)
        As.append(Ai if o == order else -Ai)
        if o:
            eqs = [-eq for eq in b]
        else:
            rhs = b
    return (As, rhs)

def matrix_exp(A, t):
    if False:
        i = 10
        return i + 15
    "\n    Matrix exponential $\\exp(A*t)$ for the matrix ``A`` and scalar ``t``.\n\n    Explanation\n    ===========\n\n    This functions returns the $\\exp(A*t)$ by doing a simple\n    matrix multiplication:\n\n    .. math:: \\exp(A*t) = P * expJ * P^{-1}\n\n    where $expJ$ is $\\exp(J*t)$. $J$ is the Jordan normal\n    form of $A$ and $P$ is matrix such that:\n\n    .. math:: A = P * J * P^{-1}\n\n    The matrix exponential $\\exp(A*t)$ appears in the solution of linear\n    differential equations. For example if $x$ is a vector and $A$ is a matrix\n    then the initial value problem\n\n    .. math:: \\frac{dx(t)}{dt} = A \\times x(t),   x(0) = x0\n\n    has the unique solution\n\n    .. math:: x(t) = \\exp(A t) x0\n\n    Examples\n    ========\n\n    >>> from sympy import Symbol, Matrix, pprint\n    >>> from sympy.solvers.ode.systems import matrix_exp\n    >>> t = Symbol('t')\n\n    We will consider a 2x2 matrix for comupting the exponential\n\n    >>> A = Matrix([[2, -5], [2, -4]])\n    >>> pprint(A)\n    [2  -5]\n    [     ]\n    [2  -4]\n\n    Now, exp(A*t) is given as follows:\n\n    >>> pprint(matrix_exp(A, t))\n    [   -t           -t                    -t              ]\n    [3*e  *sin(t) + e  *cos(t)         -5*e  *sin(t)       ]\n    [                                                      ]\n    [         -t                     -t           -t       ]\n    [      2*e  *sin(t)         - 3*e  *sin(t) + e  *cos(t)]\n\n    Parameters\n    ==========\n\n    A : Matrix\n        The matrix $A$ in the expression $\\exp(A*t)$\n    t : Symbol\n        The independent variable\n\n    See Also\n    ========\n\n    matrix_exp_jordan_form: For exponential of Jordan normal form\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Jordan_normal_form\n    .. [2] https://en.wikipedia.org/wiki/Matrix_exponential\n\n    "
    (P, expJ) = matrix_exp_jordan_form(A, t)
    return P * expJ * P.inv()

def matrix_exp_jordan_form(A, t):
    if False:
        i = 10
        return i + 15
    "\n    Matrix exponential $\\exp(A*t)$ for the matrix *A* and scalar *t*.\n\n    Explanation\n    ===========\n\n    Returns the Jordan form of the $\\exp(A*t)$ along with the matrix $P$ such that:\n\n    .. math::\n        \\exp(A*t) = P * expJ * P^{-1}\n\n    Examples\n    ========\n\n    >>> from sympy import Matrix, Symbol\n    >>> from sympy.solvers.ode.systems import matrix_exp, matrix_exp_jordan_form\n    >>> t = Symbol('t')\n\n    We will consider a 2x2 defective matrix. This shows that our method\n    works even for defective matrices.\n\n    >>> A = Matrix([[1, 1], [0, 1]])\n\n    It can be observed that this function gives us the Jordan normal form\n    and the required invertible matrix P.\n\n    >>> P, expJ = matrix_exp_jordan_form(A, t)\n\n    Here, it is shown that P and expJ returned by this function is correct\n    as they satisfy the formula: P * expJ * P_inverse = exp(A*t).\n\n    >>> P * expJ * P.inv() == matrix_exp(A, t)\n    True\n\n    Parameters\n    ==========\n\n    A : Matrix\n        The matrix $A$ in the expression $\\exp(A*t)$\n    t : Symbol\n        The independent variable\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Defective_matrix\n    .. [2] https://en.wikipedia.org/wiki/Jordan_matrix\n    .. [3] https://en.wikipedia.org/wiki/Jordan_normal_form\n\n    "
    (N, M) = A.shape
    if N != M:
        raise ValueError('Needed square matrix but got shape (%s, %s)' % (N, M))
    elif A.has(t):
        raise ValueError('Matrix A should not depend on t')

    def jordan_chains(A):
        if False:
            print('Hello World!')
        'Chains from Jordan normal form analogous to M.eigenvects().\n        Returns a dict with eignevalues as keys like:\n            {e1: [[v111,v112,...], [v121, v122,...]], e2:...}\n        where vijk is the kth vector in the jth chain for eigenvalue i.\n        '
        (P, blocks) = A.jordan_cells()
        basis = [P[:, i] for i in range(P.shape[1])]
        n = 0
        chains = {}
        for b in blocks:
            eigval = b[0, 0]
            size = b.shape[0]
            if eigval not in chains:
                chains[eigval] = []
            chains[eigval].append(basis[n:n + size])
            n += size
        return chains
    eigenchains = jordan_chains(A)
    eigenchains_iter = sorted(eigenchains.items(), key=default_sort_key)
    isreal = not A.has(I)
    blocks = []
    vectors = []
    seen_conjugate = set()
    for (e, chains) in eigenchains_iter:
        for chain in chains:
            n = len(chain)
            if isreal and e != e.conjugate() and (e.conjugate() in eigenchains):
                if e in seen_conjugate:
                    continue
                seen_conjugate.add(e.conjugate())
                exprt = exp(re(e) * t)
                imrt = im(e) * t
                imblock = Matrix([[cos(imrt), sin(imrt)], [-sin(imrt), cos(imrt)]])
                expJblock2 = Matrix(n, n, lambda i, j: imblock * t ** (j - i) / factorial(j - i) if j >= i else zeros(2, 2))
                expJblock = Matrix(2 * n, 2 * n, lambda i, j: expJblock2[i // 2, j // 2][i % 2, j % 2])
                blocks.append(exprt * expJblock)
                for i in range(n):
                    vectors.append(re(chain[i]))
                    vectors.append(im(chain[i]))
            else:
                vectors.extend(chain)
                fun = lambda i, j: t ** (j - i) / factorial(j - i) if j >= i else 0
                expJblock = Matrix(n, n, fun)
                blocks.append(exp(e * t) * expJblock)
    expJ = Matrix.diag(*blocks)
    P = Matrix(N, N, lambda i, j: vectors[j][i])
    return (P, expJ)

def linodesolve(A, t, b=None, B=None, type='auto', doit=False, tau=None):
    if False:
        return 10
    '\n    System of n equations linear first-order differential equations\n\n    Explanation\n    ===========\n\n    This solver solves the system of ODEs of the following form:\n\n    .. math::\n        X\'(t) = A(t) X(t) +  b(t)\n\n    Here, $A(t)$ is the coefficient matrix, $X(t)$ is the vector of n independent variables,\n    $b(t)$ is the non-homogeneous term and $X\'(t)$ is the derivative of $X(t)$\n\n    Depending on the properties of $A(t)$ and $b(t)$, this solver evaluates the solution\n    differently.\n\n    When $A(t)$ is constant coefficient matrix and $b(t)$ is zero vector i.e. system is homogeneous,\n    the system is "type1". The solution is:\n\n    .. math::\n        X(t) = \\exp(A t) C\n\n    Here, $C$ is a vector of constants and $A$ is the constant coefficient matrix.\n\n    When $A(t)$ is constant coefficient matrix and $b(t)$ is non-zero i.e. system is non-homogeneous,\n    the system is "type2". The solution is:\n\n    .. math::\n        X(t) = e^{A t} ( \\int e^{- A t} b \\,dt + C)\n\n    When $A(t)$ is coefficient matrix such that its commutative with its antiderivative $B(t)$ and\n    $b(t)$ is a zero vector i.e. system is homogeneous, the system is "type3". The solution is:\n\n    .. math::\n        X(t) = \\exp(B(t)) C\n\n    When $A(t)$ is commutative with its antiderivative $B(t)$ and $b(t)$ is non-zero i.e. system is\n    non-homogeneous, the system is "type4". The solution is:\n\n    .. math::\n        X(t) =  e^{B(t)} ( \\int e^{-B(t)} b(t) \\,dt + C)\n\n    When $A(t)$ is a coefficient matrix such that it can be factorized into a scalar and a constant\n    coefficient matrix:\n\n    .. math::\n        A(t) = f(t) * A\n\n    Where $f(t)$ is a scalar expression in the independent variable $t$ and $A$ is a constant matrix,\n    then we can do the following substitutions:\n\n    .. math::\n        tau = \\int f(t) dt, X(t) = Y(tau), b(t) = b(f^{-1}(tau))\n\n    Here, the substitution for the non-homogeneous term is done only when its non-zero.\n    Using these substitutions, our original system becomes:\n\n    .. math::\n        Y\'(tau) = A * Y(tau) + b(tau)/f(tau)\n\n    The above system can be easily solved using the solution for "type1" or "type2" depending\n    on the homogeneity of the system. After we get the solution for $Y(tau)$, we substitute the\n    solution for $tau$ as $t$ to get back $X(t)$\n\n    .. math::\n        X(t) = Y(tau)\n\n    Systems of "type5" and "type6" have a commutative antiderivative but we use this solution\n    because its faster to compute.\n\n    The final solution is the general solution for all the four equations since a constant coefficient\n    matrix is always commutative with its antidervative.\n\n    An additional feature of this function is, if someone wants to substitute for value of the independent\n    variable, they can pass the substitution `tau` and the solution will have the independent variable\n    substituted with the passed expression(`tau`).\n\n    Parameters\n    ==========\n\n    A : Matrix\n        Coefficient matrix of the system of linear first order ODEs.\n    t : Symbol\n        Independent variable in the system of ODEs.\n    b : Matrix or None\n        Non-homogeneous term in the system of ODEs. If None is passed,\n        a homogeneous system of ODEs is assumed.\n    B : Matrix or None\n        Antiderivative of the coefficient matrix. If the antiderivative\n        is not passed and the solution requires the term, then the solver\n        would compute it internally.\n    type : String\n        Type of the system of ODEs passed. Depending on the type, the\n        solution is evaluated. The type values allowed and the corresponding\n        system it solves are: "type1" for constant coefficient homogeneous\n        "type2" for constant coefficient non-homogeneous, "type3" for non-constant\n        coefficient homogeneous, "type4" for non-constant coefficient non-homogeneous,\n        "type5" and "type6" for non-constant coefficient homogeneous and non-homogeneous\n        systems respectively where the coefficient matrix can be factorized to a constant\n        coefficient matrix.\n        The default value is "auto" which will let the solver decide the correct type of\n        the system passed.\n    doit : Boolean\n        Evaluate the solution if True, default value is False\n    tau: Expression\n        Used to substitute for the value of `t` after we get the solution of the system.\n\n    Examples\n    ========\n\n    To solve the system of ODEs using this function directly, several things must be\n    done in the right order. Wrong inputs to the function will lead to incorrect results.\n\n    >>> from sympy import symbols, Function, Eq\n    >>> from sympy.solvers.ode.systems import canonical_odes, linear_ode_to_matrix, linodesolve, linodesolve_type\n    >>> from sympy.solvers.ode.subscheck import checkodesol\n    >>> f, g = symbols("f, g", cls=Function)\n    >>> x, a = symbols("x, a")\n    >>> funcs = [f(x), g(x)]\n    >>> eqs = [Eq(f(x).diff(x) - f(x), a*g(x) + 1), Eq(g(x).diff(x) + g(x), a*f(x))]\n\n    Here, it is important to note that before we derive the coefficient matrix, it is\n    important to get the system of ODEs into the desired form. For that we will use\n    :obj:`sympy.solvers.ode.systems.canonical_odes()`.\n\n    >>> eqs = canonical_odes(eqs, funcs, x)\n    >>> eqs\n    [[Eq(Derivative(f(x), x), a*g(x) + f(x) + 1), Eq(Derivative(g(x), x), a*f(x) - g(x))]]\n\n    Now, we will use :obj:`sympy.solvers.ode.systems.linear_ode_to_matrix()` to get the coefficient matrix and the\n    non-homogeneous term if it is there.\n\n    >>> eqs = eqs[0]\n    >>> (A1, A0), b = linear_ode_to_matrix(eqs, funcs, x, 1)\n    >>> A = A0\n\n    We have the coefficient matrices and the non-homogeneous term ready. Now, we can use\n    :obj:`sympy.solvers.ode.systems.linodesolve_type()` to get the information for the system of ODEs\n    to finally pass it to the solver.\n\n    >>> system_info = linodesolve_type(A, x, b=b)\n    >>> sol_vector = linodesolve(A, x, b=b, B=system_info[\'antiderivative\'], type=system_info[\'type_of_equation\'])\n\n    Now, we can prove if the solution is correct or not by using :obj:`sympy.solvers.ode.checkodesol()`\n\n    >>> sol = [Eq(f, s) for f, s in zip(funcs, sol_vector)]\n    >>> checkodesol(eqs, sol)\n    (True, [0, 0])\n\n    We can also use the doit method to evaluate the solutions passed by the function.\n\n    >>> sol_vector_evaluated = linodesolve(A, x, b=b, type="type2", doit=True)\n\n    Now, we will look at a system of ODEs which is non-constant.\n\n    >>> eqs = [Eq(f(x).diff(x), f(x) + x*g(x)), Eq(g(x).diff(x), -x*f(x) + g(x))]\n\n    The system defined above is already in the desired form, so we do not have to convert it.\n\n    >>> (A1, A0), b = linear_ode_to_matrix(eqs, funcs, x, 1)\n    >>> A = A0\n\n    A user can also pass the commutative antiderivative required for type3 and type4 system of ODEs.\n    Passing an incorrect one will lead to incorrect results. If the coefficient matrix is not commutative\n    with its antiderivative, then :obj:`sympy.solvers.ode.systems.linodesolve_type()` raises a NotImplementedError.\n    If it does have a commutative antiderivative, then the function just returns the information about the system.\n\n    >>> system_info = linodesolve_type(A, x, b=b)\n\n    Now, we can pass the antiderivative as an argument to get the solution. If the system information is not\n    passed, then the solver will compute the required arguments internally.\n\n    >>> sol_vector = linodesolve(A, x, b=b)\n\n    Once again, we can verify the solution obtained.\n\n    >>> sol = [Eq(f, s) for f, s in zip(funcs, sol_vector)]\n    >>> checkodesol(eqs, sol)\n    (True, [0, 0])\n\n    Returns\n    =======\n\n    List\n\n    Raises\n    ======\n\n    ValueError\n        This error is raised when the coefficient matrix, non-homogeneous term\n        or the antiderivative, if passed, are not a matrix or\n        do not have correct dimensions\n    NonSquareMatrixError\n        When the coefficient matrix or its antiderivative, if passed is not a\n        square matrix\n    NotImplementedError\n        If the coefficient matrix does not have a commutative antiderivative\n\n    See Also\n    ========\n\n    linear_ode_to_matrix: Coefficient matrix computation function\n    canonical_odes: System of ODEs representation change\n    linodesolve_type: Getting information about systems of ODEs to pass in this solver\n\n    '
    if not isinstance(A, MatrixBase):
        raise ValueError(filldedent('            The coefficients of the system of ODEs should be of type Matrix\n        '))
    if not A.is_square:
        raise NonSquareMatrixError(filldedent('            The coefficient matrix must be a square\n        '))
    if b is not None:
        if not isinstance(b, MatrixBase):
            raise ValueError(filldedent('                The non-homogeneous terms of the system of ODEs should be of type Matrix\n            '))
        if A.rows != b.rows:
            raise ValueError(filldedent('                The system of ODEs should have the same number of non-homogeneous terms and the number of\n                equations\n            '))
    if B is not None:
        if not isinstance(B, MatrixBase):
            raise ValueError(filldedent('                The antiderivative of coefficients of the system of ODEs should be of type Matrix\n            '))
        if not B.is_square:
            raise NonSquareMatrixError(filldedent('                The antiderivative of the coefficient matrix must be a square\n            '))
        if A.rows != B.rows:
            raise ValueError(filldedent('                        The coefficient matrix and its antiderivative should have same dimensions\n                    '))
    if not any((type == 'type{}'.format(i) for i in range(1, 7))) and (not type == 'auto'):
        raise ValueError(filldedent('                    The input type should be a valid one\n                '))
    n = A.rows
    Cvect = Matrix([Dummy() for _ in range(n)])
    if b is None and any((type == typ for typ in ['type2', 'type4', 'type6'])):
        b = zeros(n, 1)
    is_transformed = tau is not None
    passed_type = type
    if type == 'auto':
        system_info = linodesolve_type(A, t, b=b)
        type = system_info['type_of_equation']
        B = system_info['antiderivative']
    if type in ('type5', 'type6'):
        is_transformed = True
        if passed_type != 'auto':
            if tau is None:
                system_info = _first_order_type5_6_subs(A, t, b=b)
                if not system_info:
                    raise ValueError(filldedent("\n                        The system passed isn't {}.\n                    ".format(type)))
                tau = system_info['tau']
                t = system_info['t_']
                A = system_info['A']
                b = system_info['b']
    intx_wrtt = lambda x: Integral(x, t) if x else 0
    if type in ('type1', 'type2', 'type5', 'type6'):
        (P, J) = matrix_exp_jordan_form(A, t)
        P = simplify(P)
        if type in ('type1', 'type5'):
            sol_vector = P * (J * Cvect)
        else:
            Jinv = J.subs(t, -t)
            sol_vector = P * J * ((Jinv * P.inv() * b).applyfunc(intx_wrtt) + Cvect)
    else:
        if B is None:
            (B, _) = _is_commutative_anti_derivative(A, t)
        if type == 'type3':
            sol_vector = B.exp() * Cvect
        else:
            sol_vector = B.exp() * (((-B).exp() * b).applyfunc(intx_wrtt) + Cvect)
    if is_transformed:
        sol_vector = sol_vector.subs(t, tau)
    gens = sol_vector.atoms(exp)
    if type != 'type1':
        sol_vector = [expand_mul(s) for s in sol_vector]
    sol_vector = [collect(s, ordered(gens), exact=True) for s in sol_vector]
    if doit:
        sol_vector = [s.doit() for s in sol_vector]
    return sol_vector

def _matrix_is_constant(M, t):
    if False:
        print('Hello World!')
    'Checks if the matrix M is independent of t or not.'
    return all((coef.as_independent(t, as_Add=True)[1] == 0 for coef in M))

def canonical_odes(eqs, funcs, t):
    if False:
        print('Hello World!')
    '\n    Function that solves for highest order derivatives in a system\n\n    Explanation\n    ===========\n\n    This function inputs a system of ODEs and based on the system,\n    the dependent variables and their highest order, returns the system\n    in the following form:\n\n    .. math::\n        X\'(t) = A(t) X(t) + b(t)\n\n    Here, $X(t)$ is the vector of dependent variables of lower order, $A(t)$ is\n    the coefficient matrix, $b(t)$ is the non-homogeneous term and $X\'(t)$ is the\n    vector of dependent variables in their respective highest order. We use the term\n    canonical form to imply the system of ODEs which is of the above form.\n\n    If the system passed has a non-linear term with multiple solutions, then a list of\n    systems is returned in its canonical form.\n\n    Parameters\n    ==========\n\n    eqs : List\n        List of the ODEs\n    funcs : List\n        List of dependent variables\n    t : Symbol\n        Independent variable\n\n    Examples\n    ========\n\n    >>> from sympy import symbols, Function, Eq, Derivative\n    >>> from sympy.solvers.ode.systems import canonical_odes\n    >>> f, g = symbols("f g", cls=Function)\n    >>> x, y = symbols("x y")\n    >>> funcs = [f(x), g(x)]\n    >>> eqs = [Eq(f(x).diff(x) - 7*f(x), 12*g(x)), Eq(g(x).diff(x) + g(x), 20*f(x))]\n\n    >>> canonical_eqs = canonical_odes(eqs, funcs, x)\n    >>> canonical_eqs\n    [[Eq(Derivative(f(x), x), 7*f(x) + 12*g(x)), Eq(Derivative(g(x), x), 20*f(x) - g(x))]]\n\n    >>> system = [Eq(Derivative(f(x), x)**2 - 2*Derivative(f(x), x) + 1, 4), Eq(-y*f(x) + Derivative(g(x), x), 0)]\n\n    >>> canonical_system = canonical_odes(system, funcs, x)\n    >>> canonical_system\n    [[Eq(Derivative(f(x), x), -1), Eq(Derivative(g(x), x), y*f(x))], [Eq(Derivative(f(x), x), 3), Eq(Derivative(g(x), x), y*f(x))]]\n\n    Returns\n    =======\n\n    List\n\n    '
    from sympy.solvers.solvers import solve
    order = _get_func_order(eqs, funcs)
    canon_eqs = solve(eqs, *[func.diff(t, order[func]) for func in funcs], dict=True)
    systems = []
    for eq in canon_eqs:
        system = [Eq(func.diff(t, order[func]), eq[func.diff(t, order[func])]) for func in funcs]
        systems.append(system)
    return systems

def _is_commutative_anti_derivative(A, t):
    if False:
        i = 10
        return i + 15
    '\n    Helper function for determining if the Matrix passed is commutative with its antiderivative\n\n    Explanation\n    ===========\n\n    This function checks if the Matrix $A$ passed is commutative with its antiderivative with respect\n    to the independent variable $t$.\n\n    .. math::\n        B(t) = \\int A(t) dt\n\n    The function outputs two values, first one being the antiderivative $B(t)$, second one being a\n    boolean value, if True, then the matrix $A(t)$ passed is commutative with $B(t)$, else the matrix\n    passed isn\'t commutative with $B(t)$.\n\n    Parameters\n    ==========\n\n    A : Matrix\n        The matrix which has to be checked\n    t : Symbol\n        Independent variable\n\n    Examples\n    ========\n\n    >>> from sympy import symbols, Matrix\n    >>> from sympy.solvers.ode.systems import _is_commutative_anti_derivative\n    >>> t = symbols("t")\n    >>> A = Matrix([[1, t], [-t, 1]])\n\n    >>> B, is_commuting = _is_commutative_anti_derivative(A, t)\n    >>> is_commuting\n    True\n\n    Returns\n    =======\n\n    Matrix, Boolean\n\n    '
    B = integrate(A, t)
    is_commuting = (B * A - A * B).applyfunc(expand).applyfunc(factor_terms).is_zero_matrix
    is_commuting = False if is_commuting is None else is_commuting
    return (B, is_commuting)

def _factor_matrix(A, t):
    if False:
        return 10
    term = None
    for element in A:
        temp_term = element.as_independent(t)[1]
        if temp_term.has(t):
            term = temp_term
            break
    if term is not None:
        A_factored = (A / term).applyfunc(ratsimp)
        can_factor = _matrix_is_constant(A_factored, t)
        term = (term, A_factored) if can_factor else None
    return term

def _is_second_order_type2(A, t):
    if False:
        for i in range(10):
            print('nop')
    term = _factor_matrix(A, t)
    is_type2 = False
    if term is not None:
        term = 1 / term[0]
        is_type2 = term.is_polynomial()
    if is_type2:
        poly = Poly(term.expand(), t)
        monoms = poly.monoms()
        if monoms[0][0] in (2, 4):
            cs = _get_poly_coeffs(poly, 4)
            (a, b, c, d, e) = cs
            a1 = powdenest(sqrt(a), force=True)
            c1 = powdenest(sqrt(e), force=True)
            b1 = powdenest(sqrt(c - 2 * a1 * c1), force=True)
            is_type2 = b == 2 * a1 * b1 and d == 2 * b1 * c1
            term = a1 * t ** 2 + b1 * t + c1
        else:
            is_type2 = False
    return (is_type2, term)

def _get_poly_coeffs(poly, order):
    if False:
        i = 10
        return i + 15
    cs = [0 for _ in range(order + 1)]
    for (c, m) in zip(poly.coeffs(), poly.monoms()):
        cs[-1 - m[0]] = c
    return cs

def _match_second_order_type(A1, A0, t, b=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Works only for second order system in its canonical form.\n\n    Type 0: Constant coefficient matrix, can be simply solved by\n            introducing dummy variables.\n    Type 1: When the substitution: $U = t*X' - X$ works for reducing\n            the second order system to first order system.\n    Type 2: When the system is of the form: $poly * X'' = A*X$ where\n            $poly$ is square of a quadratic polynomial with respect to\n            *t* and $A$ is a constant coefficient matrix.\n\n    "
    match = {'type_of_equation': 'type0'}
    n = A1.shape[0]
    if _matrix_is_constant(A1, t) and _matrix_is_constant(A0, t):
        return match
    if (A1 + A0 * t).applyfunc(expand_mul).is_zero_matrix:
        match.update({'type_of_equation': 'type1', 'A1': A1})
    elif A1.is_zero_matrix and (b is None or b.is_zero_matrix):
        (is_type2, term) = _is_second_order_type2(A0, t)
        if is_type2:
            (a, b, c) = _get_poly_coeffs(Poly(term, t), 2)
            A = (A0 * (term ** 2).expand()).applyfunc(ratsimp) + (b ** 2 / 4 - a * c) * eye(n, n)
            tau = integrate(1 / term, t)
            t_ = Symbol('{}_'.format(t))
            match.update({'type_of_equation': 'type2', 'A0': A, 'g(t)': sqrt(term), 'tau': tau, 'is_transformed': True, 't_': t_})
    return match

def _second_order_subs_type1(A, b, funcs, t):
    if False:
        for i in range(10):
            print('nop')
    "\n    For a linear, second order system of ODEs, a particular substitution.\n\n    A system of the below form can be reduced to a linear first order system of\n    ODEs:\n    .. math::\n        X'' = A(t) * (t*X' - X) + b(t)\n\n    By substituting:\n    .. math::  U = t*X' - X\n\n    To get the system:\n    .. math::  U' = t*(A(t)*U + b(t))\n\n    Where $U$ is the vector of dependent variables, $X$ is the vector of dependent\n    variables in `funcs` and $X'$ is the first order derivative of $X$ with respect to\n    $t$. It may or may not reduce the system into linear first order system of ODEs.\n\n    Then a check is made to determine if the system passed can be reduced or not, if\n    this substitution works, then the system is reduced and its solved for the new\n    substitution. After we get the solution for $U$:\n\n    .. math::  U = a(t)\n\n    We substitute and return the reduced system:\n\n    .. math::\n        a(t) = t*X' - X\n\n    Parameters\n    ==========\n\n    A: Matrix\n        Coefficient matrix($A(t)*t$) of the second order system of this form.\n    b: Matrix\n        Non-homogeneous term($b(t)$) of the system of ODEs.\n    funcs: List\n        List of dependent variables\n    t: Symbol\n        Independent variable of the system of ODEs.\n\n    Returns\n    =======\n\n    List\n\n    "
    U = Matrix([t * func.diff(t) - func for func in funcs])
    sol = linodesolve(A, t, t * b)
    reduced_eqs = [Eq(u, s) for (s, u) in zip(sol, U)]
    reduced_eqs = canonical_odes(reduced_eqs, funcs, t)[0]
    return reduced_eqs

def _second_order_subs_type2(A, funcs, t_):
    if False:
        i = 10
        return i + 15
    "\n    Returns a second order system based on the coefficient matrix passed.\n\n    Explanation\n    ===========\n\n    This function returns a system of second order ODE of the following form:\n\n    .. math::\n        X'' = A * X\n\n    Here, $X$ is the vector of dependent variables, but a bit modified, $A$ is the\n    coefficient matrix passed.\n\n    Along with returning the second order system, this function also returns the new\n    dependent variables with the new independent variable `t_` passed.\n\n    Parameters\n    ==========\n\n    A: Matrix\n        Coefficient matrix of the system\n    funcs: List\n        List of old dependent variables\n    t_: Symbol\n        New independent variable\n\n    Returns\n    =======\n\n    List, List\n\n    "
    func_names = [func.func.__name__ for func in funcs]
    new_funcs = [Function(Dummy('{}_'.format(name)))(t_) for name in func_names]
    rhss = A * Matrix(new_funcs)
    new_eqs = [Eq(func.diff(t_, 2), rhs) for (func, rhs) in zip(new_funcs, rhss)]
    return (new_eqs, new_funcs)

def _is_euler_system(As, t):
    if False:
        return 10
    return all((_matrix_is_constant((A * t ** i).applyfunc(ratsimp), t) for (i, A) in enumerate(As)))

def _classify_linear_system(eqs, funcs, t, is_canon=False):
    if False:
        return 10
    "\n    Returns a dictionary with details of the eqs if the system passed is linear\n    and can be classified by this function else returns None\n\n    Explanation\n    ===========\n\n    This function takes the eqs, converts it into a form Ax = b where x is a vector of terms\n    containing dependent variables and their derivatives till their maximum order. If it is\n    possible to convert eqs into Ax = b, then all the equations in eqs are linear otherwise\n    they are non-linear.\n\n    To check if the equations are constant coefficient, we need to check if all the terms in\n    A obtained above are constant or not.\n\n    To check if the equations are homogeneous or not, we need to check if b is a zero matrix\n    or not.\n\n    Parameters\n    ==========\n\n    eqs: List\n        List of ODEs\n    funcs: List\n        List of dependent variables\n    t: Symbol\n        Independent variable of the equations in eqs\n    is_canon: Boolean\n        If True, then this function will not try to get the\n        system in canonical form. Default value is False\n\n    Returns\n    =======\n\n    match = {\n        'no_of_equation': len(eqs),\n        'eq': eqs,\n        'func': funcs,\n        'order': order,\n        'is_linear': is_linear,\n        'is_constant': is_constant,\n        'is_homogeneous': is_homogeneous,\n    }\n\n    Dict or list of Dicts or None\n        Dict with values for keys:\n            1. no_of_equation: Number of equations\n            2. eq: The set of equations\n            3. func: List of dependent variables\n            4. order: A dictionary that gives the order of the\n                      dependent variable in eqs\n            5. is_linear: Boolean value indicating if the set of\n                          equations are linear or not.\n            6. is_constant: Boolean value indicating if the set of\n                          equations have constant coefficients or not.\n            7. is_homogeneous: Boolean value indicating if the set of\n                          equations are homogeneous or not.\n            8. commutative_antiderivative: Antiderivative of the coefficient\n                          matrix if the coefficient matrix is non-constant\n                          and commutative with its antiderivative. This key\n                          may or may not exist.\n            9. is_general: Boolean value indicating if the system of ODEs is\n                           solvable using one of the general case solvers or not.\n            10. rhs: rhs of the non-homogeneous system of ODEs in Matrix form. This\n                     key may or may not exist.\n            11. is_higher_order: True if the system passed has an order greater than 1.\n                                 This key may or may not exist.\n            12. is_second_order: True if the system passed is a second order ODE. This\n                                 key may or may not exist.\n        This Dict is the answer returned if the eqs are linear and constant\n        coefficient. Otherwise, None is returned.\n\n    "
    if len(funcs) != len(eqs):
        raise ValueError('Number of functions given is not equal to the number of equations %s' % funcs)
    for func in funcs:
        if len(func.args) != 1:
            raise ValueError('dsolve() and classify_sysode() work with functions of one variable only, not %s' % func)
    order = _get_func_order(eqs, funcs)
    system_order = max((order[func] for func in funcs))
    is_higher_order = system_order > 1
    is_second_order = system_order == 2 and all((order[func] == 2 for func in funcs))
    try:
        canon_eqs = canonical_odes(eqs, funcs, t) if not is_canon else [eqs]
        if len(canon_eqs) == 1:
            (As, b) = linear_ode_to_matrix(canon_eqs[0], funcs, t, system_order)
        else:
            match = {'is_implicit': True, 'canon_eqs': canon_eqs}
            return match
    except ODENonlinearError:
        return None
    is_linear = True
    is_homogeneous = True if b.is_zero_matrix else False
    match = {'no_of_equation': len(eqs), 'eq': eqs, 'func': funcs, 'order': order, 'is_linear': is_linear, 'is_homogeneous': is_homogeneous, 'is_general': True}
    if not is_homogeneous:
        match['rhs'] = b
    is_constant = all((_matrix_is_constant(A_, t) for A_ in As))
    if not is_higher_order:
        A = As[1]
        match['func_coeff'] = A
        is_constant = _matrix_is_constant(A, t)
        match['is_constant'] = is_constant
        try:
            system_info = linodesolve_type(A, t, b=b)
        except NotImplementedError:
            return None
        match.update(system_info)
        antiderivative = match.pop('antiderivative')
        if not is_constant:
            match['commutative_antiderivative'] = antiderivative
        return match
    else:
        match['type_of_equation'] = 'type0'
        if is_second_order:
            (A1, A0) = As[1:]
            match_second_order = _match_second_order_type(A1, A0, t)
            match.update(match_second_order)
            match['is_second_order'] = True
        if match['type_of_equation'] == 'type0' and (not is_constant):
            is_euler = _is_euler_system(As, t)
            if is_euler:
                t_ = Symbol('{}_'.format(t))
                match.update({'is_transformed': True, 'type_of_equation': 'type1', 't_': t_})
            else:
                is_jordan = lambda M: M == Matrix.jordan_block(M.shape[0], M[0, 0])
                terms = _factor_matrix(As[-1], t)
                if all((A.is_zero_matrix for A in As[1:-1])) and terms is not None and (not is_jordan(terms[1])):
                    (P, J) = terms[1].jordan_form()
                    match.update({'type_of_equation': 'type2', 'J': J, 'f(t)': terms[0], 'P': P, 'is_transformed': True})
            if match['type_of_equation'] != 'type0' and is_second_order:
                match.pop('is_second_order', None)
        match['is_higher_order'] = is_higher_order
        return match

def _preprocess_eqs(eqs):
    if False:
        for i in range(10):
            print('nop')
    processed_eqs = []
    for eq in eqs:
        processed_eqs.append(eq if isinstance(eq, Equality) else Eq(eq, 0))
    return processed_eqs

def _eqs2dict(eqs, funcs):
    if False:
        for i in range(10):
            print('nop')
    eqsorig = {}
    eqsmap = {}
    funcset = set(funcs)
    for eq in eqs:
        (f1,) = eq.lhs.atoms(AppliedUndef)
        f2s = eq.rhs.atoms(AppliedUndef) - {f1} & funcset
        eqsmap[f1] = f2s
        eqsorig[f1] = eq
    return (eqsmap, eqsorig)

def _dict2graph(d):
    if False:
        print('Hello World!')
    nodes = list(d)
    edges = [(f1, f2) for (f1, f2s) in d.items() for f2 in f2s]
    G = (nodes, edges)
    return G

def _is_type1(scc, t):
    if False:
        for i in range(10):
            print('nop')
    (eqs, funcs) = scc
    try:
        ((A1, A0), b) = linear_ode_to_matrix(eqs, funcs, t, 1)
    except (ODENonlinearError, ODEOrderError):
        return False
    if _matrix_is_constant(A0, t) and b.is_zero_matrix:
        return True
    return False

def _combine_type1_subsystems(subsystem, funcs, t):
    if False:
        for i in range(10):
            print('nop')
    indices = [i for (i, sys) in enumerate(zip(subsystem, funcs)) if _is_type1(sys, t)]
    remove = set()
    for (ip, i) in enumerate(indices):
        for j in indices[ip + 1:]:
            if any((eq2.has(funcs[i]) for eq2 in subsystem[j])):
                subsystem[j] = subsystem[i] + subsystem[j]
                remove.add(i)
    subsystem = [sys for (i, sys) in enumerate(subsystem) if i not in remove]
    return subsystem

def _component_division(eqs, funcs, t):
    if False:
        return 10
    (eqsmap, eqsorig) = _eqs2dict(eqs, funcs)
    subsystems = []
    for cc in connected_components(_dict2graph(eqsmap)):
        eqsmap_c = {f: eqsmap[f] for f in cc}
        sccs = strongly_connected_components(_dict2graph(eqsmap_c))
        subsystem = [[eqsorig[f] for f in scc] for scc in sccs]
        subsystem = _combine_type1_subsystems(subsystem, sccs, t)
        subsystems.append(subsystem)
    return subsystems

def _linear_ode_solver(match):
    if False:
        return 10
    t = match['t']
    funcs = match['func']
    rhs = match.get('rhs', None)
    tau = match.get('tau', None)
    t = match['t_'] if 't_' in match else t
    A = match['func_coeff']
    B = match.get('commutative_antiderivative', None)
    type = match['type_of_equation']
    sol_vector = linodesolve(A, t, b=rhs, B=B, type=type, tau=tau)
    sol = [Eq(f, s) for (f, s) in zip(funcs, sol_vector)]
    return sol

def _select_equations(eqs, funcs, key=lambda x: x):
    if False:
        for i in range(10):
            print('nop')
    eq_dict = {e.lhs: e.rhs for e in eqs}
    return [Eq(f, eq_dict[key(f)]) for f in funcs]

def _higher_order_ode_solver(match):
    if False:
        return 10
    eqs = match['eq']
    funcs = match['func']
    t = match['t']
    sysorder = match['order']
    type = match.get('type_of_equation', 'type0')
    is_second_order = match.get('is_second_order', False)
    is_transformed = match.get('is_transformed', False)
    is_euler = is_transformed and type == 'type1'
    is_higher_order_type2 = is_transformed and type == 'type2' and ('P' in match)
    if is_second_order:
        (new_eqs, new_funcs) = _second_order_to_first_order(eqs, funcs, t, A1=match.get('A1', None), A0=match.get('A0', None), b=match.get('rhs', None), type=type, t_=match.get('t_', None))
    else:
        (new_eqs, new_funcs) = _higher_order_to_first_order(eqs, sysorder, t, funcs=funcs, type=type, J=match.get('J', None), f_t=match.get('f(t)', None), P=match.get('P', None), b=match.get('rhs', None))
    if is_transformed:
        t = match.get('t_', t)
    if not is_higher_order_type2:
        new_eqs = _select_equations(new_eqs, [f.diff(t) for f in new_funcs])
    sol = None
    try:
        if not is_higher_order_type2:
            sol = _strong_component_solver(new_eqs, new_funcs, t)
    except NotImplementedError:
        sol = None
    if sol is None:
        try:
            sol = _component_solver(new_eqs, new_funcs, t)
        except NotImplementedError:
            sol = None
    if sol is None:
        return sol
    is_second_order_type2 = is_second_order and type == 'type2'
    underscores = '__' if is_transformed else '_'
    sol = _select_equations(sol, funcs, key=lambda x: Function(Dummy('{}{}0'.format(x.func.__name__, underscores)))(t))
    if match.get('is_transformed', False):
        if is_second_order_type2:
            g_t = match['g(t)']
            tau = match['tau']
            sol = [Eq(s.lhs, s.rhs.subs(t, tau) * g_t) for s in sol]
        elif is_euler:
            t = match['t']
            tau = match['t_']
            sol = [s.subs(tau, log(t)) for s in sol]
        elif is_higher_order_type2:
            P = match['P']
            sol_vector = P * Matrix([s.rhs for s in sol])
            sol = [Eq(f, s) for (f, s) in zip(funcs, sol_vector)]
    return sol

def _strong_component_solver(eqs, funcs, t):
    if False:
        print('Hello World!')
    from sympy.solvers.ode.ode import dsolve, constant_renumber
    match = _classify_linear_system(eqs, funcs, t, is_canon=True)
    sol = None
    if match:
        match['t'] = t
        if match.get('is_higher_order', False):
            sol = _higher_order_ode_solver(match)
        elif match.get('is_linear', False):
            sol = _linear_ode_solver(match)
        if sol is None and len(eqs) == 1:
            sol = dsolve(eqs[0], func=funcs[0])
            variables = Tuple(eqs[0]).free_symbols
            new_constants = [Dummy() for _ in range(ode_order(eqs[0], funcs[0]))]
            sol = constant_renumber(sol, variables=variables, newconstants=new_constants)
            sol = [sol]
    return sol

def _get_funcs_from_canon(eqs):
    if False:
        for i in range(10):
            print('nop')
    return [eq.lhs.args[0] for eq in eqs]

def _weak_component_solver(wcc, t):
    if False:
        return 10
    eqs = []
    for scc in wcc:
        eqs += scc
    funcs = _get_funcs_from_canon(eqs)
    sol = _strong_component_solver(eqs, funcs, t)
    if sol:
        return sol
    sol = []
    for (j, scc) in enumerate(wcc):
        eqs = scc
        funcs = _get_funcs_from_canon(eqs)
        comp_eqs = [eq.subs({s.lhs: s.rhs for s in sol}) for eq in eqs]
        scc_sol = _strong_component_solver(comp_eqs, funcs, t)
        if scc_sol is None:
            raise NotImplementedError(filldedent('\n                The system of ODEs passed cannot be solved by dsolve_system.\n            '))
        sol += scc_sol
    return sol

def _component_solver(eqs, funcs, t):
    if False:
        while True:
            i = 10
    components = _component_division(eqs, funcs, t)
    sol = []
    for wcc in components:
        sol += _weak_component_solver(wcc, t)
    return sol

def _second_order_to_first_order(eqs, funcs, t, type='auto', A1=None, A0=None, b=None, t_=None):
    if False:
        while True:
            i = 10
    '\n    Expects the system to be in second order and in canonical form\n\n    Explanation\n    ===========\n\n    Reduces a second order system into a first order one depending on the type of second\n    order system.\n    1. "type0": If this is passed, then the system will be reduced to first order by\n                introducing dummy variables.\n    2. "type1": If this is passed, then a particular substitution will be used to reduce the\n                the system into first order.\n    3. "type2": If this is passed, then the system will be transformed with new dependent\n                variables and independent variables. This transformation is a part of solving\n                the corresponding system of ODEs.\n\n    `A1` and `A0` are the coefficient matrices from the system and it is assumed that the\n    second order system has the form given below:\n\n    .. math::\n        A2 * X\'\' = A1 * X\' + A0 * X + b\n\n    Here, $A2$ is the coefficient matrix for the vector $X\'\'$ and $b$ is the non-homogeneous\n    term.\n\n    Default value for `b` is None but if `A1` and `A0` are passed and `b` is not passed, then the\n    system will be assumed homogeneous.\n\n    '
    is_a1 = A1 is None
    is_a0 = A0 is None
    if type == 'type1' and is_a1 or (type == 'type2' and is_a0) or (type == 'auto' and (is_a1 or is_a0)):
        ((A2, A1, A0), b) = linear_ode_to_matrix(eqs, funcs, t, 2)
        if not A2.is_Identity:
            raise ValueError(filldedent('\n                The system must be in its canonical form.\n            '))
    if type == 'auto':
        match = _match_second_order_type(A1, A0, t)
        type = match['type_of_equation']
        A1 = match.get('A1', None)
        A0 = match.get('A0', None)
    sys_order = {func: 2 for func in funcs}
    if type == 'type1':
        if b is None:
            b = zeros(len(eqs))
        eqs = _second_order_subs_type1(A1, b, funcs, t)
        sys_order = {func: 1 for func in funcs}
    if type == 'type2':
        if t_ is None:
            t_ = Symbol('{}_'.format(t))
        t = t_
        (eqs, funcs) = _second_order_subs_type2(A0, funcs, t_)
        sys_order = {func: 2 for func in funcs}
    return _higher_order_to_first_order(eqs, sys_order, t, funcs=funcs)

def _higher_order_type2_to_sub_systems(J, f_t, funcs, t, max_order, b=None, P=None):
    if False:
        i = 10
        return i + 15
    if J is None or f_t is None or (not _matrix_is_constant(J, t)):
        raise ValueError(filldedent("\n            Correctly input for args 'A' and 'f_t' for Linear, Higher Order,\n            Type 2\n        "))
    if P is None and b is not None and (not b.is_zero_matrix):
        raise ValueError(filldedent("\n            Provide the keyword 'P' for matrix P in A = P * J * P-1.\n        "))
    new_funcs = Matrix([Function(Dummy('{}__0'.format(f.func.__name__)))(t) for f in funcs])
    new_eqs = new_funcs.diff(t, max_order) - f_t * J * new_funcs
    if b is not None and (not b.is_zero_matrix):
        new_eqs -= P.inv() * b
    new_eqs = canonical_odes(new_eqs, new_funcs, t)[0]
    return (new_eqs, new_funcs)

def _higher_order_to_first_order(eqs, sys_order, t, funcs=None, type='type0', **kwargs):
    if False:
        print('Hello World!')
    if funcs is None:
        funcs = sys_order.keys()
    if type == 'type1':
        t_ = Symbol('{}_'.format(t))
        new_funcs = [Function(Dummy('{}_'.format(f.func.__name__)))(t_) for f in funcs]
        max_order = max((sys_order[func] for func in funcs))
        subs_dict = dict(zip(funcs, new_funcs))
        subs_dict[t] = exp(t_)
        free_function = Function(Dummy())

        def _get_coeffs_from_subs_expression(expr):
            if False:
                while True:
                    i = 10
            if isinstance(expr, Subs):
                free_symbol = expr.args[1][0]
                term = expr.args[0]
                return {ode_order(term, free_symbol): 1}
            if isinstance(expr, Mul):
                coeff = expr.args[0]
                order = list(_get_coeffs_from_subs_expression(expr.args[1]).keys())[0]
                return {order: coeff}
            if isinstance(expr, Add):
                coeffs = {}
                for arg in expr.args:
                    if isinstance(arg, Mul):
                        coeffs.update(_get_coeffs_from_subs_expression(arg))
                    else:
                        order = list(_get_coeffs_from_subs_expression(arg).keys())[0]
                        coeffs[order] = 1
                return coeffs
        for o in range(1, max_order + 1):
            expr = free_function(log(t_)).diff(t_, o) * t_ ** o
            coeff_dict = _get_coeffs_from_subs_expression(expr)
            coeffs = [coeff_dict[order] if order in coeff_dict else 0 for order in range(o + 1)]
            expr_to_subs = sum((free_function(t_).diff(t_, i) * c for (i, c) in enumerate(coeffs))) / t ** o
            subs_dict.update({f.diff(t, o): expr_to_subs.subs(free_function(t_), nf) for (f, nf) in zip(funcs, new_funcs)})
        new_eqs = [eq.subs(subs_dict) for eq in eqs]
        new_sys_order = {nf: sys_order[f] for (f, nf) in zip(funcs, new_funcs)}
        new_eqs = canonical_odes(new_eqs, new_funcs, t_)[0]
        return _higher_order_to_first_order(new_eqs, new_sys_order, t_, funcs=new_funcs)
    if type == 'type2':
        J = kwargs.get('J', None)
        f_t = kwargs.get('f_t', None)
        b = kwargs.get('b', None)
        P = kwargs.get('P', None)
        max_order = max((sys_order[func] for func in funcs))
        return _higher_order_type2_to_sub_systems(J, f_t, funcs, t, max_order, P=P, b=b)
    new_funcs = []
    for prev_func in funcs:
        func_name = prev_func.func.__name__
        func = Function(Dummy('{}_0'.format(func_name)))(t)
        new_funcs.append(func)
        subs_dict = {prev_func: func}
        new_eqs = []
        for i in range(1, sys_order[prev_func]):
            new_func = Function(Dummy('{}_{}'.format(func_name, i)))(t)
            subs_dict[prev_func.diff(t, i)] = new_func
            new_funcs.append(new_func)
            prev_f = subs_dict[prev_func.diff(t, i - 1)]
            new_eq = Eq(prev_f.diff(t), new_func)
            new_eqs.append(new_eq)
        eqs = [eq.subs(subs_dict) for eq in eqs] + new_eqs
    return (eqs, new_funcs)

def dsolve_system(eqs, funcs=None, t=None, ics=None, doit=False, simplify=True):
    if False:
        return 10
    '\n    Solves any(supported) system of Ordinary Differential Equations\n\n    Explanation\n    ===========\n\n    This function takes a system of ODEs as an input, determines if the\n    it is solvable by this function, and returns the solution if found any.\n\n    This function can handle:\n    1. Linear, First Order, Constant coefficient homogeneous system of ODEs\n    2. Linear, First Order, Constant coefficient non-homogeneous system of ODEs\n    3. Linear, First Order, non-constant coefficient homogeneous system of ODEs\n    4. Linear, First Order, non-constant coefficient non-homogeneous system of ODEs\n    5. Any implicit system which can be divided into system of ODEs which is of the above 4 forms\n    6. Any higher order linear system of ODEs that can be reduced to one of the 5 forms of systems described above.\n\n    The types of systems described above are not limited by the number of equations, i.e. this\n    function can solve the above types irrespective of the number of equations in the system passed.\n    But, the bigger the system, the more time it will take to solve the system.\n\n    This function returns a list of solutions. Each solution is a list of equations where LHS is\n    the dependent variable and RHS is an expression in terms of the independent variable.\n\n    Among the non constant coefficient types, not all the systems are solvable by this function. Only\n    those which have either a coefficient matrix with a commutative antiderivative or those systems which\n    may be divided further so that the divided systems may have coefficient matrix with commutative antiderivative.\n\n    Parameters\n    ==========\n\n    eqs : List\n        system of ODEs to be solved\n    funcs : List or None\n        List of dependent variables that make up the system of ODEs\n    t : Symbol or None\n        Independent variable in the system of ODEs\n    ics : Dict or None\n        Set of initial boundary/conditions for the system of ODEs\n    doit : Boolean\n        Evaluate the solutions if True. Default value is True. Can be\n        set to false if the integral evaluation takes too much time and/or\n        is not required.\n    simplify: Boolean\n        Simplify the solutions for the systems. Default value is True.\n        Can be set to false if simplification takes too much time and/or\n        is not required.\n\n    Examples\n    ========\n\n    >>> from sympy import symbols, Eq, Function\n    >>> from sympy.solvers.ode.systems import dsolve_system\n    >>> f, g = symbols("f g", cls=Function)\n    >>> x = symbols("x")\n\n    >>> eqs = [Eq(f(x).diff(x), g(x)), Eq(g(x).diff(x), f(x))]\n    >>> dsolve_system(eqs)\n    [[Eq(f(x), -C1*exp(-x) + C2*exp(x)), Eq(g(x), C1*exp(-x) + C2*exp(x))]]\n\n    You can also pass the initial conditions for the system of ODEs:\n\n    >>> dsolve_system(eqs, ics={f(0): 1, g(0): 0})\n    [[Eq(f(x), exp(x)/2 + exp(-x)/2), Eq(g(x), exp(x)/2 - exp(-x)/2)]]\n\n    Optionally, you can pass the dependent variables and the independent\n    variable for which the system is to be solved:\n\n    >>> funcs = [f(x), g(x)]\n    >>> dsolve_system(eqs, funcs=funcs, t=x)\n    [[Eq(f(x), -C1*exp(-x) + C2*exp(x)), Eq(g(x), C1*exp(-x) + C2*exp(x))]]\n\n    Lets look at an implicit system of ODEs:\n\n    >>> eqs = [Eq(f(x).diff(x)**2, g(x)**2), Eq(g(x).diff(x), g(x))]\n    >>> dsolve_system(eqs)\n    [[Eq(f(x), C1 - C2*exp(x)), Eq(g(x), C2*exp(x))], [Eq(f(x), C1 + C2*exp(x)), Eq(g(x), C2*exp(x))]]\n\n    Returns\n    =======\n\n    List of List of Equations\n\n    Raises\n    ======\n\n    NotImplementedError\n        When the system of ODEs is not solvable by this function.\n    ValueError\n        When the parameters passed are not in the required form.\n\n    '
    from sympy.solvers.ode.ode import solve_ics, _extract_funcs, constant_renumber
    if not iterable(eqs):
        raise ValueError(filldedent('\n            List of equations should be passed. The input is not valid.\n        '))
    eqs = _preprocess_eqs(eqs)
    if funcs is not None and (not isinstance(funcs, list)):
        raise ValueError(filldedent('\n            Input to the funcs should be a list of functions.\n        '))
    if funcs is None:
        funcs = _extract_funcs(eqs)
    if any((len(func.args) != 1 for func in funcs)):
        raise ValueError(filldedent('\n            dsolve_system can solve a system of ODEs with only one independent\n            variable.\n        '))
    if len(eqs) != len(funcs):
        raise ValueError(filldedent('\n            Number of equations and number of functions do not match\n        '))
    if t is not None and (not isinstance(t, Symbol)):
        raise ValueError(filldedent('\n            The independent variable must be of type Symbol\n        '))
    if t is None:
        t = list(list(eqs[0].atoms(Derivative))[0].atoms(Symbol))[0]
    sols = []
    canon_eqs = canonical_odes(eqs, funcs, t)
    for canon_eq in canon_eqs:
        try:
            sol = _strong_component_solver(canon_eq, funcs, t)
        except NotImplementedError:
            sol = None
        if sol is None:
            sol = _component_solver(canon_eq, funcs, t)
        sols.append(sol)
    if sols:
        final_sols = []
        variables = Tuple(*eqs).free_symbols
        for sol in sols:
            sol = _select_equations(sol, funcs)
            sol = constant_renumber(sol, variables=variables)
            if ics:
                constants = Tuple(*sol).free_symbols - variables
                solved_constants = solve_ics(sol, funcs, constants, ics)
                sol = [s.subs(solved_constants) for s in sol]
            if simplify:
                constants = Tuple(*sol).free_symbols - variables
                sol = simpsol(sol, [t], constants, doit=doit)
            final_sols.append(sol)
        sols = final_sols
    return sols