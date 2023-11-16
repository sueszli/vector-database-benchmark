from sympy.core.add import Add
from sympy.core.assumptions import check_assumptions
from sympy.core.containers import Tuple
from sympy.core.exprtools import factor_terms
from sympy.core.function import _mexpand
from sympy.core.mul import Mul
from sympy.core.numbers import Rational, int_valued
from sympy.core.intfunc import igcdex, ilcm, igcd, integer_nthroot, isqrt
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.sorting import default_sort_key, ordered
from sympy.core.symbol import Symbol, symbols
from sympy.core.sympify import _sympify
from sympy.functions.elementary.complexes import sign
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices.dense import MutableDenseMatrix as Matrix
from sympy.ntheory.factor_ import divisors, factorint, multiplicity, perfect_power
from sympy.ntheory.generate import nextprime
from sympy.ntheory.primetest import is_square, isprime
from sympy.ntheory.residue_ntheory import sqrt_mod
from sympy.polys.polyerrors import GeneratorsNeeded
from sympy.polys.polytools import Poly, factor_list
from sympy.simplify.simplify import signsimp
from sympy.solvers.solveset import solveset_real
from sympy.utilities import numbered_symbols
from sympy.utilities.misc import as_int, filldedent
from sympy.utilities.iterables import is_sequence, subsets, permute_signs, signed_permutations, ordered_partitions
__all__ = ['diophantine', 'classify_diop']

class DiophantineSolutionSet(set):
    """
    Container for a set of solutions to a particular diophantine equation.

    The base representation is a set of tuples representing each of the solutions.

    Parameters
    ==========

    symbols : list
        List of free symbols in the original equation.
    parameters: list
        List of parameters to be used in the solution.

    Examples
    ========

    Adding solutions:

        >>> from sympy.solvers.diophantine.diophantine import DiophantineSolutionSet
        >>> from sympy.abc import x, y, t, u
        >>> s1 = DiophantineSolutionSet([x, y], [t, u])
        >>> s1
        set()
        >>> s1.add((2, 3))
        >>> s1.add((-1, u))
        >>> s1
        {(-1, u), (2, 3)}
        >>> s2 = DiophantineSolutionSet([x, y], [t, u])
        >>> s2.add((3, 4))
        >>> s1.update(*s2)
        >>> s1
        {(-1, u), (2, 3), (3, 4)}

    Conversion of solutions into dicts:

        >>> list(s1.dict_iterator())
        [{x: -1, y: u}, {x: 2, y: 3}, {x: 3, y: 4}]

    Substituting values:

        >>> s3 = DiophantineSolutionSet([x, y], [t, u])
        >>> s3.add((t**2, t + u))
        >>> s3
        {(t**2, t + u)}
        >>> s3.subs({t: 2, u: 3})
        {(4, 5)}
        >>> s3.subs(t, -1)
        {(1, u - 1)}
        >>> s3.subs(t, 3)
        {(9, u + 3)}

    Evaluation at specific values. Positional arguments are given in the same order as the parameters:

        >>> s3(-2, 3)
        {(4, 1)}
        >>> s3(5)
        {(25, u + 5)}
        >>> s3(None, 2)
        {(t**2, t + 2)}
    """

    def __init__(self, symbols_seq, parameters):
        if False:
            while True:
                i = 10
        super().__init__()
        if not is_sequence(symbols_seq):
            raise ValueError('Symbols must be given as a sequence.')
        if not is_sequence(parameters):
            raise ValueError('Parameters must be given as a sequence.')
        self.symbols = tuple(symbols_seq)
        self.parameters = tuple(parameters)

    def add(self, solution):
        if False:
            return 10
        if len(solution) != len(self.symbols):
            raise ValueError('Solution should have a length of %s, not %s' % (len(self.symbols), len(solution)))
        args = set(solution)
        for i in range(len(solution)):
            x = solution[i]
            if not type(x) is int and (-x).is_Symbol and (-x not in args):
                solution = [_.subs(-x, x) for _ in solution]
        super().add(Tuple(*solution))

    def update(self, *solutions):
        if False:
            return 10
        for solution in solutions:
            self.add(solution)

    def dict_iterator(self):
        if False:
            return 10
        for solution in ordered(self):
            yield dict(zip(self.symbols, solution))

    def subs(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        result = DiophantineSolutionSet(self.symbols, self.parameters)
        for solution in self:
            result.add(solution.subs(*args, **kwargs))
        return result

    def __call__(self, *args):
        if False:
            print('Hello World!')
        if len(args) > len(self.parameters):
            raise ValueError('Evaluation should have at most %s values, not %s' % (len(self.parameters), len(args)))
        rep = {p: v for (p, v) in zip(self.parameters, args) if v is not None}
        return self.subs(rep)

class DiophantineEquationType:
    """
    Internal representation of a particular diophantine equation type.

    Parameters
    ==========

    equation :
        The diophantine equation that is being solved.
    free_symbols : list (optional)
        The symbols being solved for.

    Attributes
    ==========

    total_degree :
        The maximum of the degrees of all terms in the equation
    homogeneous :
        Does the equation contain a term of degree 0
    homogeneous_order :
        Does the equation contain any coefficient that is in the symbols being solved for
    dimension :
        The number of symbols being solved for
    """
    name = None

    def __init__(self, equation, free_symbols=None):
        if False:
            while True:
                i = 10
        self.equation = _sympify(equation).expand(force=True)
        if free_symbols is not None:
            self.free_symbols = free_symbols
        else:
            self.free_symbols = list(self.equation.free_symbols)
            self.free_symbols.sort(key=default_sort_key)
        if not self.free_symbols:
            raise ValueError('equation should have 1 or more free symbols')
        self.coeff = self.equation.as_coefficients_dict()
        if not all((int_valued(c) for c in self.coeff.values())):
            raise TypeError('Coefficients should be Integers')
        self.total_degree = Poly(self.equation).total_degree()
        self.homogeneous = 1 not in self.coeff
        self.homogeneous_order = not set(self.coeff) & set(self.free_symbols)
        self.dimension = len(self.free_symbols)
        self._parameters = None

    def matches(self):
        if False:
            print('Hello World!')
        '\n        Determine whether the given equation can be matched to the particular equation type.\n        '
        return False

    @property
    def n_parameters(self):
        if False:
            for i in range(10):
                print('nop')
        return self.dimension

    @property
    def parameters(self):
        if False:
            print('Hello World!')
        if self._parameters is None:
            self._parameters = symbols('t_:%i' % (self.n_parameters,), integer=True)
        return self._parameters

    def solve(self, parameters=None, limit=None) -> DiophantineSolutionSet:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError('No solver has been written for %s.' % self.name)

    def pre_solve(self, parameters=None):
        if False:
            for i in range(10):
                print('nop')
        if not self.matches():
            raise ValueError('This equation does not match the %s equation type.' % self.name)
        if parameters is not None:
            if len(parameters) != self.n_parameters:
                raise ValueError('Expected %s parameter(s) but got %s' % (self.n_parameters, len(parameters)))
        self._parameters = parameters

class Univariate(DiophantineEquationType):
    """
    Representation of a univariate diophantine equation.

    A univariate diophantine equation is an equation of the form
    `a_{0} + a_{1}x + a_{2}x^2 + .. + a_{n}x^n = 0` where `a_{1}, a_{2}, ..a_{n}` are
    integer constants and `x` is an integer variable.

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import Univariate
    >>> from sympy.abc import x
    >>> Univariate((x - 2)*(x - 3)**2).solve() # solves equation (x - 2)*(x - 3)**2 == 0
    {(2,), (3,)}

    """
    name = 'univariate'

    def matches(self):
        if False:
            return 10
        return self.dimension == 1

    def solve(self, parameters=None, limit=None):
        if False:
            while True:
                i = 10
        self.pre_solve(parameters)
        result = DiophantineSolutionSet(self.free_symbols, parameters=self.parameters)
        for i in solveset_real(self.equation, self.free_symbols[0]).intersect(S.Integers):
            result.add((i,))
        return result

class Linear(DiophantineEquationType):
    """
    Representation of a linear diophantine equation.

    A linear diophantine equation is an equation of the form `a_{1}x_{1} +
    a_{2}x_{2} + .. + a_{n}x_{n} = 0` where `a_{1}, a_{2}, ..a_{n}` are
    integer constants and `x_{1}, x_{2}, ..x_{n}` are integer variables.

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import Linear
    >>> from sympy.abc import x, y, z
    >>> l1 = Linear(2*x - 3*y - 5)
    >>> l1.matches() # is this equation linear
    True
    >>> l1.solve() # solves equation 2*x - 3*y - 5 == 0
    {(3*t_0 - 5, 2*t_0 - 5)}

    Here x = -3*t_0 - 5 and y = -2*t_0 - 5

    >>> Linear(2*x - 3*y - 4*z -3).solve()
    {(t_0, 2*t_0 + 4*t_1 + 3, -t_0 - 3*t_1 - 3)}

    """
    name = 'linear'

    def matches(self):
        if False:
            i = 10
            return i + 15
        return self.total_degree == 1

    def solve(self, parameters=None, limit=None):
        if False:
            i = 10
            return i + 15
        self.pre_solve(parameters)
        coeff = self.coeff
        var = self.free_symbols
        if 1 in coeff:
            c = -coeff[1]
        else:
            c = 0
        result = DiophantineSolutionSet(var, parameters=self.parameters)
        params = result.parameters
        if len(var) == 1:
            (q, r) = divmod(c, coeff[var[0]])
            if not r:
                result.add((q,))
                return result
            else:
                return result
        "\n        base_solution_linear() can solve diophantine equations of the form:\n\n        a*x + b*y == c\n\n        We break down multivariate linear diophantine equations into a\n        series of bivariate linear diophantine equations which can then\n        be solved individually by base_solution_linear().\n\n        Consider the following:\n\n        a_0*x_0 + a_1*x_1 + a_2*x_2 == c\n\n        which can be re-written as:\n\n        a_0*x_0 + g_0*y_0 == c\n\n        where\n\n        g_0 == gcd(a_1, a_2)\n\n        and\n\n        y == (a_1*x_1)/g_0 + (a_2*x_2)/g_0\n\n        This leaves us with two binary linear diophantine equations.\n        For the first equation:\n\n        a == a_0\n        b == g_0\n        c == c\n\n        For the second:\n\n        a == a_1/g_0\n        b == a_2/g_0\n        c == the solution we find for y_0 in the first equation.\n\n        The arrays A and B are the arrays of integers used for\n        'a' and 'b' in each of the n-1 bivariate equations we solve.\n        "
        A = [coeff[v] for v in var]
        B = []
        if len(var) > 2:
            B.append(igcd(A[-2], A[-1]))
            A[-2] = A[-2] // B[0]
            A[-1] = A[-1] // B[0]
            for i in range(len(A) - 3, 0, -1):
                gcd = igcd(B[0], A[i])
                B[0] = B[0] // gcd
                A[i] = A[i] // gcd
                B.insert(0, gcd)
        B.append(A[-1])
        "\n        Consider the trivariate linear equation:\n\n        4*x_0 + 6*x_1 + 3*x_2 == 2\n\n        This can be re-written as:\n\n        4*x_0 + 3*y_0 == 2\n\n        where\n\n        y_0 == 2*x_1 + x_2\n        (Note that gcd(3, 6) == 3)\n\n        The complete integral solution to this equation is:\n\n        x_0 ==  2 + 3*t_0\n        y_0 == -2 - 4*t_0\n\n        where 't_0' is any integer.\n\n        Now that we have a solution for 'x_0', find 'x_1' and 'x_2':\n\n        2*x_1 + x_2 == -2 - 4*t_0\n\n        We can then solve for '-2' and '-4' independently,\n        and combine the results:\n\n        2*x_1a + x_2a == -2\n        x_1a == 0 + t_0\n        x_2a == -2 - 2*t_0\n\n        2*x_1b + x_2b == -4*t_0\n        x_1b == 0*t_0 + t_1\n        x_2b == -4*t_0 - 2*t_1\n\n        ==>\n\n        x_1 == t_0 + t_1\n        x_2 == -2 - 6*t_0 - 2*t_1\n\n        where 't_0' and 't_1' are any integers.\n\n        Note that:\n\n        4*(2 + 3*t_0) + 6*(t_0 + t_1) + 3*(-2 - 6*t_0 - 2*t_1) == 2\n\n        for any integral values of 't_0', 't_1'; as required.\n\n        This method is generalised for many variables, below.\n\n        "
        solutions = []
        for (Ai, Bi) in zip(A, B):
            (tot_x, tot_y) = ([], [])
            for (j, arg) in enumerate(Add.make_args(c)):
                if arg.is_Integer:
                    (k, p) = (arg, S.One)
                    pnew = params[0]
                else:
                    (k, p) = arg.as_coeff_Mul()
                    pnew = params[params.index(p) + 1]
                sol = (sol_x, sol_y) = base_solution_linear(k, Ai, Bi, pnew)
                if p is S.One:
                    if None in sol:
                        return result
                else:
                    if isinstance(sol_x, Add):
                        sol_x = sol_x.args[0] * p + sol_x.args[1]
                    if isinstance(sol_y, Add):
                        sol_y = sol_y.args[0] * p + sol_y.args[1]
                tot_x.append(sol_x)
                tot_y.append(sol_y)
            solutions.append(Add(*tot_x))
            c = Add(*tot_y)
        solutions.append(c)
        result.add(solutions)
        return result

class BinaryQuadratic(DiophantineEquationType):
    """
    Representation of a binary quadratic diophantine equation.

    A binary quadratic diophantine equation is an equation of the
    form `Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0`, where `A, B, C, D, E,
    F` are integer constants and `x` and `y` are integer variables.

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy.solvers.diophantine.diophantine import BinaryQuadratic
    >>> b1 = BinaryQuadratic(x**3 + y**2 + 1)
    >>> b1.matches()
    False
    >>> b2 = BinaryQuadratic(x**2 + y**2 + 2*x + 2*y + 2)
    >>> b2.matches()
    True
    >>> b2.solve()
    {(-1, -1)}

    References
    ==========

    .. [1] Methods to solve Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0, [online],
          Available: https://www.alpertron.com.ar/METHODS.HTM
    .. [2] Solving the equation ax^2+ bxy + cy^2 + dx + ey + f= 0, [online],
          Available: https://web.archive.org/web/20160323033111/http://www.jpr2718.org/ax2p.pdf

    """
    name = 'binary_quadratic'

    def matches(self):
        if False:
            for i in range(10):
                print('nop')
        return self.total_degree == 2 and self.dimension == 2

    def solve(self, parameters=None, limit=None) -> DiophantineSolutionSet:
        if False:
            i = 10
            return i + 15
        self.pre_solve(parameters)
        var = self.free_symbols
        coeff = self.coeff
        (x, y) = var
        A = coeff[x ** 2]
        B = coeff[x * y]
        C = coeff[y ** 2]
        D = coeff[x]
        E = coeff[y]
        F = coeff[S.One]
        (A, B, C, D, E, F) = [as_int(i) for i in _remove_gcd(A, B, C, D, E, F)]
        result = DiophantineSolutionSet(var, self.parameters)
        (t, u) = result.parameters
        discr = B ** 2 - 4 * A * C
        if A == 0 and C == 0 and (B != 0):
            if D * E - B * F == 0:
                (q, r) = divmod(E, B)
                if not r:
                    result.add((-q, t))
                (q, r) = divmod(D, B)
                if not r:
                    result.add((t, -q))
            else:
                div = divisors(D * E - B * F)
                div = div + [-term for term in div]
                for d in div:
                    (x0, r) = divmod(d - E, B)
                    if not r:
                        (q, r) = divmod(D * E - B * F, d)
                        if not r:
                            (y0, r) = divmod(q - D, B)
                            if not r:
                                result.add((x0, y0))
        elif discr == 0:
            if A == 0:
                s = BinaryQuadratic(self.equation, free_symbols=[y, x]).solve(parameters=[t, u])
                for soln in s:
                    result.add((soln[1], soln[0]))
            else:
                g = sign(A) * igcd(A, C)
                a = A // g
                c = C // g
                e = sign(B / A)
                sqa = isqrt(a)
                sqc = isqrt(c)
                _c = e * sqc * D - sqa * E
                if not _c:
                    z = Symbol('z', real=True)
                    eq = sqa * g * z ** 2 + D * z + sqa * F
                    roots = solveset_real(eq, z).intersect(S.Integers)
                    for root in roots:
                        ans = diop_solve(sqa * x + e * sqc * y - root)
                        result.add((ans[0], ans[1]))
                elif int_valued(c):
                    solve_x = lambda u: -e * sqc * g * _c * t ** 2 - (E + 2 * e * sqc * g * u) * t - (e * sqc * g * u ** 2 + E * u + e * sqc * F) // _c
                    solve_y = lambda u: sqa * g * _c * t ** 2 + (D + 2 * sqa * g * u) * t + (sqa * g * u ** 2 + D * u + sqa * F) // _c
                    for z0 in range(0, abs(_c)):
                        if divisible(sqa * g * z0 ** 2 + D * z0 + sqa * F, _c) and divisible(e * sqc * g * z0 ** 2 + E * z0 + e * sqc * F, _c):
                            result.add((solve_x(z0), solve_y(z0)))
        elif is_square(discr):
            if A != 0:
                r = sqrt(discr)
                (u, v) = symbols('u, v', integer=True)
                eq = _mexpand(4 * A * r * u * v + 4 * A * D * (B * v + r * u + r * v - B * u) + 2 * A * 4 * A * E * (u - v) + 4 * A * r * 4 * A * F)
                solution = diop_solve(eq, t)
                for (s0, t0) in solution:
                    num = B * t0 + r * s0 + r * t0 - B * s0
                    x_0 = S(num) / (4 * A * r)
                    y_0 = S(s0 - t0) / (2 * r)
                    if isinstance(s0, Symbol) or isinstance(t0, Symbol):
                        if len(check_param(x_0, y_0, 4 * A * r, parameters)) > 0:
                            ans = check_param(x_0, y_0, 4 * A * r, parameters)
                            result.update(*ans)
                    elif x_0.is_Integer and y_0.is_Integer:
                        if is_solution_quad(var, coeff, x_0, y_0):
                            result.add((x_0, y_0))
            else:
                s = BinaryQuadratic(self.equation, free_symbols=var[::-1]).solve(parameters=[t, u])
                while s:
                    result.add(s.pop()[::-1])
        else:
            (P, Q) = _transformation_to_DN(var, coeff)
            (D, N) = _find_DN(var, coeff)
            solns_pell = diop_DN(D, N)
            if D < 0:
                for (x0, y0) in solns_pell:
                    for x in [-x0, x0]:
                        for y in [-y0, y0]:
                            s = P * Matrix([x, y]) + Q
                            try:
                                result.add([as_int(_) for _ in s])
                            except ValueError:
                                pass
            else:
                solns_pell = set(solns_pell)
                for (X, Y) in list(solns_pell):
                    solns_pell.add((-X, -Y))
                a = diop_DN(D, 1)
                T = a[0][0]
                U = a[0][1]
                if all((int_valued(_) for _ in P[:4] + Q[:2])):
                    for (r, s) in solns_pell:
                        _a = (r + s * sqrt(D)) * (T + U * sqrt(D)) ** t
                        _b = (r - s * sqrt(D)) * (T - U * sqrt(D)) ** t
                        x_n = _mexpand(S(_a + _b) / 2)
                        y_n = _mexpand(S(_a - _b) / (2 * sqrt(D)))
                        s = P * Matrix([x_n, y_n]) + Q
                        result.add(s)
                else:
                    L = ilcm(*[_.q for _ in P[:4] + Q[:2]])
                    k = 1
                    T_k = T
                    U_k = U
                    while (T_k - 1) % L != 0 or U_k % L != 0:
                        (T_k, U_k) = (T_k * T + D * U_k * U, T_k * U + U_k * T)
                        k += 1
                    for (X, Y) in solns_pell:
                        for i in range(k):
                            if all((int_valued(_) for _ in P * Matrix([X, Y]) + Q)):
                                _a = (X + sqrt(D) * Y) * (T_k + sqrt(D) * U_k) ** t
                                _b = (X - sqrt(D) * Y) * (T_k - sqrt(D) * U_k) ** t
                                Xt = S(_a + _b) / 2
                                Yt = S(_a - _b) / (2 * sqrt(D))
                                s = P * Matrix([Xt, Yt]) + Q
                                result.add(s)
                            (X, Y) = (X * T + D * U * Y, X * U + Y * T)
        return result

class InhomogeneousTernaryQuadratic(DiophantineEquationType):
    """

    Representation of an inhomogeneous ternary quadratic.

    No solver is currently implemented for this equation type.

    """
    name = 'inhomogeneous_ternary_quadratic'

    def matches(self):
        if False:
            i = 10
            return i + 15
        if not (self.total_degree == 2 and self.dimension == 3):
            return False
        if not self.homogeneous:
            return False
        return not self.homogeneous_order

class HomogeneousTernaryQuadraticNormal(DiophantineEquationType):
    """
    Representation of a homogeneous ternary quadratic normal diophantine equation.

    Examples
    ========

    >>> from sympy.abc import x, y, z
    >>> from sympy.solvers.diophantine.diophantine import HomogeneousTernaryQuadraticNormal
    >>> HomogeneousTernaryQuadraticNormal(4*x**2 - 5*y**2 + z**2).solve()
    {(1, 2, 4)}

    """
    name = 'homogeneous_ternary_quadratic_normal'

    def matches(self):
        if False:
            while True:
                i = 10
        if not (self.total_degree == 2 and self.dimension == 3):
            return False
        if not self.homogeneous:
            return False
        if not self.homogeneous_order:
            return False
        nonzero = [k for k in self.coeff if self.coeff[k]]
        return len(nonzero) == 3 and all((i ** 2 in nonzero for i in self.free_symbols))

    def solve(self, parameters=None, limit=None) -> DiophantineSolutionSet:
        if False:
            for i in range(10):
                print('nop')
        self.pre_solve(parameters)
        var = self.free_symbols
        coeff = self.coeff
        (x, y, z) = var
        a = coeff[x ** 2]
        b = coeff[y ** 2]
        c = coeff[z ** 2]
        ((sqf_of_a, sqf_of_b, sqf_of_c), (a_1, b_1, c_1), (a_2, b_2, c_2)) = sqf_normal(a, b, c, steps=True)
        A = -a_2 * c_2
        B = -b_2 * c_2
        result = DiophantineSolutionSet(var, parameters=self.parameters)
        if A < 0 and B < 0:
            return result
        if sqrt_mod(-b_2 * c_2, a_2) is None or sqrt_mod(-c_2 * a_2, b_2) is None or sqrt_mod(-a_2 * b_2, c_2) is None:
            return result
        (z_0, x_0, y_0) = descent(A, B)
        (z_0, q) = _rational_pq(z_0, abs(c_2))
        x_0 *= q
        y_0 *= q
        (x_0, y_0, z_0) = _remove_gcd(x_0, y_0, z_0)
        if sign(a) == sign(b):
            (x_0, y_0, z_0) = holzer(x_0, y_0, z_0, abs(a_2), abs(b_2), abs(c_2))
        elif sign(a) == sign(c):
            (x_0, z_0, y_0) = holzer(x_0, z_0, y_0, abs(a_2), abs(c_2), abs(b_2))
        else:
            (y_0, z_0, x_0) = holzer(y_0, z_0, x_0, abs(b_2), abs(c_2), abs(a_2))
        x_0 = reconstruct(b_1, c_1, x_0)
        y_0 = reconstruct(a_1, c_1, y_0)
        z_0 = reconstruct(a_1, b_1, z_0)
        sq_lcm = ilcm(sqf_of_a, sqf_of_b, sqf_of_c)
        x_0 = abs(x_0 * sq_lcm // sqf_of_a)
        y_0 = abs(y_0 * sq_lcm // sqf_of_b)
        z_0 = abs(z_0 * sq_lcm // sqf_of_c)
        result.add(_remove_gcd(x_0, y_0, z_0))
        return result

class HomogeneousTernaryQuadratic(DiophantineEquationType):
    """
    Representation of a homogeneous ternary quadratic diophantine equation.

    Examples
    ========

    >>> from sympy.abc import x, y, z
    >>> from sympy.solvers.diophantine.diophantine import HomogeneousTernaryQuadratic
    >>> HomogeneousTernaryQuadratic(x**2 + y**2 - 3*z**2 + x*y).solve()
    {(-1, 2, 1)}
    >>> HomogeneousTernaryQuadratic(3*x**2 + y**2 - 3*z**2 + 5*x*y + y*z).solve()
    {(3, 12, 13)}

    """
    name = 'homogeneous_ternary_quadratic'

    def matches(self):
        if False:
            i = 10
            return i + 15
        if not (self.total_degree == 2 and self.dimension == 3):
            return False
        if not self.homogeneous:
            return False
        if not self.homogeneous_order:
            return False
        nonzero = [k for k in self.coeff if self.coeff[k]]
        return not (len(nonzero) == 3 and all((i ** 2 in nonzero for i in self.free_symbols)))

    def solve(self, parameters=None, limit=None):
        if False:
            for i in range(10):
                print('nop')
        self.pre_solve(parameters)
        _var = self.free_symbols
        coeff = self.coeff
        (x, y, z) = _var
        var = [x, y, z]
        result = DiophantineSolutionSet(var, parameters=self.parameters)

        def unpack_sol(sol):
            if False:
                for i in range(10):
                    print('nop')
            if len(sol) > 0:
                return list(sol)[0]
            return (None, None, None)
        if not any((coeff[i ** 2] for i in var)):
            if coeff[x * z]:
                sols = diophantine(coeff[x * y] * x + coeff[y * z] * z - x * z)
                s = sols.pop()
                min_sum = abs(s[0]) + abs(s[1])
                for r in sols:
                    m = abs(r[0]) + abs(r[1])
                    if m < min_sum:
                        s = r
                        min_sum = m
                result.add(_remove_gcd(s[0], -coeff[x * z], s[1]))
                return result
            else:
                (var[0], var[1]) = (_var[1], _var[0])
                (y_0, x_0, z_0) = unpack_sol(_diop_ternary_quadratic(var, coeff))
                if x_0 is not None:
                    result.add((x_0, y_0, z_0))
                return result
        if coeff[x ** 2] == 0:
            if coeff[y ** 2] == 0:
                (var[0], var[2]) = (_var[2], _var[0])
                (z_0, y_0, x_0) = unpack_sol(_diop_ternary_quadratic(var, coeff))
            else:
                (var[0], var[1]) = (_var[1], _var[0])
                (y_0, x_0, z_0) = unpack_sol(_diop_ternary_quadratic(var, coeff))
        elif coeff[x * y] or coeff[x * z]:
            A = coeff[x ** 2]
            B = coeff[x * y]
            C = coeff[x * z]
            D = coeff[y ** 2]
            E = coeff[y * z]
            F = coeff[z ** 2]
            _coeff = {}
            _coeff[x ** 2] = 4 * A ** 2
            _coeff[y ** 2] = 4 * A * D - B ** 2
            _coeff[z ** 2] = 4 * A * F - C ** 2
            _coeff[y * z] = 4 * A * E - 2 * B * C
            _coeff[x * y] = 0
            _coeff[x * z] = 0
            (x_0, y_0, z_0) = unpack_sol(_diop_ternary_quadratic(var, _coeff))
            if x_0 is None:
                return result
            (p, q) = _rational_pq(B * y_0 + C * z_0, 2 * A)
            (x_0, y_0, z_0) = (x_0 * q - p, y_0 * q, z_0 * q)
        elif coeff[z * y] != 0:
            if coeff[y ** 2] == 0:
                if coeff[z ** 2] == 0:
                    A = coeff[x ** 2]
                    E = coeff[y * z]
                    (b, a) = _rational_pq(-E, A)
                    (x_0, y_0, z_0) = (b, a, b)
                else:
                    (var[0], var[2]) = (_var[2], _var[0])
                    (z_0, y_0, x_0) = unpack_sol(_diop_ternary_quadratic(var, coeff))
            else:
                (var[0], var[1]) = (_var[1], _var[0])
                (y_0, x_0, z_0) = unpack_sol(_diop_ternary_quadratic(var, coeff))
        else:
            (x_0, y_0, z_0) = unpack_sol(_diop_ternary_quadratic_normal(var, coeff))
        if x_0 is None:
            return result
        result.add(_remove_gcd(x_0, y_0, z_0))
        return result

class InhomogeneousGeneralQuadratic(DiophantineEquationType):
    """

    Representation of an inhomogeneous general quadratic.

    No solver is currently implemented for this equation type.

    """
    name = 'inhomogeneous_general_quadratic'

    def matches(self):
        if False:
            for i in range(10):
                print('nop')
        if not (self.total_degree == 2 and self.dimension >= 3):
            return False
        if not self.homogeneous_order:
            return True
        elif any((k.is_Mul for k in self.coeff)):
            return not self.homogeneous
        return False

class HomogeneousGeneralQuadratic(DiophantineEquationType):
    """

    Representation of a homogeneous general quadratic.

    No solver is currently implemented for this equation type.

    """
    name = 'homogeneous_general_quadratic'

    def matches(self):
        if False:
            print('Hello World!')
        if not (self.total_degree == 2 and self.dimension >= 3):
            return False
        if not self.homogeneous_order:
            return False
        elif any((k.is_Mul for k in self.coeff)):
            return self.homogeneous
        return False

class GeneralSumOfSquares(DiophantineEquationType):
    """
    Representation of the diophantine equation

    `x_{1}^2 + x_{2}^2 + . . . + x_{n}^2 - k = 0`.

    Details
    =======

    When `n = 3` if `k = 4^a(8m + 7)` for some `a, m \\in Z` then there will be
    no solutions. Refer [1]_ for more details.

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import GeneralSumOfSquares
    >>> from sympy.abc import a, b, c, d, e
    >>> GeneralSumOfSquares(a**2 + b**2 + c**2 + d**2 + e**2 - 2345).solve()
    {(15, 22, 22, 24, 24)}

    By default only 1 solution is returned. Use the `limit` keyword for more:

    >>> sorted(GeneralSumOfSquares(a**2 + b**2 + c**2 + d**2 + e**2 - 2345).solve(limit=3))
    [(15, 22, 22, 24, 24), (16, 19, 24, 24, 24), (16, 20, 22, 23, 26)]

    References
    ==========

    .. [1] Representing an integer as a sum of three squares, [online],
        Available:
        https://www.proofwiki.org/wiki/Integer_as_Sum_of_Three_Squares
    """
    name = 'general_sum_of_squares'

    def matches(self):
        if False:
            for i in range(10):
                print('nop')
        if not (self.total_degree == 2 and self.dimension >= 3):
            return False
        if not self.homogeneous_order:
            return False
        if any((k.is_Mul for k in self.coeff)):
            return False
        return all((self.coeff[k] == 1 for k in self.coeff if k != 1))

    def solve(self, parameters=None, limit=1):
        if False:
            for i in range(10):
                print('nop')
        self.pre_solve(parameters)
        var = self.free_symbols
        k = -int(self.coeff[1])
        n = self.dimension
        result = DiophantineSolutionSet(var, parameters=self.parameters)
        if k < 0 or limit < 1:
            return result
        signs = [-1 if x.is_nonpositive else 1 for x in var]
        negs = signs.count(-1) != 0
        took = 0
        for t in sum_of_squares(k, n, zeros=True):
            if negs:
                result.add([signs[i] * j for (i, j) in enumerate(t)])
            else:
                result.add(t)
            took += 1
            if took == limit:
                break
        return result

class GeneralPythagorean(DiophantineEquationType):
    """
    Representation of the general pythagorean equation,
    `a_{1}^2x_{1}^2 + a_{2}^2x_{2}^2 + . . . + a_{n}^2x_{n}^2 - a_{n + 1}^2x_{n + 1}^2 = 0`.

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import GeneralPythagorean
    >>> from sympy.abc import a, b, c, d, e, x, y, z, t
    >>> GeneralPythagorean(a**2 + b**2 + c**2 - d**2).solve()
    {(t_0**2 + t_1**2 - t_2**2, 2*t_0*t_2, 2*t_1*t_2, t_0**2 + t_1**2 + t_2**2)}
    >>> GeneralPythagorean(9*a**2 - 4*b**2 + 16*c**2 + 25*d**2 + e**2).solve(parameters=[x, y, z, t])
    {(-10*t**2 + 10*x**2 + 10*y**2 + 10*z**2, 15*t**2 + 15*x**2 + 15*y**2 + 15*z**2, 15*t*x, 12*t*y, 60*t*z)}
    """
    name = 'general_pythagorean'

    def matches(self):
        if False:
            i = 10
            return i + 15
        if not (self.total_degree == 2 and self.dimension >= 3):
            return False
        if not self.homogeneous_order:
            return False
        if any((k.is_Mul for k in self.coeff)):
            return False
        if all((self.coeff[k] == 1 for k in self.coeff if k != 1)):
            return False
        if not all((is_square(abs(self.coeff[k])) for k in self.coeff)):
            return False
        return abs(sum((sign(self.coeff[k]) for k in self.coeff))) == self.dimension - 2

    @property
    def n_parameters(self):
        if False:
            print('Hello World!')
        return self.dimension - 1

    def solve(self, parameters=None, limit=1):
        if False:
            return 10
        self.pre_solve(parameters)
        coeff = self.coeff
        var = self.free_symbols
        n = self.dimension
        if sign(coeff[var[0] ** 2]) + sign(coeff[var[1] ** 2]) + sign(coeff[var[2] ** 2]) < 0:
            for key in coeff.keys():
                coeff[key] = -coeff[key]
        result = DiophantineSolutionSet(var, parameters=self.parameters)
        index = 0
        for (i, v) in enumerate(var):
            if sign(coeff[v ** 2]) == -1:
                index = i
        m = result.parameters
        ith = sum((m_i ** 2 for m_i in m))
        L = [ith - 2 * m[n - 2] ** 2]
        L.extend([2 * m[i] * m[n - 2] for i in range(n - 2)])
        sol = L[:index] + [ith] + L[index:]
        lcm = 1
        for (i, v) in enumerate(var):
            if i == index or (index > 0 and i == 0) or (index == 0 and i == 1):
                lcm = ilcm(lcm, sqrt(abs(coeff[v ** 2])))
            else:
                s = sqrt(coeff[v ** 2])
                lcm = ilcm(lcm, s if _odd(s) else s // 2)
        for (i, v) in enumerate(var):
            sol[i] = lcm * sol[i] / sqrt(abs(coeff[v ** 2]))
        result.add(sol)
        return result

class CubicThue(DiophantineEquationType):
    """
    Representation of a cubic Thue diophantine equation.

    A cubic Thue diophantine equation is a polynomial of the form
    `f(x, y) = r` of degree 3, where `x` and `y` are integers
    and `r` is a rational number.

    No solver is currently implemented for this equation type.

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy.solvers.diophantine.diophantine import CubicThue
    >>> c1 = CubicThue(x**3 + y**2 + 1)
    >>> c1.matches()
    True

    """
    name = 'cubic_thue'

    def matches(self):
        if False:
            i = 10
            return i + 15
        return self.total_degree == 3 and self.dimension == 2

class GeneralSumOfEvenPowers(DiophantineEquationType):
    """
    Representation of the diophantine equation

    `x_{1}^e + x_{2}^e + . . . + x_{n}^e - k = 0`

    where `e` is an even, integer power.

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import GeneralSumOfEvenPowers
    >>> from sympy.abc import a, b
    >>> GeneralSumOfEvenPowers(a**4 + b**4 - (2**4 + 3**4)).solve()
    {(2, 3)}

    """
    name = 'general_sum_of_even_powers'

    def matches(self):
        if False:
            return 10
        if not self.total_degree > 3:
            return False
        if self.total_degree % 2 != 0:
            return False
        if not all((k.is_Pow and k.exp == self.total_degree for k in self.coeff if k != 1)):
            return False
        return all((self.coeff[k] == 1 for k in self.coeff if k != 1))

    def solve(self, parameters=None, limit=1):
        if False:
            while True:
                i = 10
        self.pre_solve(parameters)
        var = self.free_symbols
        coeff = self.coeff
        p = None
        for q in coeff.keys():
            if q.is_Pow and coeff[q]:
                p = q.exp
        k = len(var)
        n = -coeff[1]
        result = DiophantineSolutionSet(var, parameters=self.parameters)
        if n < 0 or limit < 1:
            return result
        sign = [-1 if x.is_nonpositive else 1 for x in var]
        negs = sign.count(-1) != 0
        took = 0
        for t in power_representation(n, p, k):
            if negs:
                result.add([sign[i] * j for (i, j) in enumerate(t)])
            else:
                result.add(t)
            took += 1
            if took == limit:
                break
        return result
all_diop_classes = [Linear, Univariate, BinaryQuadratic, InhomogeneousTernaryQuadratic, HomogeneousTernaryQuadraticNormal, HomogeneousTernaryQuadratic, InhomogeneousGeneralQuadratic, HomogeneousGeneralQuadratic, GeneralSumOfSquares, GeneralPythagorean, CubicThue, GeneralSumOfEvenPowers]
diop_known = {diop_class.name for diop_class in all_diop_classes}

def _sorted_tuple(*i):
    if False:
        while True:
            i = 10
    return tuple(sorted(i))

def _remove_gcd(*x):
    if False:
        for i in range(10):
            print('nop')
    try:
        g = igcd(*x)
    except ValueError:
        fx = list(filter(None, x))
        if len(fx) < 2:
            return x
        g = igcd(*[i.as_content_primitive()[0] for i in fx])
    except TypeError:
        raise TypeError('_remove_gcd(a,b,c) or _remove_gcd(*container)')
    if g == 1:
        return x
    return tuple([i // g for i in x])

def _rational_pq(a, b):
    if False:
        print('Hello World!')
    return _remove_gcd(sign(b) * a, abs(b))

def _nint_or_floor(p, q):
    if False:
        print('Hello World!')
    (w, r) = divmod(p, q)
    if abs(r) <= abs(q) // 2:
        return w
    return w + 1

def _odd(i):
    if False:
        while True:
            i = 10
    return i % 2 != 0

def _even(i):
    if False:
        while True:
            i = 10
    return i % 2 == 0

def diophantine(eq, param=symbols('t', integer=True), syms=None, permute=False):
    if False:
        while True:
            i = 10
    '\n    Simplify the solution procedure of diophantine equation ``eq`` by\n    converting it into a product of terms which should equal zero.\n\n    Explanation\n    ===========\n\n    For example, when solving, `x^2 - y^2 = 0` this is treated as\n    `(x + y)(x - y) = 0` and `x + y = 0` and `x - y = 0` are solved\n    independently and combined. Each term is solved by calling\n    ``diop_solve()``. (Although it is possible to call ``diop_solve()``\n    directly, one must be careful to pass an equation in the correct\n    form and to interpret the output correctly; ``diophantine()`` is\n    the public-facing function to use in general.)\n\n    Output of ``diophantine()`` is a set of tuples. The elements of the\n    tuple are the solutions for each variable in the equation and\n    are arranged according to the alphabetic ordering of the variables.\n    e.g. For an equation with two variables, `a` and `b`, the first\n    element of the tuple is the solution for `a` and the second for `b`.\n\n    Usage\n    =====\n\n    ``diophantine(eq, t, syms)``: Solve the diophantine\n    equation ``eq``.\n    ``t`` is the optional parameter to be used by ``diop_solve()``.\n    ``syms`` is an optional list of symbols which determines the\n    order of the elements in the returned tuple.\n\n    By default, only the base solution is returned. If ``permute`` is set to\n    True then permutations of the base solution and/or permutations of the\n    signs of the values will be returned when applicable.\n\n    Details\n    =======\n\n    ``eq`` should be an expression which is assumed to be zero.\n    ``t`` is the parameter to be used in the solution.\n\n    Examples\n    ========\n\n    >>> from sympy import diophantine\n    >>> from sympy.abc import a, b\n    >>> eq = a**4 + b**4 - (2**4 + 3**4)\n    >>> diophantine(eq)\n    {(2, 3)}\n    >>> diophantine(eq, permute=True)\n    {(-3, -2), (-3, 2), (-2, -3), (-2, 3), (2, -3), (2, 3), (3, -2), (3, 2)}\n\n    >>> from sympy.abc import x, y, z\n    >>> diophantine(x**2 - y**2)\n    {(t_0, -t_0), (t_0, t_0)}\n\n    >>> diophantine(x*(2*x + 3*y - z))\n    {(0, n1, n2), (t_0, t_1, 2*t_0 + 3*t_1)}\n    >>> diophantine(x**2 + 3*x*y + 4*x)\n    {(0, n1), (-3*t_0 - 4, t_0)}\n\n    See Also\n    ========\n\n    diop_solve\n    sympy.utilities.iterables.permute_signs\n    sympy.utilities.iterables.signed_permutations\n    '
    eq = _sympify(eq)
    if isinstance(eq, Eq):
        eq = eq.lhs - eq.rhs
    try:
        var = list(eq.expand(force=True).free_symbols)
        var.sort(key=default_sort_key)
        if syms:
            if not is_sequence(syms):
                raise TypeError('syms should be given as a sequence, e.g. a list')
            syms = [i for i in syms if i in var]
            if syms != var:
                dict_sym_index = dict(zip(syms, range(len(syms))))
                return {tuple([t[dict_sym_index[i]] for i in var]) for t in diophantine(eq, param, permute=permute)}
        (n, d) = eq.as_numer_denom()
        if n.is_number:
            return set()
        if not d.is_number:
            dsol = diophantine(d)
            good = diophantine(n) - dsol
            return {s for s in good if _mexpand(d.subs(zip(var, s)))}
        else:
            eq = n
        eq = factor_terms(eq)
        assert not eq.is_number
        eq = eq.as_independent(*var, as_Add=False)[1]
        p = Poly(eq)
        assert not any((g.is_number for g in p.gens))
        eq = p.as_expr()
        assert eq.is_polynomial()
    except (GeneratorsNeeded, AssertionError):
        raise TypeError(filldedent('\n    Equation should be a polynomial with Rational coefficients.'))
    do_permute_signs = False
    do_permute_signs_var = False
    permute_few_signs = False
    try:
        (v, c, t) = classify_diop(eq)
        if permute:
            len_var = len(v)
            permute_signs_for = [GeneralSumOfSquares.name, GeneralSumOfEvenPowers.name]
            permute_signs_check = [HomogeneousTernaryQuadratic.name, HomogeneousTernaryQuadraticNormal.name, BinaryQuadratic.name]
            if t in permute_signs_for:
                do_permute_signs_var = True
            elif t in permute_signs_check:
                if len_var == 3:
                    var_mul = list(subsets(v, 2))
                    xy_coeff = True
                    x_coeff = True
                    var1_mul_var2 = (a[0] * a[1] for a in var_mul)
                    for v1_mul_v2 in var1_mul_var2:
                        try:
                            coeff = c[v1_mul_v2]
                        except KeyError:
                            coeff = 0
                        xy_coeff = bool(xy_coeff) and bool(coeff)
                    var_mul = list(subsets(v, 1))
                    for v1 in var_mul:
                        try:
                            coeff = c[v1[0]]
                        except KeyError:
                            coeff = 0
                        x_coeff = bool(x_coeff) and bool(coeff)
                    if not any((xy_coeff, x_coeff)):
                        do_permute_signs = True
                    elif not x_coeff:
                        permute_few_signs = True
                elif len_var == 2:
                    var_mul = list(subsets(v, 2))
                    xy_coeff = True
                    x_coeff = True
                    var1_mul_var2 = (x[0] * x[1] for x in var_mul)
                    for v1_mul_v2 in var1_mul_var2:
                        try:
                            coeff = c[v1_mul_v2]
                        except KeyError:
                            coeff = 0
                        xy_coeff = bool(xy_coeff) and bool(coeff)
                    var_mul = list(subsets(v, 1))
                    for v1 in var_mul:
                        try:
                            coeff = c[v1[0]]
                        except KeyError:
                            coeff = 0
                        x_coeff = bool(x_coeff) and bool(coeff)
                    if not any((xy_coeff, x_coeff)):
                        do_permute_signs = True
                    elif not x_coeff:
                        permute_few_signs = True
        if t == 'general_sum_of_squares':
            terms = [(eq, 1)]
        else:
            raise TypeError
    except (TypeError, NotImplementedError):
        fl = factor_list(eq)
        if fl[0].is_Rational and fl[0] != 1:
            return diophantine(eq / fl[0], param=param, syms=syms, permute=permute)
        terms = fl[1]
    sols = set()
    for term in terms:
        (base, _) = term
        (var_t, _, eq_type) = classify_diop(base, _dict=False)
        (_, base) = signsimp(base, evaluate=False).as_coeff_Mul()
        solution = diop_solve(base, param)
        if eq_type in [Linear.name, HomogeneousTernaryQuadratic.name, HomogeneousTernaryQuadraticNormal.name, GeneralPythagorean.name]:
            sols.add(merge_solution(var, var_t, solution))
        elif eq_type in [BinaryQuadratic.name, GeneralSumOfSquares.name, GeneralSumOfEvenPowers.name, Univariate.name]:
            for sol in solution:
                sols.add(merge_solution(var, var_t, sol))
        else:
            raise NotImplementedError('unhandled type: %s' % eq_type)
    if () in sols:
        sols.remove(())
    null = tuple([0] * len(var))
    if not sols and eq.subs(zip(var, null)).is_zero:
        sols.add(null)
    final_soln = set()
    for sol in sols:
        if all((int_valued(s) for s in sol)):
            if do_permute_signs:
                permuted_sign = set(permute_signs(sol))
                final_soln.update(permuted_sign)
            elif permute_few_signs:
                lst = list(permute_signs(sol))
                lst = list(filter(lambda x: x[0] * x[1] == sol[1] * sol[0], lst))
                permuted_sign = set(lst)
                final_soln.update(permuted_sign)
            elif do_permute_signs_var:
                permuted_sign_var = set(signed_permutations(sol))
                final_soln.update(permuted_sign_var)
            else:
                final_soln.add(sol)
        else:
            final_soln.add(sol)
    return final_soln

def merge_solution(var, var_t, solution):
    if False:
        for i in range(10):
            print('nop')
    '\n    This is used to construct the full solution from the solutions of sub\n    equations.\n\n    Explanation\n    ===========\n\n    For example when solving the equation `(x - y)(x^2 + y^2 - z^2) = 0`,\n    solutions for each of the equations `x - y = 0` and `x^2 + y^2 - z^2` are\n    found independently. Solutions for `x - y = 0` are `(x, y) = (t, t)`. But\n    we should introduce a value for z when we output the solution for the\n    original equation. This function converts `(t, t)` into `(t, t, n_{1})`\n    where `n_{1}` is an integer parameter.\n    '
    sol = []
    if None in solution:
        return ()
    solution = iter(solution)
    params = numbered_symbols('n', integer=True, start=1)
    for v in var:
        if v in var_t:
            sol.append(next(solution))
        else:
            sol.append(next(params))
    for (val, symb) in zip(sol, var):
        if check_assumptions(val, **symb.assumptions0) is False:
            return ()
    return tuple(sol)

def _diop_solve(eq, params=None):
    if False:
        while True:
            i = 10
    for diop_type in all_diop_classes:
        if diop_type(eq).matches():
            return diop_type(eq).solve(parameters=params)

def diop_solve(eq, param=symbols('t', integer=True)):
    if False:
        return 10
    '\n    Solves the diophantine equation ``eq``.\n\n    Explanation\n    ===========\n\n    Unlike ``diophantine()``, factoring of ``eq`` is not attempted. Uses\n    ``classify_diop()`` to determine the type of the equation and calls\n    the appropriate solver function.\n\n    Use of ``diophantine()`` is recommended over other helper functions.\n    ``diop_solve()`` can return either a set or a tuple depending on the\n    nature of the equation.\n\n    Usage\n    =====\n\n    ``diop_solve(eq, t)``: Solve diophantine equation, ``eq`` using ``t``\n    as a parameter if needed.\n\n    Details\n    =======\n\n    ``eq`` should be an expression which is assumed to be zero.\n    ``t`` is a parameter to be used in the solution.\n\n    Examples\n    ========\n\n    >>> from sympy.solvers.diophantine import diop_solve\n    >>> from sympy.abc import x, y, z, w\n    >>> diop_solve(2*x + 3*y - 5)\n    (3*t_0 - 5, 5 - 2*t_0)\n    >>> diop_solve(4*x + 3*y - 4*z + 5)\n    (t_0, 8*t_0 + 4*t_1 + 5, 7*t_0 + 3*t_1 + 5)\n    >>> diop_solve(x + 3*y - 4*z + w - 6)\n    (t_0, t_0 + t_1, 6*t_0 + 5*t_1 + 4*t_2 - 6, 5*t_0 + 4*t_1 + 3*t_2 - 6)\n    >>> diop_solve(x**2 + y**2 - 5)\n    {(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)}\n\n\n    See Also\n    ========\n\n    diophantine()\n    '
    (var, coeff, eq_type) = classify_diop(eq, _dict=False)
    if eq_type == Linear.name:
        return diop_linear(eq, param)
    elif eq_type == BinaryQuadratic.name:
        return diop_quadratic(eq, param)
    elif eq_type == HomogeneousTernaryQuadratic.name:
        return diop_ternary_quadratic(eq, parameterize=True)
    elif eq_type == HomogeneousTernaryQuadraticNormal.name:
        return diop_ternary_quadratic_normal(eq, parameterize=True)
    elif eq_type == GeneralPythagorean.name:
        return diop_general_pythagorean(eq, param)
    elif eq_type == Univariate.name:
        return diop_univariate(eq)
    elif eq_type == GeneralSumOfSquares.name:
        return diop_general_sum_of_squares(eq, limit=S.Infinity)
    elif eq_type == GeneralSumOfEvenPowers.name:
        return diop_general_sum_of_even_powers(eq, limit=S.Infinity)
    if eq_type is not None and eq_type not in diop_known:
        raise ValueError(filldedent('\n    Although this type of equation was identified, it is not yet\n    handled. It should, however, be listed in `diop_known` at the\n    top of this file. Developers should see comments at the end of\n    `classify_diop`.\n            '))
    else:
        raise NotImplementedError('No solver has been written for %s.' % eq_type)

def classify_diop(eq, _dict=True):
    if False:
        return 10
    matched = False
    diop_type = None
    for diop_class in all_diop_classes:
        diop_type = diop_class(eq)
        if diop_type.matches():
            matched = True
            break
    if matched:
        return (diop_type.free_symbols, dict(diop_type.coeff) if _dict else diop_type.coeff, diop_type.name)
    raise NotImplementedError(filldedent('\n        This equation is not yet recognized or else has not been\n        simplified sufficiently to put it in a form recognized by\n        diop_classify().'))
classify_diop.func_doc = "\n    Helper routine used by diop_solve() to find information about ``eq``.\n\n    Explanation\n    ===========\n\n    Returns a tuple containing the type of the diophantine equation\n    along with the variables (free symbols) and their coefficients.\n    Variables are returned as a list and coefficients are returned\n    as a dict with the key being the respective term and the constant\n    term is keyed to 1. The type is one of the following:\n\n    * %s\n\n    Usage\n    =====\n\n    ``classify_diop(eq)``: Return variables, coefficients and type of the\n    ``eq``.\n\n    Details\n    =======\n\n    ``eq`` should be an expression which is assumed to be zero.\n    ``_dict`` is for internal use: when True (default) a dict is returned,\n    otherwise a defaultdict which supplies 0 for missing keys is returned.\n\n    Examples\n    ========\n\n    >>> from sympy.solvers.diophantine import classify_diop\n    >>> from sympy.abc import x, y, z, w, t\n    >>> classify_diop(4*x + 6*y - 4)\n    ([x, y], {1: -4, x: 4, y: 6}, 'linear')\n    >>> classify_diop(x + 3*y -4*z + 5)\n    ([x, y, z], {1: 5, x: 1, y: 3, z: -4}, 'linear')\n    >>> classify_diop(x**2 + y**2 - x*y + x + 5)\n    ([x, y], {1: 5, x: 1, x**2: 1, y**2: 1, x*y: -1}, 'binary_quadratic')\n    " % '\n    * '.join(sorted(diop_known))

def diop_linear(eq, param=symbols('t', integer=True)):
    if False:
        print('Hello World!')
    '\n    Solves linear diophantine equations.\n\n    A linear diophantine equation is an equation of the form `a_{1}x_{1} +\n    a_{2}x_{2} + .. + a_{n}x_{n} = 0` where `a_{1}, a_{2}, ..a_{n}` are\n    integer constants and `x_{1}, x_{2}, ..x_{n}` are integer variables.\n\n    Usage\n    =====\n\n    ``diop_linear(eq)``: Returns a tuple containing solutions to the\n    diophantine equation ``eq``. Values in the tuple is arranged in the same\n    order as the sorted variables.\n\n    Details\n    =======\n\n    ``eq`` is a linear diophantine equation which is assumed to be zero.\n    ``param`` is the parameter to be used in the solution.\n\n    Examples\n    ========\n\n    >>> from sympy.solvers.diophantine.diophantine import diop_linear\n    >>> from sympy.abc import x, y, z\n    >>> diop_linear(2*x - 3*y - 5) # solves equation 2*x - 3*y - 5 == 0\n    (3*t_0 - 5, 2*t_0 - 5)\n\n    Here x = -3*t_0 - 5 and y = -2*t_0 - 5\n\n    >>> diop_linear(2*x - 3*y - 4*z -3)\n    (t_0, 2*t_0 + 4*t_1 + 3, -t_0 - 3*t_1 - 3)\n\n    See Also\n    ========\n\n    diop_quadratic(), diop_ternary_quadratic(), diop_general_pythagorean(),\n    diop_general_sum_of_squares()\n    '
    (var, coeff, diop_type) = classify_diop(eq, _dict=False)
    if diop_type == Linear.name:
        parameters = None
        if param is not None:
            parameters = symbols('%s_0:%i' % (param, len(var)), integer=True)
        result = Linear(eq).solve(parameters=parameters)
        if param is None:
            result = result(*[0] * len(result.parameters))
        if len(result) > 0:
            return list(result)[0]
        else:
            return tuple([None] * len(result.parameters))

def base_solution_linear(c, a, b, t=None):
    if False:
        return 10
    '\n    Return the base solution for the linear equation, `ax + by = c`.\n\n    Explanation\n    ===========\n\n    Used by ``diop_linear()`` to find the base solution of a linear\n    Diophantine equation. If ``t`` is given then the parametrized solution is\n    returned.\n\n    Usage\n    =====\n\n    ``base_solution_linear(c, a, b, t)``: ``a``, ``b``, ``c`` are coefficients\n    in `ax + by = c` and ``t`` is the parameter to be used in the solution.\n\n    Examples\n    ========\n\n    >>> from sympy.solvers.diophantine.diophantine import base_solution_linear\n    >>> from sympy.abc import t\n    >>> base_solution_linear(5, 2, 3) # equation 2*x + 3*y = 5\n    (-5, 5)\n    >>> base_solution_linear(0, 5, 7) # equation 5*x + 7*y = 0\n    (0, 0)\n    >>> base_solution_linear(5, 2, 3, t) # equation 2*x + 3*y = 5\n    (3*t - 5, 5 - 2*t)\n    >>> base_solution_linear(0, 5, 7, t) # equation 5*x + 7*y = 0\n    (7*t, -5*t)\n    '
    (a, b, c) = _remove_gcd(a, b, c)
    if c == 0:
        if t is not None:
            if b < 0:
                t = -t
            return (b * t, -a * t)
        else:
            return (0, 0)
    else:
        (x0, y0, d) = igcdex(abs(a), abs(b))
        x0 *= sign(a)
        y0 *= sign(b)
        if divisible(c, d):
            if t is not None:
                if b < 0:
                    t = -t
                return (c * x0 + b * t, c * y0 - a * t)
            else:
                return (c * x0, c * y0)
        else:
            return (None, None)

def diop_univariate(eq):
    if False:
        return 10
    '\n    Solves a univariate diophantine equations.\n\n    Explanation\n    ===========\n\n    A univariate diophantine equation is an equation of the form\n    `a_{0} + a_{1}x + a_{2}x^2 + .. + a_{n}x^n = 0` where `a_{1}, a_{2}, ..a_{n}` are\n    integer constants and `x` is an integer variable.\n\n    Usage\n    =====\n\n    ``diop_univariate(eq)``: Returns a set containing solutions to the\n    diophantine equation ``eq``.\n\n    Details\n    =======\n\n    ``eq`` is a univariate diophantine equation which is assumed to be zero.\n\n    Examples\n    ========\n\n    >>> from sympy.solvers.diophantine.diophantine import diop_univariate\n    >>> from sympy.abc import x\n    >>> diop_univariate((x - 2)*(x - 3)**2) # solves equation (x - 2)*(x - 3)**2 == 0\n    {(2,), (3,)}\n\n    '
    (var, coeff, diop_type) = classify_diop(eq, _dict=False)
    if diop_type == Univariate.name:
        return {(int(i),) for i in solveset_real(eq, var[0]).intersect(S.Integers)}

def divisible(a, b):
    if False:
        while True:
            i = 10
    '\n    Returns `True` if ``a`` is divisible by ``b`` and `False` otherwise.\n    '
    return not a % b

def diop_quadratic(eq, param=symbols('t', integer=True)):
    if False:
        print('Hello World!')
    '\n    Solves quadratic diophantine equations.\n\n    i.e. equations of the form `Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0`. Returns a\n    set containing the tuples `(x, y)` which contains the solutions. If there\n    are no solutions then `(None, None)` is returned.\n\n    Usage\n    =====\n\n    ``diop_quadratic(eq, param)``: ``eq`` is a quadratic binary diophantine\n    equation. ``param`` is used to indicate the parameter to be used in the\n    solution.\n\n    Details\n    =======\n\n    ``eq`` should be an expression which is assumed to be zero.\n    ``param`` is a parameter to be used in the solution.\n\n    Examples\n    ========\n\n    >>> from sympy.abc import x, y, t\n    >>> from sympy.solvers.diophantine.diophantine import diop_quadratic\n    >>> diop_quadratic(x**2 + y**2 + 2*x + 2*y + 2, t)\n    {(-1, -1)}\n\n    References\n    ==========\n\n    .. [1] Methods to solve Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0, [online],\n          Available: https://www.alpertron.com.ar/METHODS.HTM\n    .. [2] Solving the equation ax^2+ bxy + cy^2 + dx + ey + f= 0, [online],\n          Available: https://web.archive.org/web/20160323033111/http://www.jpr2718.org/ax2p.pdf\n\n    See Also\n    ========\n\n    diop_linear(), diop_ternary_quadratic(), diop_general_sum_of_squares(),\n    diop_general_pythagorean()\n    '
    (var, coeff, diop_type) = classify_diop(eq, _dict=False)
    if diop_type == BinaryQuadratic.name:
        if param is not None:
            parameters = [param, Symbol('u', integer=True)]
        else:
            parameters = None
        return set(BinaryQuadratic(eq).solve(parameters=parameters))

def is_solution_quad(var, coeff, u, v):
    if False:
        i = 10
        return i + 15
    '\n    Check whether `(u, v)` is solution to the quadratic binary diophantine\n    equation with the variable list ``var`` and coefficient dictionary\n    ``coeff``.\n\n    Not intended for use by normal users.\n    '
    reps = dict(zip(var, (u, v)))
    eq = Add(*[j * i.xreplace(reps) for (i, j) in coeff.items()])
    return _mexpand(eq) == 0

def diop_DN(D, N, t=symbols('t', integer=True)):
    if False:
        return 10
    '\n    Solves the equation `x^2 - Dy^2 = N`.\n\n    Explanation\n    ===========\n\n    Mainly concerned with the case `D > 0, D` is not a perfect square,\n    which is the same as the generalized Pell equation. The LMM\n    algorithm [1]_ is used to solve this equation.\n\n    Returns one solution tuple, (`x, y)` for each class of the solutions.\n    Other solutions of the class can be constructed according to the\n    values of ``D`` and ``N``.\n\n    Usage\n    =====\n\n    ``diop_DN(D, N, t)``: D and N are integers as in `x^2 - Dy^2 = N` and\n    ``t`` is the parameter to be used in the solutions.\n\n    Details\n    =======\n\n    ``D`` and ``N`` correspond to D and N in the equation.\n    ``t`` is the parameter to be used in the solutions.\n\n    Examples\n    ========\n\n    >>> from sympy.solvers.diophantine.diophantine import diop_DN\n    >>> diop_DN(13, -4) # Solves equation x**2 - 13*y**2 = -4\n    [(3, 1), (393, 109), (36, 10)]\n\n    The output can be interpreted as follows: There are three fundamental\n    solutions to the equation `x^2 - 13y^2 = -4` given by (3, 1), (393, 109)\n    and (36, 10). Each tuple is in the form (x, y), i.e. solution (3, 1) means\n    that `x = 3` and `y = 1`.\n\n    >>> diop_DN(986, 1) # Solves equation x**2 - 986*y**2 = 1\n    [(49299, 1570)]\n\n    See Also\n    ========\n\n    find_DN(), diop_bf_DN()\n\n    References\n    ==========\n\n    .. [1] Solving the generalized Pell equation x**2 - D*y**2 = N, John P.\n        Robertson, July 31, 2004, Pages 16 - 17. [online], Available:\n        https://web.archive.org/web/20160323033128/http://www.jpr2718.org/pell.pdf\n    '
    if D < 0:
        if N == 0:
            return [(0, 0)]
        elif N < 0:
            return []
        elif N > 0:
            sol = []
            for d in divisors(square_factor(N)):
                sols = cornacchia(1, -D, N // d ** 2)
                if sols:
                    for (x, y) in sols:
                        sol.append((d * x, d * y))
                        if D == -1:
                            sol.append((d * y, d * x))
            return sol
    elif D == 0:
        if N < 0:
            return []
        if N == 0:
            return [(0, t)]
        (sN, _exact) = integer_nthroot(N, 2)
        if _exact:
            return [(sN, t)]
        else:
            return []
    else:
        (sD, _exact) = integer_nthroot(D, 2)
        if _exact:
            if N == 0:
                return [(sD * t, t)]
            else:
                sol = []
                for y in range(floor(sign(N) * (N - 1) / (2 * sD)) + 1):
                    try:
                        (sq, _exact) = integer_nthroot(D * y ** 2 + N, 2)
                    except ValueError:
                        _exact = False
                    if _exact:
                        sol.append((sq, y))
                return sol
        elif 1 < N ** 2 < D:
            return _special_diop_DN(D, N)
        elif N == 0:
            return [(0, 0)]
        elif abs(N) == 1:
            pqa = PQa(0, 1, D)
            j = 0
            G = []
            B = []
            for i in pqa:
                a = i[2]
                G.append(i[5])
                B.append(i[4])
                if j != 0 and a == 2 * sD:
                    break
                j = j + 1
            if _odd(j):
                if N == -1:
                    x = G[j - 1]
                    y = B[j - 1]
                else:
                    count = j
                    while count < 2 * j - 1:
                        i = next(pqa)
                        G.append(i[5])
                        B.append(i[4])
                        count += 1
                    x = G[count]
                    y = B[count]
            elif N == 1:
                x = G[j - 1]
                y = B[j - 1]
            else:
                return []
            return [(x, y)]
        else:
            fs = []
            sol = []
            div = divisors(N)
            for d in div:
                if divisible(N, d ** 2):
                    fs.append(d)
            for f in fs:
                m = N // f ** 2
                zs = sqrt_mod(D, abs(m), all_roots=True)
                zs = [i for i in zs if i <= abs(m) // 2]
                if abs(m) != 2:
                    zs = zs + [-i for i in zs if i]
                for z in zs:
                    pqa = PQa(z, abs(m), D)
                    j = 0
                    G = []
                    B = []
                    for i in pqa:
                        G.append(i[5])
                        B.append(i[4])
                        if j != 0 and abs(i[1]) == 1:
                            r = G[j - 1]
                            s = B[j - 1]
                            if r ** 2 - D * s ** 2 == m:
                                sol.append((f * r, f * s))
                            elif diop_DN(D, -1) != []:
                                a = diop_DN(D, -1)
                                sol.append((f * (r * a[0][0] + a[0][1] * s * D), f * (r * a[0][1] + s * a[0][0])))
                            break
                        j = j + 1
                        if j == length(z, abs(m), D):
                            break
            return sol

def _special_diop_DN(D, N):
    if False:
        return 10
    '\n    Solves the equation `x^2 - Dy^2 = N` for the special case where\n    `1 < N**2 < D` and `D` is not a perfect square.\n    It is better to call `diop_DN` rather than this function, as\n    the former checks the condition `1 < N**2 < D`, and calls the latter only\n    if appropriate.\n\n    Usage\n    =====\n\n    WARNING: Internal method. Do not call directly!\n\n    ``_special_diop_DN(D, N)``: D and N are integers as in `x^2 - Dy^2 = N`.\n\n    Details\n    =======\n\n    ``D`` and ``N`` correspond to D and N in the equation.\n\n    Examples\n    ========\n\n    >>> from sympy.solvers.diophantine.diophantine import _special_diop_DN\n    >>> _special_diop_DN(13, -3) # Solves equation x**2 - 13*y**2 = -3\n    [(7, 2), (137, 38)]\n\n    The output can be interpreted as follows: There are two fundamental\n    solutions to the equation `x^2 - 13y^2 = -3` given by (7, 2) and\n    (137, 38). Each tuple is in the form (x, y), i.e. solution (7, 2) means\n    that `x = 7` and `y = 2`.\n\n    >>> _special_diop_DN(2445, -20) # Solves equation x**2 - 2445*y**2 = -20\n    [(445, 9), (17625560, 356454), (698095554475, 14118073569)]\n\n    See Also\n    ========\n\n    diop_DN()\n\n    References\n    ==========\n\n    .. [1] Section 4.4.4 of the following book:\n        Quadratic Diophantine Equations, T. Andreescu and D. Andrica,\n        Springer, 2015.\n    '
    sqrt_D = sqrt(D)
    F = [(N, 1)]
    f = 2
    while True:
        f2 = f ** 2
        if f2 > abs(N):
            break
        (n, r) = divmod(N, f2)
        if r == 0:
            F.append((n, f))
        f += 1
    P = 0
    Q = 1
    (G0, G1) = (0, 1)
    (B0, B1) = (1, 0)
    solutions = []
    i = 0
    while True:
        a = floor((P + sqrt_D) / Q)
        P = a * Q - P
        Q = (D - P ** 2) // Q
        G2 = a * G1 + G0
        B2 = a * B1 + B0
        for (n, f) in F:
            if G2 ** 2 - D * B2 ** 2 == n:
                solutions.append((f * G2, f * B2))
        i += 1
        if Q == 1 and i % 2 == 0:
            break
        (G0, G1) = (G1, G2)
        (B0, B1) = (B1, B2)
    return solutions

def cornacchia(a, b, m):
    if False:
        for i in range(10):
            print('nop')
    '\n    Solves `ax^2 + by^2 = m` where `\\gcd(a, b) = 1 = gcd(a, m)` and `a, b > 0`.\n\n    Explanation\n    ===========\n\n    Uses the algorithm due to Cornacchia. The method only finds primitive\n    solutions, i.e. ones with `\\gcd(x, y) = 1`. So this method cannot be used to\n    find the solutions of `x^2 + y^2 = 20` since the only solution to former is\n    `(x, y) = (4, 2)` and it is not primitive. When `a = b`, only the\n    solutions with `x \\leq y` are found. For more details, see the References.\n\n    Examples\n    ========\n\n    >>> from sympy.solvers.diophantine.diophantine import cornacchia\n    >>> cornacchia(2, 3, 35) # equation 2x**2 + 3y**2 = 35\n    {(2, 3), (4, 1)}\n    >>> cornacchia(1, 1, 25) # equation x**2 + y**2 = 25\n    {(4, 3)}\n\n    References\n    ===========\n\n    .. [1] A. Nitaj, "L\'algorithme de Cornacchia"\n    .. [2] Solving the diophantine equation ax**2 + by**2 = m by Cornacchia\'s\n        method, [online], Available:\n        http://www.numbertheory.org/php/cornacchia.html\n\n    See Also\n    ========\n\n    sympy.utilities.iterables.signed_permutations\n    '
    sols = set()
    a1 = igcdex(a, m)[0]
    v = sqrt_mod(-b * a1, m, all_roots=True)
    if not v:
        return None
    for t in v:
        if t < m // 2:
            continue
        (u, r) = (t, m)
        while True:
            (u, r) = (r, u % r)
            if a * r ** 2 < m:
                break
        m1 = m - a * r ** 2
        if m1 % b == 0:
            m1 = m1 // b
            (s, _exact) = integer_nthroot(m1, 2)
            if _exact:
                if a == b and r < s:
                    (r, s) = (s, r)
                sols.add((int(r), int(s)))
    return sols

def PQa(P_0, Q_0, D):
    if False:
        print('Hello World!')
    "\n    Returns useful information needed to solve the Pell equation.\n\n    Explanation\n    ===========\n\n    There are six sequences of integers defined related to the continued\n    fraction representation of `\\\\frac{P + \\sqrt{D}}{Q}`, namely {`P_{i}`},\n    {`Q_{i}`}, {`a_{i}`},{`A_{i}`}, {`B_{i}`}, {`G_{i}`}. ``PQa()`` Returns\n    these values as a 6-tuple in the same order as mentioned above. Refer [1]_\n    for more detailed information.\n\n    Usage\n    =====\n\n    ``PQa(P_0, Q_0, D)``: ``P_0``, ``Q_0`` and ``D`` are integers corresponding\n    to `P_{0}`, `Q_{0}` and `D` in the continued fraction\n    `\\\\frac{P_{0} + \\sqrt{D}}{Q_{0}}`.\n    Also it's assumed that `P_{0}^2 == D mod(|Q_{0}|)` and `D` is square free.\n\n    Examples\n    ========\n\n    >>> from sympy.solvers.diophantine.diophantine import PQa\n    >>> pqa = PQa(13, 4, 5) # (13 + sqrt(5))/4\n    >>> next(pqa) # (P_0, Q_0, a_0, A_0, B_0, G_0)\n    (13, 4, 3, 3, 1, -1)\n    >>> next(pqa) # (P_1, Q_1, a_1, A_1, B_1, G_1)\n    (-1, 1, 1, 4, 1, 3)\n\n    References\n    ==========\n\n    .. [1] Solving the generalized Pell equation x^2 - Dy^2 = N, John P.\n        Robertson, July 31, 2004, Pages 4 - 8. https://web.archive.org/web/20160323033128/http://www.jpr2718.org/pell.pdf\n    "
    A_i_2 = B_i_1 = 0
    A_i_1 = B_i_2 = 1
    G_i_2 = -P_0
    G_i_1 = Q_0
    P_i = P_0
    Q_i = Q_0
    while True:
        a_i = floor((P_i + sqrt(D)) / Q_i)
        A_i = a_i * A_i_1 + A_i_2
        B_i = a_i * B_i_1 + B_i_2
        G_i = a_i * G_i_1 + G_i_2
        yield (P_i, Q_i, a_i, A_i, B_i, G_i)
        (A_i_1, A_i_2) = (A_i, A_i_1)
        (B_i_1, B_i_2) = (B_i, B_i_1)
        (G_i_1, G_i_2) = (G_i, G_i_1)
        P_i = a_i * Q_i - P_i
        Q_i = (D - P_i ** 2) / Q_i

def diop_bf_DN(D, N, t=symbols('t', integer=True)):
    if False:
        for i in range(10):
            print('nop')
    '\n    Uses brute force to solve the equation, `x^2 - Dy^2 = N`.\n\n    Explanation\n    ===========\n\n    Mainly concerned with the generalized Pell equation which is the case when\n    `D > 0, D` is not a perfect square. For more information on the case refer\n    [1]_. Let `(t, u)` be the minimal positive solution of the equation\n    `x^2 - Dy^2 = 1`. Then this method requires\n    `\\sqrt{\\\\frac{\\mid N \\mid (t \\pm 1)}{2D}}` to be small.\n\n    Usage\n    =====\n\n    ``diop_bf_DN(D, N, t)``: ``D`` and ``N`` are coefficients in\n    `x^2 - Dy^2 = N` and ``t`` is the parameter to be used in the solutions.\n\n    Details\n    =======\n\n    ``D`` and ``N`` correspond to D and N in the equation.\n    ``t`` is the parameter to be used in the solutions.\n\n    Examples\n    ========\n\n    >>> from sympy.solvers.diophantine.diophantine import diop_bf_DN\n    >>> diop_bf_DN(13, -4)\n    [(3, 1), (-3, 1), (36, 10)]\n    >>> diop_bf_DN(986, 1)\n    [(49299, 1570)]\n\n    See Also\n    ========\n\n    diop_DN()\n\n    References\n    ==========\n\n    .. [1] Solving the generalized Pell equation x**2 - D*y**2 = N, John P.\n        Robertson, July 31, 2004, Page 15. https://web.archive.org/web/20160323033128/http://www.jpr2718.org/pell.pdf\n    '
    D = as_int(D)
    N = as_int(N)
    sol = []
    a = diop_DN(D, 1)
    u = a[0][0]
    if abs(N) == 1:
        return diop_DN(D, N)
    elif N > 1:
        L1 = 0
        L2 = integer_nthroot(int(N * (u - 1) / (2 * D)), 2)[0] + 1
    elif N < -1:
        (L1, _exact) = integer_nthroot(-int(N / D), 2)
        if not _exact:
            L1 += 1
        L2 = integer_nthroot(-int(N * (u + 1) / (2 * D)), 2)[0] + 1
    elif D < 0:
        return [(0, 0)]
    elif D == 0:
        return [(0, t)]
    else:
        (sD, _exact) = integer_nthroot(D, 2)
        if _exact:
            return [(sD * t, t), (-sD * t, t)]
        else:
            return [(0, 0)]
    for y in range(L1, L2):
        try:
            (x, _exact) = integer_nthroot(N + D * y ** 2, 2)
        except ValueError:
            _exact = False
        if _exact:
            sol.append((x, y))
            if not equivalent(x, y, -x, y, D, N):
                sol.append((-x, y))
    return sol

def equivalent(u, v, r, s, D, N):
    if False:
        return 10
    '\n    Returns True if two solutions `(u, v)` and `(r, s)` of `x^2 - Dy^2 = N`\n    belongs to the same equivalence class and False otherwise.\n\n    Explanation\n    ===========\n\n    Two solutions `(u, v)` and `(r, s)` to the above equation fall to the same\n    equivalence class iff both `(ur - Dvs)` and `(us - vr)` are divisible by\n    `N`. See reference [1]_. No test is performed to test whether `(u, v)` and\n    `(r, s)` are actually solutions to the equation. User should take care of\n    this.\n\n    Usage\n    =====\n\n    ``equivalent(u, v, r, s, D, N)``: `(u, v)` and `(r, s)` are two solutions\n    of the equation `x^2 - Dy^2 = N` and all parameters involved are integers.\n\n    Examples\n    ========\n\n    >>> from sympy.solvers.diophantine.diophantine import equivalent\n    >>> equivalent(18, 5, -18, -5, 13, -1)\n    True\n    >>> equivalent(3, 1, -18, 393, 109, -4)\n    False\n\n    References\n    ==========\n\n    .. [1] Solving the generalized Pell equation x**2 - D*y**2 = N, John P.\n        Robertson, July 31, 2004, Page 12. https://web.archive.org/web/20160323033128/http://www.jpr2718.org/pell.pdf\n\n    '
    return divisible(u * r - D * v * s, N) and divisible(u * s - v * r, N)

def length(P, Q, D):
    if False:
        return 10
    '\n    Returns the (length of aperiodic part + length of periodic part) of\n    continued fraction representation of `\\\\frac{P + \\sqrt{D}}{Q}`.\n\n    It is important to remember that this does NOT return the length of the\n    periodic part but the sum of the lengths of the two parts as mentioned\n    above.\n\n    Usage\n    =====\n\n    ``length(P, Q, D)``: ``P``, ``Q`` and ``D`` are integers corresponding to\n    the continued fraction `\\\\frac{P + \\sqrt{D}}{Q}`.\n\n    Details\n    =======\n\n    ``P``, ``D`` and ``Q`` corresponds to P, D and Q in the continued fraction,\n    `\\\\frac{P + \\sqrt{D}}{Q}`.\n\n    Examples\n    ========\n\n    >>> from sympy.solvers.diophantine.diophantine import length\n    >>> length(-2, 4, 5) # (-2 + sqrt(5))/4\n    3\n    >>> length(-5, 4, 17) # (-5 + sqrt(17))/4\n    4\n\n    See Also\n    ========\n    sympy.ntheory.continued_fraction.continued_fraction_periodic\n    '
    from sympy.ntheory.continued_fraction import continued_fraction_periodic
    v = continued_fraction_periodic(P, Q, D)
    if isinstance(v[-1], list):
        rpt = len(v[-1])
        nonrpt = len(v) - 1
    else:
        rpt = 0
        nonrpt = len(v)
    return rpt + nonrpt

def transformation_to_DN(eq):
    if False:
        while True:
            i = 10
    '\n    This function transforms general quadratic,\n    `ax^2 + bxy + cy^2 + dx + ey + f = 0`\n    to more easy to deal with `X^2 - DY^2 = N` form.\n\n    Explanation\n    ===========\n\n    This is used to solve the general quadratic equation by transforming it to\n    the latter form. Refer to [1]_ for more detailed information on the\n    transformation. This function returns a tuple (A, B) where A is a 2 X 2\n    matrix and B is a 2 X 1 matrix such that,\n\n    Transpose([x y]) =  A * Transpose([X Y]) + B\n\n    Usage\n    =====\n\n    ``transformation_to_DN(eq)``: where ``eq`` is the quadratic to be\n    transformed.\n\n    Examples\n    ========\n\n    >>> from sympy.abc import x, y\n    >>> from sympy.solvers.diophantine.diophantine import transformation_to_DN\n    >>> A, B = transformation_to_DN(x**2 - 3*x*y - y**2 - 2*y + 1)\n    >>> A\n    Matrix([\n    [1/26, 3/26],\n    [   0, 1/13]])\n    >>> B\n    Matrix([\n    [-6/13],\n    [-4/13]])\n\n    A, B  returned are such that Transpose((x y)) =  A * Transpose((X Y)) + B.\n    Substituting these values for `x` and `y` and a bit of simplifying work\n    will give an equation of the form `x^2 - Dy^2 = N`.\n\n    >>> from sympy.abc import X, Y\n    >>> from sympy import Matrix, simplify\n    >>> u = (A*Matrix([X, Y]) + B)[0] # Transformation for x\n    >>> u\n    X/26 + 3*Y/26 - 6/13\n    >>> v = (A*Matrix([X, Y]) + B)[1] # Transformation for y\n    >>> v\n    Y/13 - 4/13\n\n    Next we will substitute these formulas for `x` and `y` and do\n    ``simplify()``.\n\n    >>> eq = simplify((x**2 - 3*x*y - y**2 - 2*y + 1).subs(zip((x, y), (u, v))))\n    >>> eq\n    X**2/676 - Y**2/52 + 17/13\n\n    By multiplying the denominator appropriately, we can get a Pell equation\n    in the standard form.\n\n    >>> eq * 676\n    X**2 - 13*Y**2 + 884\n\n    If only the final equation is needed, ``find_DN()`` can be used.\n\n    See Also\n    ========\n\n    find_DN()\n\n    References\n    ==========\n\n    .. [1] Solving the equation ax^2 + bxy + cy^2 + dx + ey + f = 0,\n           John P.Robertson, May 8, 2003, Page 7 - 11.\n           https://web.archive.org/web/20160323033111/http://www.jpr2718.org/ax2p.pdf\n    '
    (var, coeff, diop_type) = classify_diop(eq, _dict=False)
    if diop_type == BinaryQuadratic.name:
        return _transformation_to_DN(var, coeff)

def _transformation_to_DN(var, coeff):
    if False:
        while True:
            i = 10
    (x, y) = var
    a = coeff[x ** 2]
    b = coeff[x * y]
    c = coeff[y ** 2]
    d = coeff[x]
    e = coeff[y]
    f = coeff[1]
    (a, b, c, d, e, f) = [as_int(i) for i in _remove_gcd(a, b, c, d, e, f)]
    (X, Y) = symbols('X, Y', integer=True)
    if b:
        (B, C) = _rational_pq(2 * a, b)
        (A, T) = _rational_pq(a, B ** 2)
        coeff = {X ** 2: A * B, X * Y: 0, Y ** 2: B * (c * T - A * C ** 2), X: d * T, Y: B * e * T - d * T * C, 1: f * T * B}
        (A_0, B_0) = _transformation_to_DN([X, Y], coeff)
        return (Matrix(2, 2, [S.One / B, -S(C) / B, 0, 1]) * A_0, Matrix(2, 2, [S.One / B, -S(C) / B, 0, 1]) * B_0)
    elif d:
        (B, C) = _rational_pq(2 * a, d)
        (A, T) = _rational_pq(a, B ** 2)
        coeff = {X ** 2: A, X * Y: 0, Y ** 2: c * T, X: 0, Y: e * T, 1: f * T - A * C ** 2}
        (A_0, B_0) = _transformation_to_DN([X, Y], coeff)
        return (Matrix(2, 2, [S.One / B, 0, 0, 1]) * A_0, Matrix(2, 2, [S.One / B, 0, 0, 1]) * B_0 + Matrix([-S(C) / B, 0]))
    elif e:
        (B, C) = _rational_pq(2 * c, e)
        (A, T) = _rational_pq(c, B ** 2)
        coeff = {X ** 2: a * T, X * Y: 0, Y ** 2: A, X: 0, Y: 0, 1: f * T - A * C ** 2}
        (A_0, B_0) = _transformation_to_DN([X, Y], coeff)
        return (Matrix(2, 2, [1, 0, 0, S.One / B]) * A_0, Matrix(2, 2, [1, 0, 0, S.One / B]) * B_0 + Matrix([0, -S(C) / B]))
    else:
        return (Matrix(2, 2, [S.One / a, 0, 0, 1]), Matrix([0, 0]))

def find_DN(eq):
    if False:
        i = 10
        return i + 15
    '\n    This function returns a tuple, `(D, N)` of the simplified form,\n    `x^2 - Dy^2 = N`, corresponding to the general quadratic,\n    `ax^2 + bxy + cy^2 + dx + ey + f = 0`.\n\n    Solving the general quadratic is then equivalent to solving the equation\n    `X^2 - DY^2 = N` and transforming the solutions by using the transformation\n    matrices returned by ``transformation_to_DN()``.\n\n    Usage\n    =====\n\n    ``find_DN(eq)``: where ``eq`` is the quadratic to be transformed.\n\n    Examples\n    ========\n\n    >>> from sympy.abc import x, y\n    >>> from sympy.solvers.diophantine.diophantine import find_DN\n    >>> find_DN(x**2 - 3*x*y - y**2 - 2*y + 1)\n    (13, -884)\n\n    Interpretation of the output is that we get `X^2 -13Y^2 = -884` after\n    transforming `x^2 - 3xy - y^2 - 2y + 1` using the transformation returned\n    by ``transformation_to_DN()``.\n\n    See Also\n    ========\n\n    transformation_to_DN()\n\n    References\n    ==========\n\n    .. [1] Solving the equation ax^2 + bxy + cy^2 + dx + ey + f = 0,\n           John P.Robertson, May 8, 2003, Page 7 - 11.\n           https://web.archive.org/web/20160323033111/http://www.jpr2718.org/ax2p.pdf\n    '
    (var, coeff, diop_type) = classify_diop(eq, _dict=False)
    if diop_type == BinaryQuadratic.name:
        return _find_DN(var, coeff)

def _find_DN(var, coeff):
    if False:
        for i in range(10):
            print('nop')
    (x, y) = var
    (X, Y) = symbols('X, Y', integer=True)
    (A, B) = _transformation_to_DN(var, coeff)
    u = (A * Matrix([X, Y]) + B)[0]
    v = (A * Matrix([X, Y]) + B)[1]
    eq = x ** 2 * coeff[x ** 2] + x * y * coeff[x * y] + y ** 2 * coeff[y ** 2] + x * coeff[x] + y * coeff[y] + coeff[1]
    simplified = _mexpand(eq.subs(zip((x, y), (u, v))))
    coeff = simplified.as_coefficients_dict()
    return (-coeff[Y ** 2] / coeff[X ** 2], -coeff[1] / coeff[X ** 2])

def check_param(x, y, a, params):
    if False:
        return 10
    '\n    If there is a number modulo ``a`` such that ``x`` and ``y`` are both\n    integers, then return a parametric representation for ``x`` and ``y``\n    else return (None, None).\n\n    Here ``x`` and ``y`` are functions of ``t``.\n    '
    from sympy.simplify.simplify import clear_coefficients
    if x.is_number and (not x.is_Integer):
        return DiophantineSolutionSet([x, y], parameters=params)
    if y.is_number and (not y.is_Integer):
        return DiophantineSolutionSet([x, y], parameters=params)
    (m, n) = symbols('m, n', integer=True)
    (c, p) = (m * x + n * y).as_content_primitive()
    if a % c.q:
        return DiophantineSolutionSet([x, y], parameters=params)
    eq = clear_coefficients(x, m)[1] - clear_coefficients(y, n)[1]
    (junk, eq) = eq.as_content_primitive()
    return _diop_solve(eq, params=params)

def diop_ternary_quadratic(eq, parameterize=False):
    if False:
        print('Hello World!')
    '\n    Solves the general quadratic ternary form,\n    `ax^2 + by^2 + cz^2 + fxy + gyz + hxz = 0`.\n\n    Returns a tuple `(x, y, z)` which is a base solution for the above\n    equation. If there are no solutions, `(None, None, None)` is returned.\n\n    Usage\n    =====\n\n    ``diop_ternary_quadratic(eq)``: Return a tuple containing a basic solution\n    to ``eq``.\n\n    Details\n    =======\n\n    ``eq`` should be an homogeneous expression of degree two in three variables\n    and it is assumed to be zero.\n\n    Examples\n    ========\n\n    >>> from sympy.abc import x, y, z\n    >>> from sympy.solvers.diophantine.diophantine import diop_ternary_quadratic\n    >>> diop_ternary_quadratic(x**2 + 3*y**2 - z**2)\n    (1, 0, 1)\n    >>> diop_ternary_quadratic(4*x**2 + 5*y**2 - z**2)\n    (1, 0, 2)\n    >>> diop_ternary_quadratic(45*x**2 - 7*y**2 - 8*x*y - z**2)\n    (28, 45, 105)\n    >>> diop_ternary_quadratic(x**2 - 49*y**2 - z**2 + 13*z*y -8*x*y)\n    (9, 1, 5)\n    '
    (var, coeff, diop_type) = classify_diop(eq, _dict=False)
    if diop_type in (HomogeneousTernaryQuadratic.name, HomogeneousTernaryQuadraticNormal.name):
        sol = _diop_ternary_quadratic(var, coeff)
        if len(sol) > 0:
            (x_0, y_0, z_0) = list(sol)[0]
        else:
            (x_0, y_0, z_0) = (None, None, None)
        if parameterize:
            return _parametrize_ternary_quadratic((x_0, y_0, z_0), var, coeff)
        return (x_0, y_0, z_0)

def _diop_ternary_quadratic(_var, coeff):
    if False:
        for i in range(10):
            print('nop')
    eq = sum([i * coeff[i] for i in coeff])
    if HomogeneousTernaryQuadratic(eq).matches():
        return HomogeneousTernaryQuadratic(eq, free_symbols=_var).solve()
    elif HomogeneousTernaryQuadraticNormal(eq).matches():
        return HomogeneousTernaryQuadraticNormal(eq, free_symbols=_var).solve()

def transformation_to_normal(eq):
    if False:
        return 10
    '\n    Returns the transformation Matrix that converts a general ternary\n    quadratic equation ``eq`` (`ax^2 + by^2 + cz^2 + dxy + eyz + fxz`)\n    to a form without cross terms: `ax^2 + by^2 + cz^2 = 0`. This is\n    not used in solving ternary quadratics; it is only implemented for\n    the sake of completeness.\n    '
    (var, coeff, diop_type) = classify_diop(eq, _dict=False)
    if diop_type in ('homogeneous_ternary_quadratic', 'homogeneous_ternary_quadratic_normal'):
        return _transformation_to_normal(var, coeff)

def _transformation_to_normal(var, coeff):
    if False:
        for i in range(10):
            print('nop')
    _var = list(var)
    (x, y, z) = var
    if not any((coeff[i ** 2] for i in var)):
        a = coeff[x * y]
        b = coeff[y * z]
        c = coeff[x * z]
        swap = False
        if not a:
            swap = True
            (a, b) = (b, a)
        T = Matrix(((1, 1, -b / a), (1, -1, -c / a), (0, 0, 1)))
        if swap:
            T.row_swap(0, 1)
            T.col_swap(0, 1)
        return T
    if coeff[x ** 2] == 0:
        if coeff[y ** 2] == 0:
            (_var[0], _var[2]) = (var[2], var[0])
            T = _transformation_to_normal(_var, coeff)
            T.row_swap(0, 2)
            T.col_swap(0, 2)
            return T
        else:
            (_var[0], _var[1]) = (var[1], var[0])
            T = _transformation_to_normal(_var, coeff)
            T.row_swap(0, 1)
            T.col_swap(0, 1)
            return T
    if coeff[x * y] != 0 or coeff[x * z] != 0:
        A = coeff[x ** 2]
        B = coeff[x * y]
        C = coeff[x * z]
        D = coeff[y ** 2]
        E = coeff[y * z]
        F = coeff[z ** 2]
        _coeff = {}
        _coeff[x ** 2] = 4 * A ** 2
        _coeff[y ** 2] = 4 * A * D - B ** 2
        _coeff[z ** 2] = 4 * A * F - C ** 2
        _coeff[y * z] = 4 * A * E - 2 * B * C
        _coeff[x * y] = 0
        _coeff[x * z] = 0
        T_0 = _transformation_to_normal(_var, _coeff)
        return Matrix(3, 3, [1, S(-B) / (2 * A), S(-C) / (2 * A), 0, 1, 0, 0, 0, 1]) * T_0
    elif coeff[y * z] != 0:
        if coeff[y ** 2] == 0:
            if coeff[z ** 2] == 0:
                return Matrix(3, 3, [1, 0, 0, 0, 1, 1, 0, 1, -1])
            else:
                (_var[0], _var[2]) = (var[2], var[0])
                T = _transformation_to_normal(_var, coeff)
                T.row_swap(0, 2)
                T.col_swap(0, 2)
                return T
        else:
            (_var[0], _var[1]) = (var[1], var[0])
            T = _transformation_to_normal(_var, coeff)
            T.row_swap(0, 1)
            T.col_swap(0, 1)
            return T
    else:
        return Matrix.eye(3)

def parametrize_ternary_quadratic(eq):
    if False:
        print('Hello World!')
    '\n    Returns the parametrized general solution for the ternary quadratic\n    equation ``eq`` which has the form\n    `ax^2 + by^2 + cz^2 + fxy + gyz + hxz = 0`.\n\n    Examples\n    ========\n\n    >>> from sympy import Tuple, ordered\n    >>> from sympy.abc import x, y, z\n    >>> from sympy.solvers.diophantine.diophantine import parametrize_ternary_quadratic\n\n    The parametrized solution may be returned with three parameters:\n\n    >>> parametrize_ternary_quadratic(2*x**2 + y**2 - 2*z**2)\n    (p**2 - 2*q**2, -2*p**2 + 4*p*q - 4*p*r - 4*q**2, p**2 - 4*p*q + 2*q**2 - 4*q*r)\n\n    There might also be only two parameters:\n\n    >>> parametrize_ternary_quadratic(4*x**2 + 2*y**2 - 3*z**2)\n    (2*p**2 - 3*q**2, -4*p**2 + 12*p*q - 6*q**2, 4*p**2 - 8*p*q + 6*q**2)\n\n    Notes\n    =====\n\n    Consider ``p`` and ``q`` in the previous 2-parameter\n    solution and observe that more than one solution can be represented\n    by a given pair of parameters. If `p` and ``q`` are not coprime, this is\n    trivially true since the common factor will also be a common factor of the\n    solution values. But it may also be true even when ``p`` and\n    ``q`` are coprime:\n\n    >>> sol = Tuple(*_)\n    >>> p, q = ordered(sol.free_symbols)\n    >>> sol.subs([(p, 3), (q, 2)])\n    (6, 12, 12)\n    >>> sol.subs([(q, 1), (p, 1)])\n    (-1, 2, 2)\n    >>> sol.subs([(q, 0), (p, 1)])\n    (2, -4, 4)\n    >>> sol.subs([(q, 1), (p, 0)])\n    (-3, -6, 6)\n\n    Except for sign and a common factor, these are equivalent to\n    the solution of (1, 2, 2).\n\n    References\n    ==========\n\n    .. [1] The algorithmic resolution of Diophantine equations, Nigel P. Smart,\n           London Mathematical Society Student Texts 41, Cambridge University\n           Press, Cambridge, 1998.\n\n    '
    (var, coeff, diop_type) = classify_diop(eq, _dict=False)
    if diop_type in ('homogeneous_ternary_quadratic', 'homogeneous_ternary_quadratic_normal'):
        (x_0, y_0, z_0) = list(_diop_ternary_quadratic(var, coeff))[0]
        return _parametrize_ternary_quadratic((x_0, y_0, z_0), var, coeff)

def _parametrize_ternary_quadratic(solution, _var, coeff):
    if False:
        while True:
            i = 10
    assert 1 not in coeff
    (x_0, y_0, z_0) = solution
    v = list(_var)
    if x_0 is None:
        return (None, None, None)
    if solution.count(0) >= 2:
        return (None, None, None)
    if x_0 == 0:
        (v[0], v[1]) = (v[1], v[0])
        (y_p, x_p, z_p) = _parametrize_ternary_quadratic((y_0, x_0, z_0), v, coeff)
        return (x_p, y_p, z_p)
    (x, y, z) = v
    (r, p, q) = symbols('r, p, q', integer=True)
    eq = sum((k * v for (k, v) in coeff.items()))
    eq_1 = _mexpand(eq.subs(zip((x, y, z), (r * x_0, r * y_0 + p, r * z_0 + q))))
    (A, B) = eq_1.as_independent(r, as_Add=True)
    x = A * x_0
    y = A * y_0 - _mexpand(B / r * p)
    z = A * z_0 - _mexpand(B / r * q)
    return _remove_gcd(x, y, z)

def diop_ternary_quadratic_normal(eq, parameterize=False):
    if False:
        return 10
    '\n    Solves the quadratic ternary diophantine equation,\n    `ax^2 + by^2 + cz^2 = 0`.\n\n    Explanation\n    ===========\n\n    Here the coefficients `a`, `b`, and `c` should be non zero. Otherwise the\n    equation will be a quadratic binary or univariate equation. If solvable,\n    returns a tuple `(x, y, z)` that satisfies the given equation. If the\n    equation does not have integer solutions, `(None, None, None)` is returned.\n\n    Usage\n    =====\n\n    ``diop_ternary_quadratic_normal(eq)``: where ``eq`` is an equation of the form\n    `ax^2 + by^2 + cz^2 = 0`.\n\n    Examples\n    ========\n\n    >>> from sympy.abc import x, y, z\n    >>> from sympy.solvers.diophantine.diophantine import diop_ternary_quadratic_normal\n    >>> diop_ternary_quadratic_normal(x**2 + 3*y**2 - z**2)\n    (1, 0, 1)\n    >>> diop_ternary_quadratic_normal(4*x**2 + 5*y**2 - z**2)\n    (1, 0, 2)\n    >>> diop_ternary_quadratic_normal(34*x**2 - 3*y**2 - 301*z**2)\n    (4, 9, 1)\n    '
    (var, coeff, diop_type) = classify_diop(eq, _dict=False)
    if diop_type == HomogeneousTernaryQuadraticNormal.name:
        sol = _diop_ternary_quadratic_normal(var, coeff)
        if len(sol) > 0:
            (x_0, y_0, z_0) = list(sol)[0]
        else:
            (x_0, y_0, z_0) = (None, None, None)
        if parameterize:
            return _parametrize_ternary_quadratic((x_0, y_0, z_0), var, coeff)
        return (x_0, y_0, z_0)

def _diop_ternary_quadratic_normal(var, coeff):
    if False:
        print('Hello World!')
    eq = sum([i * coeff[i] for i in coeff])
    return HomogeneousTernaryQuadraticNormal(eq, free_symbols=var).solve()

def sqf_normal(a, b, c, steps=False):
    if False:
        i = 10
        return i + 15
    "\n    Return `a', b', c'`, the coefficients of the square-free normal\n    form of `ax^2 + by^2 + cz^2 = 0`, where `a', b', c'` are pairwise\n    prime.  If `steps` is True then also return three tuples:\n    `sq`, `sqf`, and `(a', b', c')` where `sq` contains the square\n    factors of `a`, `b` and `c` after removing the `gcd(a, b, c)`;\n    `sqf` contains the values of `a`, `b` and `c` after removing\n    both the `gcd(a, b, c)` and the square factors.\n\n    The solutions for `ax^2 + by^2 + cz^2 = 0` can be\n    recovered from the solutions of `a'x^2 + b'y^2 + c'z^2 = 0`.\n\n    Examples\n    ========\n\n    >>> from sympy.solvers.diophantine.diophantine import sqf_normal\n    >>> sqf_normal(2 * 3**2 * 5, 2 * 5 * 11, 2 * 7**2 * 11)\n    (11, 1, 5)\n    >>> sqf_normal(2 * 3**2 * 5, 2 * 5 * 11, 2 * 7**2 * 11, True)\n    ((3, 1, 7), (5, 55, 11), (11, 1, 5))\n\n    References\n    ==========\n\n    .. [1] Legendre's Theorem, Legrange's Descent,\n           https://public.csusm.edu/aitken_html/notes/legendre.pdf\n\n\n    See Also\n    ========\n\n    reconstruct()\n    "
    ABC = _remove_gcd(a, b, c)
    sq = tuple((square_factor(i) for i in ABC))
    sqf = (A, B, C) = tuple([i // j ** 2 for (i, j) in zip(ABC, sq)])
    pc = igcd(A, B)
    A /= pc
    B /= pc
    pa = igcd(B, C)
    B /= pa
    C /= pa
    pb = igcd(A, C)
    A /= pb
    B /= pb
    A *= pa
    B *= pb
    C *= pc
    if steps:
        return (sq, sqf, (A, B, C))
    else:
        return (A, B, C)

def square_factor(a):
    if False:
        print('Hello World!')
    '\n    Returns an integer `c` s.t. `a = c^2k, \\ c,k \\in Z`. Here `k` is square\n    free. `a` can be given as an integer or a dictionary of factors.\n\n    Examples\n    ========\n\n    >>> from sympy.solvers.diophantine.diophantine import square_factor\n    >>> square_factor(24)\n    2\n    >>> square_factor(-36*3)\n    6\n    >>> square_factor(1)\n    1\n    >>> square_factor({3: 2, 2: 1, -1: 1})  # -18\n    3\n\n    See Also\n    ========\n    sympy.ntheory.factor_.core\n    '
    f = a if isinstance(a, dict) else factorint(a)
    return Mul(*[p ** (e // 2) for (p, e) in f.items()])

def reconstruct(A, B, z):
    if False:
        i = 10
        return i + 15
    "\n    Reconstruct the `z` value of an equivalent solution of `ax^2 + by^2 + cz^2`\n    from the `z` value of a solution of the square-free normal form of the\n    equation, `a'*x^2 + b'*y^2 + c'*z^2`, where `a'`, `b'` and `c'` are square\n    free and `gcd(a', b', c') == 1`.\n    "
    f = factorint(igcd(A, B))
    for (p, e) in f.items():
        if e != 1:
            raise ValueError('a and b should be square-free')
        z *= p
    return z

def ldescent(A, B):
    if False:
        for i in range(10):
            print('nop')
    "\n    Return a non-trivial solution to `w^2 = Ax^2 + By^2` using\n    Lagrange's method; return None if there is no such solution.\n    .\n\n    Here, `A \\neq 0` and `B \\neq 0` and `A` and `B` are square free. Output a\n    tuple `(w_0, x_0, y_0)` which is a solution to the above equation.\n\n    Examples\n    ========\n\n    >>> from sympy.solvers.diophantine.diophantine import ldescent\n    >>> ldescent(1, 1) # w^2 = x^2 + y^2\n    (1, 1, 0)\n    >>> ldescent(4, -7) # w^2 = 4x^2 - 7y^2\n    (2, -1, 0)\n\n    This means that `x = -1, y = 0` and `w = 2` is a solution to the equation\n    `w^2 = 4x^2 - 7y^2`\n\n    >>> ldescent(5, -1) # w^2 = 5x^2 - y^2\n    (2, 1, -1)\n\n    References\n    ==========\n\n    .. [1] The algorithmic resolution of Diophantine equations, Nigel P. Smart,\n           London Mathematical Society Student Texts 41, Cambridge University\n           Press, Cambridge, 1998.\n    .. [2] Efficient Solution of Rational Conices, J. E. Cremona and D. Rusin,\n           [online], Available:\n           https://nottingham-repository.worktribe.com/output/1023265/efficient-solution-of-rational-conics\n    "
    if abs(A) > abs(B):
        (w, y, x) = ldescent(B, A)
        return (w, x, y)
    if A == 1:
        return (1, 1, 0)
    if B == 1:
        return (1, 0, 1)
    if B == -1:
        return
    r = sqrt_mod(A, B)
    Q = (r ** 2 - A) // B
    if Q == 0:
        B_0 = 1
        d = 0
    else:
        div = divisors(Q)
        B_0 = None
        for i in div:
            (sQ, _exact) = integer_nthroot(abs(Q) // i, 2)
            if _exact:
                (B_0, d) = (sign(Q) * i, sQ)
                break
    if B_0 is not None:
        (W, X, Y) = ldescent(A, B_0)
        return _remove_gcd(-A * X + r * W, r * X - W, Y * (B_0 * d))

def descent(A, B):
    if False:
        return 10
    "\n    Returns a non-trivial solution, (x, y, z), to `x^2 = Ay^2 + Bz^2`\n    using Lagrange's descent method with lattice-reduction. `A` and `B`\n    are assumed to be valid for such a solution to exist.\n\n    This is faster than the normal Lagrange's descent algorithm because\n    the Gaussian reduction is used.\n\n    Examples\n    ========\n\n    >>> from sympy.solvers.diophantine.diophantine import descent\n    >>> descent(3, 1) # x**2 = 3*y**2 + z**2\n    (1, 0, 1)\n\n    `(x, y, z) = (1, 0, 1)` is a solution to the above equation.\n\n    >>> descent(41, -113)\n    (-16, -3, 1)\n\n    References\n    ==========\n\n    .. [1] Efficient Solution of Rational Conices, J. E. Cremona and D. Rusin,\n           Mathematics of Computation, Volume 00, Number 0.\n    "
    if abs(A) > abs(B):
        (x, y, z) = descent(B, A)
        return (x, z, y)
    if B == 1:
        return (1, 0, 1)
    if A == 1:
        return (1, 1, 0)
    if B == -A:
        return (0, 1, 1)
    if B == A:
        (x, z, y) = descent(-1, A)
        return (A * y, z, x)
    w = sqrt_mod(A, B)
    (x_0, z_0) = gaussian_reduce(w, A, B)
    t = (x_0 ** 2 - A * z_0 ** 2) // B
    t_2 = square_factor(t)
    t_1 = t // t_2 ** 2
    (x_1, z_1, y_1) = descent(A, t_1)
    return _remove_gcd(x_0 * x_1 + A * z_0 * z_1, z_0 * x_1 + x_0 * z_1, t_1 * t_2 * y_1)

def gaussian_reduce(w, a, b):
    if False:
        while True:
            i = 10
    '\n    Returns a reduced solution `(x, z)` to the congruence\n    `X^2 - aZ^2 \\equiv 0 \\ (mod \\ b)` so that `x^2 + |a|z^2` is minimal.\n\n    Details\n    =======\n\n    Here ``w`` is a solution of the congruence `x^2 \\equiv a \\ (mod \\ b)`\n\n    References\n    ==========\n\n    .. [1] Gaussian lattice Reduction [online]. Available:\n           https://web.archive.org/web/20201021115213/http://home.ie.cuhk.edu.hk/~wkshum/wordpress/?p=404\n    .. [2] Efficient Solution of Rational Conices, J. E. Cremona and D. Rusin,\n           Mathematics of Computation, Volume 00, Number 0.\n    '
    u = (0, 1)
    v = (1, 0)
    if dot(u, v, w, a, b) < 0:
        v = (-v[0], -v[1])
    if norm(u, w, a, b) < norm(v, w, a, b):
        (u, v) = (v, u)
    while norm(u, w, a, b) > norm(v, w, a, b):
        k = dot(u, v, w, a, b) // dot(v, v, w, a, b)
        (u, v) = (v, (u[0] - k * v[0], u[1] - k * v[1]))
    (u, v) = (v, u)
    if dot(u, v, w, a, b) < dot(v, v, w, a, b) / 2 or norm((u[0] - v[0], u[1] - v[1]), w, a, b) > norm(v, w, a, b):
        c = v
    else:
        c = (u[0] - v[0], u[1] - v[1])
    return (c[0] * w + b * c[1], c[0])

def dot(u, v, w, a, b):
    if False:
        i = 10
        return i + 15
    '\n    Returns a special dot product of the vectors `u = (u_{1}, u_{2})` and\n    `v = (v_{1}, v_{2})` which is defined in order to reduce solution of\n    the congruence equation `X^2 - aZ^2 \\equiv 0 \\ (mod \\ b)`.\n    '
    (u_1, u_2) = u
    (v_1, v_2) = v
    return (w * u_1 + b * u_2) * (w * v_1 + b * v_2) + abs(a) * u_1 * v_1

def norm(u, w, a, b):
    if False:
        while True:
            i = 10
    '\n    Returns the norm of the vector `u = (u_{1}, u_{2})` under the dot product\n    defined by `u \\cdot v = (wu_{1} + bu_{2})(w*v_{1} + bv_{2}) + |a|*u_{1}*v_{1}`\n    where `u = (u_{1}, u_{2})` and `v = (v_{1}, v_{2})`.\n    '
    (u_1, u_2) = u
    return sqrt(dot((u_1, u_2), (u_1, u_2), w, a, b))

def holzer(x, y, z, a, b, c):
    if False:
        i = 10
        return i + 15
    "\n    Simplify the solution `(x, y, z)` of the equation\n    `ax^2 + by^2 = cz^2` with `a, b, c > 0` and `z^2 \\geq \\mid ab \\mid` to\n    a new reduced solution `(x', y', z')` such that `z'^2 \\leq \\mid ab \\mid`.\n\n    The algorithm is an interpretation of Mordell's reduction as described\n    on page 8 of Cremona and Rusin's paper [1]_ and the work of Mordell in\n    reference [2]_.\n\n    References\n    ==========\n\n    .. [1] Efficient Solution of Rational Conices, J. E. Cremona and D. Rusin,\n           Mathematics of Computation, Volume 00, Number 0.\n    .. [2] Diophantine Equations, L. J. Mordell, page 48.\n\n    "
    if _odd(c):
        k = 2 * c
    else:
        k = c // 2
    small = a * b * c
    step = 0
    while True:
        (t1, t2, t3) = (a * x ** 2, b * y ** 2, c * z ** 2)
        if t1 + t2 != t3:
            if step == 0:
                raise ValueError('bad starting solution')
            break
        (x_0, y_0, z_0) = (x, y, z)
        if max(t1, t2, t3) <= small:
            break
        uv = (u, v) = base_solution_linear(k, y_0, -x_0)
        if None in uv:
            break
        (p, q) = (-(a * u * x_0 + b * v * y_0), c * z_0)
        r = Rational(p, q)
        if _even(c):
            w = _nint_or_floor(p, q)
            assert abs(w - r) <= S.Half
        else:
            w = p // q
            if _odd(a * u + b * v + c * w):
                w += 1
            assert abs(w - r) <= S.One
        A = a * u ** 2 + b * v ** 2 + c * w ** 2
        B = a * u * x_0 + b * v * y_0 + c * w * z_0
        x = Rational(x_0 * A - 2 * u * B, k)
        y = Rational(y_0 * A - 2 * v * B, k)
        z = Rational(z_0 * A - 2 * w * B, k)
        assert all((i.is_Integer for i in (x, y, z)))
        step += 1
    return tuple([int(i) for i in (x_0, y_0, z_0)])

def diop_general_pythagorean(eq, param=symbols('m', integer=True)):
    if False:
        for i in range(10):
            print('nop')
    '\n    Solves the general pythagorean equation,\n    `a_{1}^2x_{1}^2 + a_{2}^2x_{2}^2 + . . . + a_{n}^2x_{n}^2 - a_{n + 1}^2x_{n + 1}^2 = 0`.\n\n    Returns a tuple which contains a parametrized solution to the equation,\n    sorted in the same order as the input variables.\n\n    Usage\n    =====\n\n    ``diop_general_pythagorean(eq, param)``: where ``eq`` is a general\n    pythagorean equation which is assumed to be zero and ``param`` is the base\n    parameter used to construct other parameters by subscripting.\n\n    Examples\n    ========\n\n    >>> from sympy.solvers.diophantine.diophantine import diop_general_pythagorean\n    >>> from sympy.abc import a, b, c, d, e\n    >>> diop_general_pythagorean(a**2 + b**2 + c**2 - d**2)\n    (m1**2 + m2**2 - m3**2, 2*m1*m3, 2*m2*m3, m1**2 + m2**2 + m3**2)\n    >>> diop_general_pythagorean(9*a**2 - 4*b**2 + 16*c**2 + 25*d**2 + e**2)\n    (10*m1**2  + 10*m2**2  + 10*m3**2 - 10*m4**2, 15*m1**2  + 15*m2**2  + 15*m3**2  + 15*m4**2, 15*m1*m4, 12*m2*m4, 60*m3*m4)\n    '
    (var, coeff, diop_type) = classify_diop(eq, _dict=False)
    if diop_type == GeneralPythagorean.name:
        if param is None:
            params = None
        else:
            params = symbols('%s1:%i' % (param, len(var)), integer=True)
        return list(GeneralPythagorean(eq).solve(parameters=params))[0]

def diop_general_sum_of_squares(eq, limit=1):
    if False:
        print('Hello World!')
    '\n    Solves the equation `x_{1}^2 + x_{2}^2 + . . . + x_{n}^2 - k = 0`.\n\n    Returns at most ``limit`` number of solutions.\n\n    Usage\n    =====\n\n    ``general_sum_of_squares(eq, limit)`` : Here ``eq`` is an expression which\n    is assumed to be zero. Also, ``eq`` should be in the form,\n    `x_{1}^2 + x_{2}^2 + . . . + x_{n}^2 - k = 0`.\n\n    Details\n    =======\n\n    When `n = 3` if `k = 4^a(8m + 7)` for some `a, m \\in Z` then there will be\n    no solutions. Refer to [1]_ for more details.\n\n    Examples\n    ========\n\n    >>> from sympy.solvers.diophantine.diophantine import diop_general_sum_of_squares\n    >>> from sympy.abc import a, b, c, d, e\n    >>> diop_general_sum_of_squares(a**2 + b**2 + c**2 + d**2 + e**2 - 2345)\n    {(15, 22, 22, 24, 24)}\n\n    Reference\n    =========\n\n    .. [1] Representing an integer as a sum of three squares, [online],\n        Available:\n        https://www.proofwiki.org/wiki/Integer_as_Sum_of_Three_Squares\n    '
    (var, coeff, diop_type) = classify_diop(eq, _dict=False)
    if diop_type == GeneralSumOfSquares.name:
        return set(GeneralSumOfSquares(eq).solve(limit=limit))

def diop_general_sum_of_even_powers(eq, limit=1):
    if False:
        while True:
            i = 10
    '\n    Solves the equation `x_{1}^e + x_{2}^e + . . . + x_{n}^e - k = 0`\n    where `e` is an even, integer power.\n\n    Returns at most ``limit`` number of solutions.\n\n    Usage\n    =====\n\n    ``general_sum_of_even_powers(eq, limit)`` : Here ``eq`` is an expression which\n    is assumed to be zero. Also, ``eq`` should be in the form,\n    `x_{1}^e + x_{2}^e + . . . + x_{n}^e - k = 0`.\n\n    Examples\n    ========\n\n    >>> from sympy.solvers.diophantine.diophantine import diop_general_sum_of_even_powers\n    >>> from sympy.abc import a, b\n    >>> diop_general_sum_of_even_powers(a**4 + b**4 - (2**4 + 3**4))\n    {(2, 3)}\n\n    See Also\n    ========\n\n    power_representation\n    '
    (var, coeff, diop_type) = classify_diop(eq, _dict=False)
    if diop_type == GeneralSumOfEvenPowers.name:
        return set(GeneralSumOfEvenPowers(eq).solve(limit=limit))

def partition(n, k=None, zeros=False):
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns a generator that can be used to generate partitions of an integer\n    `n`.\n\n    Explanation\n    ===========\n\n    A partition of `n` is a set of positive integers which add up to `n`. For\n    example, partitions of 3 are 3, 1 + 2, 1 + 1 + 1. A partition is returned\n    as a tuple. If ``k`` equals None, then all possible partitions are returned\n    irrespective of their size, otherwise only the partitions of size ``k`` are\n    returned. If the ``zero`` parameter is set to True then a suitable\n    number of zeros are added at the end of every partition of size less than\n    ``k``.\n\n    ``zero`` parameter is considered only if ``k`` is not None. When the\n    partitions are over, the last `next()` call throws the ``StopIteration``\n    exception, so this function should always be used inside a try - except\n    block.\n\n    Details\n    =======\n\n    ``partition(n, k)``: Here ``n`` is a positive integer and ``k`` is the size\n    of the partition which is also positive integer.\n\n    Examples\n    ========\n\n    >>> from sympy.solvers.diophantine.diophantine import partition\n    >>> f = partition(5)\n    >>> next(f)\n    (1, 1, 1, 1, 1)\n    >>> next(f)\n    (1, 1, 1, 2)\n    >>> g = partition(5, 3)\n    >>> next(g)\n    (1, 1, 3)\n    >>> next(g)\n    (1, 2, 2)\n    >>> g = partition(5, 3, zeros=True)\n    >>> next(g)\n    (0, 0, 5)\n\n    '
    if not zeros or k is None:
        for i in ordered_partitions(n, k):
            yield tuple(i)
    else:
        for m in range(1, k + 1):
            for i in ordered_partitions(n, m):
                i = tuple(i)
                yield ((0,) * (k - len(i)) + i)

def prime_as_sum_of_two_squares(p):
    if False:
        return 10
    "\n    Represent a prime `p` as a unique sum of two squares; this can\n    only be done if the prime is congruent to 1 mod 4.\n\n    Examples\n    ========\n\n    >>> from sympy.solvers.diophantine.diophantine import prime_as_sum_of_two_squares\n    >>> prime_as_sum_of_two_squares(7)  # can't be done\n    >>> prime_as_sum_of_two_squares(5)\n    (1, 2)\n\n    Reference\n    =========\n\n    .. [1] Representing a number as a sum of four squares, [online],\n        Available: https://schorn.ch/lagrange.html\n\n    See Also\n    ========\n    sum_of_squares()\n    "
    if not p % 4 == 1:
        return
    if p % 8 == 5:
        b = 2
    else:
        b = 3
        while pow(b, (p - 1) // 2, p) == 1:
            b = nextprime(b)
    b = pow(b, (p - 1) // 4, p)
    a = p
    while b ** 2 > p:
        (a, b) = (b, a % b)
    return (int(a % b), int(b))

def sum_of_three_squares(n):
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns a 3-tuple $(a, b, c)$ such that $a^2 + b^2 + c^2 = n$ and\n    $a, b, c \\geq 0$.\n\n    Returns None if $n = 4^a(8m + 7)$ for some `a, m \\in \\mathbb{Z}`. See\n    [1]_ for more details.\n\n    Usage\n    =====\n\n    ``sum_of_three_squares(n)``: Here ``n`` is a non-negative integer.\n\n    Examples\n    ========\n\n    >>> from sympy.solvers.diophantine.diophantine import sum_of_three_squares\n    >>> sum_of_three_squares(44542)\n    (18, 37, 207)\n\n    References\n    ==========\n\n    .. [1] Representing a number as a sum of three squares, [online],\n        Available: https://schorn.ch/lagrange.html\n\n    See Also\n    ========\n\n    sum_of_squares()\n    '
    special = {1: (1, 0, 0), 2: (1, 1, 0), 3: (1, 1, 1), 10: (1, 3, 0), 34: (3, 3, 4), 58: (3, 7, 0), 85: (6, 7, 0), 130: (3, 11, 0), 214: (3, 6, 13), 226: (8, 9, 9), 370: (8, 9, 15), 526: (6, 7, 21), 706: (15, 15, 16), 730: (1, 27, 0), 1414: (6, 17, 33), 1906: (13, 21, 36), 2986: (21, 32, 39), 9634: (56, 57, 57)}
    v = 0
    if n == 0:
        return (0, 0, 0)
    v = multiplicity(4, n)
    n //= 4 ** v
    if n % 8 == 7:
        return
    if n in special.keys():
        (x, y, z) = special[n]
        return _sorted_tuple(2 ** v * x, 2 ** v * y, 2 ** v * z)
    (s, _exact) = integer_nthroot(n, 2)
    if _exact:
        return (2 ** v * s, 0, 0)
    x = None
    if n % 8 == 3:
        s = s if _odd(s) else s - 1
        for x in range(s, -1, -2):
            N = (n - x ** 2) // 2
            if isprime(N):
                (y, z) = prime_as_sum_of_two_squares(N)
                return _sorted_tuple(2 ** v * x, 2 ** v * (y + z), 2 ** v * abs(y - z))
        return
    if n % 8 in (2, 6):
        s = s if _odd(s) else s - 1
    else:
        s = s - 1 if _odd(s) else s
    for x in range(s, -1, -2):
        N = n - x ** 2
        if isprime(N):
            (y, z) = prime_as_sum_of_two_squares(N)
            return _sorted_tuple(2 ** v * x, 2 ** v * y, 2 ** v * z)

def sum_of_four_squares(n):
    if False:
        print('Hello World!')
    '\n    Returns a 4-tuple `(a, b, c, d)` such that `a^2 + b^2 + c^2 + d^2 = n`.\n\n    Here `a, b, c, d \\geq 0`.\n\n    Usage\n    =====\n\n    ``sum_of_four_squares(n)``: Here ``n`` is a non-negative integer.\n\n    Examples\n    ========\n\n    >>> from sympy.solvers.diophantine.diophantine import sum_of_four_squares\n    >>> sum_of_four_squares(3456)\n    (8, 8, 32, 48)\n    >>> sum_of_four_squares(1294585930293)\n    (0, 1234, 2161, 1137796)\n\n    References\n    ==========\n\n    .. [1] Representing a number as a sum of four squares, [online],\n        Available: https://schorn.ch/lagrange.html\n\n    See Also\n    ========\n\n    sum_of_squares()\n    '
    if n == 0:
        return (0, 0, 0, 0)
    v = multiplicity(4, n)
    n //= 4 ** v
    if n % 8 == 7:
        d = 2
        n = n - 4
    elif n % 8 in (2, 6):
        d = 1
        n = n - 1
    else:
        d = 0
    (x, y, z) = sum_of_three_squares(n)
    return _sorted_tuple(2 ** v * d, 2 ** v * x, 2 ** v * y, 2 ** v * z)

def power_representation(n, p, k, zeros=False):
    if False:
        while True:
            i = 10
    '\n    Returns a generator for finding k-tuples of integers,\n    `(n_{1}, n_{2}, . . . n_{k})`, such that\n    `n = n_{1}^p + n_{2}^p + . . . n_{k}^p`.\n\n    Usage\n    =====\n\n    ``power_representation(n, p, k, zeros)``: Represent non-negative number\n    ``n`` as a sum of ``k`` ``p``\\ th powers. If ``zeros`` is true, then the\n    solutions is allowed to contain zeros.\n\n    Examples\n    ========\n\n    >>> from sympy.solvers.diophantine.diophantine import power_representation\n\n    Represent 1729 as a sum of two cubes:\n\n    >>> f = power_representation(1729, 3, 2)\n    >>> next(f)\n    (9, 10)\n    >>> next(f)\n    (1, 12)\n\n    If the flag `zeros` is True, the solution may contain tuples with\n    zeros; any such solutions will be generated after the solutions\n    without zeros:\n\n    >>> list(power_representation(125, 2, 3, zeros=True))\n    [(5, 6, 8), (3, 4, 10), (0, 5, 10), (0, 2, 11)]\n\n    For even `p` the `permute_sign` function can be used to get all\n    signed values:\n\n    >>> from sympy.utilities.iterables import permute_signs\n    >>> list(permute_signs((1, 12)))\n    [(1, 12), (-1, 12), (1, -12), (-1, -12)]\n\n    All possible signed permutations can also be obtained:\n\n    >>> from sympy.utilities.iterables import signed_permutations\n    >>> list(signed_permutations((1, 12)))\n    [(1, 12), (-1, 12), (1, -12), (-1, -12), (12, 1), (-12, 1), (12, -1), (-12, -1)]\n    '
    (n, p, k) = [as_int(i) for i in (n, p, k)]
    if n < 0:
        if p % 2:
            for t in power_representation(-n, p, k, zeros):
                yield tuple((-i for i in t))
        return
    if p < 1 or k < 1:
        raise ValueError(filldedent('\n    Expecting positive integers for `(p, k)`, but got `(%s, %s)`' % (p, k)))
    if n == 0:
        if zeros:
            yield ((0,) * k)
        return
    if k == 1:
        if p == 1:
            yield (n,)
        else:
            be = perfect_power(n)
            if be:
                (b, e) = be
                (d, r) = divmod(e, p)
                if not r:
                    yield (b ** d,)
        return
    if p == 1:
        for t in partition(n, k, zeros=zeros):
            yield t
        return
    if p == 2:
        feasible = _can_do_sum_of_squares(n, k)
        if not feasible:
            return
        if not zeros and n > 33 and (k >= 5) and (k <= n) and (n - k in (13, 10, 7, 5, 4, 2, 1)):
            'Todd G. Will, "When Is n^2 a Sum of k Squares?", [online].\n                Available: https://www.maa.org/sites/default/files/Will-MMz-201037918.pdf'
            return
        if feasible is not True:
            yield prime_as_sum_of_two_squares(n)
            return
    if k == 2 and p > 2:
        be = perfect_power(n)
        if be and be[1] % p == 0:
            return
    if n >= k:
        a = integer_nthroot(n - (k - 1), p)[0]
        for t in pow_rep_recursive(a, k, n, [], p):
            yield tuple(reversed(t))
    if zeros:
        a = integer_nthroot(n, p)[0]
        for i in range(1, k):
            for t in pow_rep_recursive(a, i, n, [], p):
                yield tuple(reversed(t + (0,) * (k - i)))
sum_of_powers = power_representation

def pow_rep_recursive(n_i, k, n_remaining, terms, p):
    if False:
        i = 10
        return i + 15
    if n_i <= 0 or k <= 0:
        return
    if n_remaining < k:
        return
    if k * pow(n_i, p) < n_remaining:
        return
    if k == 0 and n_remaining == 0:
        yield tuple(terms)
    elif k == 1:
        (next_term, exact) = integer_nthroot(n_remaining, p)
        if exact and next_term <= n_i:
            yield tuple(terms + [next_term])
        return
    elif n_i >= 1 and k > 0:
        for next_term in range(1, n_i + 1):
            residual = n_remaining - pow(next_term, p)
            if residual < 0:
                break
            yield from pow_rep_recursive(next_term, k - 1, residual, terms + [next_term], p)

def sum_of_squares(n, k, zeros=False):
    if False:
        for i in range(10):
            print('nop')
    'Return a generator that yields the k-tuples of nonnegative\n    values, the squares of which sum to n. If zeros is False (default)\n    then the solution will not contain zeros. The nonnegative\n    elements of a tuple are sorted.\n\n    * If k == 1 and n is square, (n,) is returned.\n\n    * If k == 2 then n can only be written as a sum of squares if\n      every prime in the factorization of n that has the form\n      4*k + 3 has an even multiplicity. If n is prime then\n      it can only be written as a sum of two squares if it is\n      in the form 4*k + 1.\n\n    * if k == 3 then n can be written as a sum of squares if it does\n      not have the form 4**m*(8*k + 7).\n\n    * all integers can be written as the sum of 4 squares.\n\n    * if k > 4 then n can be partitioned and each partition can\n      be written as a sum of 4 squares; if n is not evenly divisible\n      by 4 then n can be written as a sum of squares only if the\n      an additional partition can be written as sum of squares.\n      For example, if k = 6 then n is partitioned into two parts,\n      the first being written as a sum of 4 squares and the second\n      being written as a sum of 2 squares -- which can only be\n      done if the condition above for k = 2 can be met, so this will\n      automatically reject certain partitions of n.\n\n    Examples\n    ========\n\n    >>> from sympy.solvers.diophantine.diophantine import sum_of_squares\n    >>> list(sum_of_squares(25, 2))\n    [(3, 4)]\n    >>> list(sum_of_squares(25, 2, True))\n    [(3, 4), (0, 5)]\n    >>> list(sum_of_squares(25, 4))\n    [(1, 2, 2, 4)]\n\n    See Also\n    ========\n\n    sympy.utilities.iterables.signed_permutations\n    '
    yield from power_representation(n, 2, k, zeros)

def _can_do_sum_of_squares(n, k):
    if False:
        print('Hello World!')
    'Return True if n can be written as the sum of k squares,\n    False if it cannot, or 1 if ``k == 2`` and ``n`` is prime (in which\n    case it *can* be written as a sum of two squares). A False\n    is returned only if it cannot be written as ``k``-squares, even\n    if 0s are allowed.\n    '
    if k < 1:
        return False
    if n < 0:
        return False
    if n == 0:
        return True
    if k == 1:
        return is_square(n)
    if k == 2:
        if n in (1, 2):
            return True
        if isprime(n):
            if n % 4 == 1:
                return 1
            return False
        else:
            f = factorint(n)
            for (p, m) in f.items():
                if p % 4 == 3 and m % 2:
                    return False
            return True
    if k == 3:
        if n // 4 ** multiplicity(4, n) % 8 == 7:
            return False
    return True