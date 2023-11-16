"""Solvers of systems of polynomial equations. """
import itertools
from sympy.core import S
from sympy.core.sorting import default_sort_key
from sympy.polys import Poly, groebner, roots
from sympy.polys.polytools import parallel_poly_from_expr
from sympy.polys.polyerrors import ComputationFailed, PolificationFailed, CoercionFailed
from sympy.simplify import rcollect
from sympy.utilities import postfixes
from sympy.utilities.misc import filldedent

class SolveFailed(Exception):
    """Raised when solver's conditions were not met. """

def solve_poly_system(seq, *gens, strict=False, **args):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return a list of solutions for the system of polynomial equations\n    or else None.\n\n    Parameters\n    ==========\n\n    seq: a list/tuple/set\n        Listing all the equations that are needed to be solved\n    gens: generators\n        generators of the equations in seq for which we want the\n        solutions\n    strict: a boolean (default is False)\n        if strict is True, NotImplementedError will be raised if\n        the solution is known to be incomplete (which can occur if\n        not all solutions are expressible in radicals)\n    args: Keyword arguments\n        Special options for solving the equations.\n\n\n    Returns\n    =======\n\n    List[Tuple]\n        a list of tuples with elements being solutions for the\n        symbols in the order they were passed as gens\n    None\n        None is returned when the computed basis contains only the ground.\n\n    Examples\n    ========\n\n    >>> from sympy import solve_poly_system\n    >>> from sympy.abc import x, y\n\n    >>> solve_poly_system([x*y - 2*y, 2*y**2 - x**2], x, y)\n    [(0, 0), (2, -sqrt(2)), (2, sqrt(2))]\n\n    >>> solve_poly_system([x**5 - x + y**3, y**2 - 1], x, y, strict=True)\n    Traceback (most recent call last):\n    ...\n    UnsolvableFactorError\n\n    '
    try:
        (polys, opt) = parallel_poly_from_expr(seq, *gens, **args)
    except PolificationFailed as exc:
        raise ComputationFailed('solve_poly_system', len(seq), exc)
    if len(polys) == len(opt.gens) == 2:
        (f, g) = polys
        if all((i <= 2 for i in f.degree_list() + g.degree_list())):
            try:
                return solve_biquadratic(f, g, opt)
            except SolveFailed:
                pass
    return solve_generic(polys, opt, strict=strict)

def solve_biquadratic(f, g, opt):
    if False:
        i = 10
        return i + 15
    "Solve a system of two bivariate quadratic polynomial equations.\n\n    Parameters\n    ==========\n\n    f: a single Expr or Poly\n        First equation\n    g: a single Expr or Poly\n        Second Equation\n    opt: an Options object\n        For specifying keyword arguments and generators\n\n    Returns\n    =======\n\n    List[Tuple]\n        a list of tuples with elements being solutions for the\n        symbols in the order they were passed as gens\n    None\n        None is returned when the computed basis contains only the ground.\n\n    Examples\n    ========\n\n    >>> from sympy import Options, Poly\n    >>> from sympy.abc import x, y\n    >>> from sympy.solvers.polysys import solve_biquadratic\n    >>> NewOption = Options((x, y), {'domain': 'ZZ'})\n\n    >>> a = Poly(y**2 - 4 + x, y, x, domain='ZZ')\n    >>> b = Poly(y*2 + 3*x - 7, y, x, domain='ZZ')\n    >>> solve_biquadratic(a, b, NewOption)\n    [(1/3, 3), (41/27, 11/9)]\n\n    >>> a = Poly(y + x**2 - 3, y, x, domain='ZZ')\n    >>> b = Poly(-y + x - 4, y, x, domain='ZZ')\n    >>> solve_biquadratic(a, b, NewOption)\n    [(7/2 - sqrt(29)/2, -sqrt(29)/2 - 1/2), (sqrt(29)/2 + 7/2, -1/2 +       sqrt(29)/2)]\n    "
    G = groebner([f, g])
    if len(G) == 1 and G[0].is_ground:
        return None
    if len(G) != 2:
        raise SolveFailed
    (x, y) = opt.gens
    (p, q) = G
    if not p.gcd(q).is_ground:
        raise SolveFailed
    p = Poly(p, x, expand=False)
    p_roots = [rcollect(expr, y) for expr in roots(p).keys()]
    q = q.ltrim(-1)
    q_roots = list(roots(q).keys())
    solutions = [(p_root.subs(y, q_root), q_root) for (q_root, p_root) in itertools.product(q_roots, p_roots)]
    return sorted(solutions, key=default_sort_key)

def solve_generic(polys, opt, strict=False):
    if False:
        print('Hello World!')
    "\n    Solve a generic system of polynomial equations.\n\n    Returns all possible solutions over C[x_1, x_2, ..., x_m] of a\n    set F = { f_1, f_2, ..., f_n } of polynomial equations, using\n    Groebner basis approach. For now only zero-dimensional systems\n    are supported, which means F can have at most a finite number\n    of solutions. If the basis contains only the ground, None is\n    returned.\n\n    The algorithm works by the fact that, supposing G is the basis\n    of F with respect to an elimination order (here lexicographic\n    order is used), G and F generate the same ideal, they have the\n    same set of solutions. By the elimination property, if G is a\n    reduced, zero-dimensional Groebner basis, then there exists an\n    univariate polynomial in G (in its last variable). This can be\n    solved by computing its roots. Substituting all computed roots\n    for the last (eliminated) variable in other elements of G, new\n    polynomial system is generated. Applying the above procedure\n    recursively, a finite number of solutions can be found.\n\n    The ability of finding all solutions by this procedure depends\n    on the root finding algorithms. If no solutions were found, it\n    means only that roots() failed, but the system is solvable. To\n    overcome this difficulty use numerical algorithms instead.\n\n    Parameters\n    ==========\n\n    polys: a list/tuple/set\n        Listing all the polynomial equations that are needed to be solved\n    opt: an Options object\n        For specifying keyword arguments and generators\n    strict: a boolean\n        If strict is True, NotImplementedError will be raised if the solution\n        is known to be incomplete\n\n    Returns\n    =======\n\n    List[Tuple]\n        a list of tuples with elements being solutions for the\n        symbols in the order they were passed as gens\n    None\n        None is returned when the computed basis contains only the ground.\n\n    References\n    ==========\n\n    .. [Buchberger01] B. Buchberger, Groebner Bases: A Short\n    Introduction for Systems Theorists, In: R. Moreno-Diaz,\n    B. Buchberger, J.L. Freire, Proceedings of EUROCAST'01,\n    February, 2001\n\n    .. [Cox97] D. Cox, J. Little, D. O'Shea, Ideals, Varieties\n    and Algorithms, Springer, Second Edition, 1997, pp. 112\n\n    Raises\n    ========\n\n    NotImplementedError\n        If the system is not zero-dimensional (does not have a finite\n        number of solutions)\n\n    UnsolvableFactorError\n        If ``strict`` is True and not all solution components are\n        expressible in radicals\n\n    Examples\n    ========\n\n    >>> from sympy import Poly, Options\n    >>> from sympy.solvers.polysys import solve_generic\n    >>> from sympy.abc import x, y\n    >>> NewOption = Options((x, y), {'domain': 'ZZ'})\n\n    >>> a = Poly(x - y + 5, x, y, domain='ZZ')\n    >>> b = Poly(x + y - 3, x, y, domain='ZZ')\n    >>> solve_generic([a, b], NewOption)\n    [(-1, 4)]\n\n    >>> a = Poly(x - 2*y + 5, x, y, domain='ZZ')\n    >>> b = Poly(2*x - y - 3, x, y, domain='ZZ')\n    >>> solve_generic([a, b], NewOption)\n    [(11/3, 13/3)]\n\n    >>> a = Poly(x**2 + y, x, y, domain='ZZ')\n    >>> b = Poly(x + y*4, x, y, domain='ZZ')\n    >>> solve_generic([a, b], NewOption)\n    [(0, 0), (1/4, -1/16)]\n\n    >>> a = Poly(x**5 - x + y**3, x, y, domain='ZZ')\n    >>> b = Poly(y**2 - 1, x, y, domain='ZZ')\n    >>> solve_generic([a, b], NewOption, strict=True)\n    Traceback (most recent call last):\n    ...\n    UnsolvableFactorError\n\n    "

    def _is_univariate(f):
        if False:
            for i in range(10):
                print('nop')
        "Returns True if 'f' is univariate in its last variable. "
        for monom in f.monoms():
            if any(monom[:-1]):
                return False
        return True

    def _subs_root(f, gen, zero):
        if False:
            for i in range(10):
                print('nop')
        'Replace generator with a root so that the result is nice. '
        p = f.as_expr({gen: zero})
        if f.degree(gen) >= 2:
            p = p.expand(deep=False)
        return p

    def _solve_reduced_system(system, gens, entry=False):
        if False:
            while True:
                i = 10
        'Recursively solves reduced polynomial systems. '
        if len(system) == len(gens) == 1:
            zeros = list(roots(system[0], gens[-1], strict=strict).keys())
            return [(zero,) for zero in zeros]
        basis = groebner(system, gens, polys=True)
        if len(basis) == 1 and basis[0].is_ground:
            if not entry:
                return []
            else:
                return None
        univariate = list(filter(_is_univariate, basis))
        if len(basis) < len(gens):
            raise NotImplementedError(filldedent('\n                only zero-dimensional systems supported\n                (finite number of solutions)\n                '))
        if len(univariate) == 1:
            f = univariate.pop()
        else:
            raise NotImplementedError(filldedent('\n                only zero-dimensional systems supported\n                (finite number of solutions)\n                '))
        gens = f.gens
        gen = gens[-1]
        zeros = list(roots(f.ltrim(gen), strict=strict).keys())
        if not zeros:
            return []
        if len(basis) == 1:
            return [(zero,) for zero in zeros]
        solutions = []
        for zero in zeros:
            new_system = []
            new_gens = gens[:-1]
            for b in basis[:-1]:
                eq = _subs_root(b, gen, zero)
                if eq is not S.Zero:
                    new_system.append(eq)
            for solution in _solve_reduced_system(new_system, new_gens):
                solutions.append(solution + (zero,))
        if solutions and len(solutions[0]) != len(gens):
            raise NotImplementedError(filldedent('\n                only zero-dimensional systems supported\n                (finite number of solutions)\n                '))
        return solutions
    try:
        result = _solve_reduced_system(polys, opt.gens, entry=True)
    except CoercionFailed:
        raise NotImplementedError
    if result is not None:
        return sorted(result, key=default_sort_key)

def solve_triangulated(polys, *gens, **args):
    if False:
        print('Hello World!')
    '\n    Solve a polynomial system using Gianni-Kalkbrenner algorithm.\n\n    The algorithm proceeds by computing one Groebner basis in the ground\n    domain and then by iteratively computing polynomial factorizations in\n    appropriately constructed algebraic extensions of the ground domain.\n\n    Parameters\n    ==========\n\n    polys: a list/tuple/set\n        Listing all the equations that are needed to be solved\n    gens: generators\n        generators of the equations in polys for which we want the\n        solutions\n    args: Keyword arguments\n        Special options for solving the equations\n\n    Returns\n    =======\n\n    List[Tuple]\n        A List of tuples. Solutions for symbols that satisfy the\n        equations listed in polys\n\n    Examples\n    ========\n\n    >>> from sympy import solve_triangulated\n    >>> from sympy.abc import x, y, z\n\n    >>> F = [x**2 + y + z - 1, x + y**2 + z - 1, x + y + z**2 - 1]\n\n    >>> solve_triangulated(F, x, y, z)\n    [(0, 0, 1), (0, 1, 0), (1, 0, 0)]\n\n    References\n    ==========\n\n    1. Patrizia Gianni, Teo Mora, Algebraic Solution of System of\n    Polynomial Equations using Groebner Bases, AAECC-5 on Applied Algebra,\n    Algebraic Algorithms and Error-Correcting Codes, LNCS 356 247--257, 1989\n\n    '
    G = groebner(polys, gens, polys=True)
    G = list(reversed(G))
    domain = args.get('domain')
    if domain is not None:
        for (i, g) in enumerate(G):
            G[i] = g.set_domain(domain)
    (f, G) = (G[0].ltrim(-1), G[1:])
    dom = f.get_domain()
    zeros = f.ground_roots()
    solutions = set()
    for zero in zeros:
        solutions.add(((zero,), dom))
    var_seq = reversed(gens[:-1])
    vars_seq = postfixes(gens[1:])
    for (var, vars) in zip(var_seq, vars_seq):
        _solutions = set()
        for (values, dom) in solutions:
            (H, mapping) = ([], list(zip(vars, values)))
            for g in G:
                _vars = (var,) + vars
                if g.has_only_gens(*_vars) and g.degree(var) != 0:
                    h = g.ltrim(var).eval(dict(mapping))
                    if g.degree(var) == h.degree():
                        H.append(h)
            p = min(H, key=lambda h: h.degree())
            zeros = p.ground_roots()
            for zero in zeros:
                if not zero.is_Rational:
                    dom_zero = dom.algebraic_field(zero)
                else:
                    dom_zero = dom
                _solutions.add(((zero,) + values, dom_zero))
        solutions = _solutions
    solutions = list(solutions)
    for (i, (solution, _)) in enumerate(solutions):
        solutions[i] = solution
    return sorted(solutions, key=default_sort_key)