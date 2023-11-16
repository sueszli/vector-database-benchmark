"""
Compute Galois groups of polynomials.

We use algorithms from [1], with some modifications to use lookup tables for
resolvents.

References
==========

.. [1] Cohen, H. *A Course in Computational Algebraic Number Theory*.

"""
from collections import defaultdict
import random
from sympy.core.symbol import Dummy, symbols
from sympy.ntheory.primetest import is_square
from sympy.polys.domains import ZZ
from sympy.polys.densebasic import dup_random
from sympy.polys.densetools import dup_eval
from sympy.polys.euclidtools import dup_discriminant
from sympy.polys.factortools import dup_factor_list, dup_irreducible_p
from sympy.polys.numberfields.galois_resolvents import GaloisGroupException, get_resolvent_by_lookup, define_resolvents, Resolvent
from sympy.polys.numberfields.utilities import coeff_search
from sympy.polys.polytools import Poly, poly_from_expr, PolificationFailed, ComputationFailed
from sympy.polys.sqfreetools import dup_sqf_p
from sympy.utilities import public

class MaxTriesException(GaloisGroupException):
    ...

def tschirnhausen_transformation(T, max_coeff=10, max_tries=30, history=None, fixed_order=True):
    if False:
        print('Hello World!')
    "\n    Given a univariate, monic, irreducible polynomial over the integers, find\n    another such polynomial defining the same number field.\n\n    Explanation\n    ===========\n\n    See Alg 6.3.4 of [1].\n\n    Parameters\n    ==========\n\n    T : Poly\n        The given polynomial\n    max_coeff : int\n        When choosing a transformation as part of the process,\n        keep the coeffs between plus and minus this.\n    max_tries : int\n        Consider at most this many transformations.\n    history : set, None, optional (default=None)\n        Pass a set of ``Poly.rep``'s in order to prevent any of these\n        polynomials from being returned as the polynomial ``U`` i.e. the\n        transformation of the given polynomial *T*. The given poly *T* will\n        automatically be added to this set, before we try to find a new one.\n    fixed_order : bool, default True\n        If ``True``, work through candidate transformations A(x) in a fixed\n        order, from small coeffs to large, resulting in deterministic behavior.\n        If ``False``, the A(x) are chosen randomly, while still working our way\n        up from small coefficients to larger ones.\n\n    Returns\n    =======\n\n    Pair ``(A, U)``\n\n        ``A`` and ``U`` are ``Poly``, ``A`` is the\n        transformation, and ``U`` is the transformed polynomial that defines\n        the same number field as *T*. The polynomial ``A`` maps the roots of\n        *T* to the roots of ``U``.\n\n    Raises\n    ======\n\n    MaxTriesException\n        if could not find a polynomial before exceeding *max_tries*.\n\n    "
    X = Dummy('X')
    n = T.degree()
    if history is None:
        history = set()
    history.add(T.rep)
    if fixed_order:
        coeff_generators = {}
        deg_coeff_sum = 3
        current_degree = 2

    def get_coeff_generator(degree):
        if False:
            i = 10
            return i + 15
        gen = coeff_generators.get(degree, coeff_search(degree, 1))
        coeff_generators[degree] = gen
        return gen
    for i in range(max_tries):
        if fixed_order:
            gen = get_coeff_generator(current_degree)
            coeffs = next(gen)
            m = max((abs(c) for c in coeffs))
            if current_degree + m > deg_coeff_sum:
                if current_degree == 2:
                    deg_coeff_sum += 1
                    current_degree = deg_coeff_sum - 1
                else:
                    current_degree -= 1
                gen = get_coeff_generator(current_degree)
                coeffs = next(gen)
            a = [ZZ(1)] + [ZZ(c) for c in coeffs]
        else:
            C = min(i // 5 + 1, max_coeff)
            d = random.randint(2, n - 1)
            a = dup_random(d, -C, C, ZZ)
        A = Poly(a, T.gen)
        U = Poly(T.resultant(X - A), X)
        if U.rep not in history and dup_sqf_p(U.rep.to_list(), ZZ):
            return (A, U)
    raise MaxTriesException

def has_square_disc(T):
    if False:
        for i in range(10):
            print('nop')
    'Convenience to check if a Poly or dup has square discriminant. '
    d = T.discriminant() if isinstance(T, Poly) else dup_discriminant(T, ZZ)
    return is_square(d)

def _galois_group_degree_3(T, max_tries=30, randomize=False):
    if False:
        return 10
    '\n    Compute the Galois group of a polynomial of degree 3.\n\n    Explanation\n    ===========\n\n    Uses Prop 6.3.5 of [1].\n\n    '
    from sympy.combinatorics.galois import S3TransitiveSubgroups
    return (S3TransitiveSubgroups.A3, True) if has_square_disc(T) else (S3TransitiveSubgroups.S3, False)

def _galois_group_degree_4_root_approx(T, max_tries=30, randomize=False):
    if False:
        print('Hello World!')
    '\n    Compute the Galois group of a polynomial of degree 4.\n\n    Explanation\n    ===========\n\n    Follows Alg 6.3.7 of [1], using a pure root approximation approach.\n\n    '
    from sympy.combinatorics.permutations import Permutation
    from sympy.combinatorics.galois import S4TransitiveSubgroups
    X = symbols('X0 X1 X2 X3')
    F1 = X[0] * X[2] + X[1] * X[3]
    s1 = [Permutation(3), Permutation(3)(0, 1), Permutation(3)(0, 3)]
    R1 = Resolvent(F1, X, s1)
    F2_pre = X[0] * X[1] ** 2 + X[1] * X[2] ** 2 + X[2] * X[3] ** 2 + X[3] * X[0] ** 2
    s2_pre = [Permutation(3), Permutation(3)(0, 2)]
    history = set()
    for i in range(max_tries):
        if i > 0:
            (_, T) = tschirnhausen_transformation(T, max_tries=max_tries, history=history, fixed_order=not randomize)
        (R_dup, _, i0) = R1.eval_for_poly(T, find_integer_root=True)
        if not dup_sqf_p(R_dup, ZZ):
            continue
        sq_disc = has_square_disc(T)
        if i0 is None:
            return (S4TransitiveSubgroups.A4, True) if sq_disc else (S4TransitiveSubgroups.S4, False)
        if sq_disc:
            return (S4TransitiveSubgroups.V, True)
        sigma = s1[i0]
        F2 = F2_pre.subs(zip(X, sigma(X)), simultaneous=True)
        s2 = [sigma * tau * sigma for tau in s2_pre]
        R2 = Resolvent(F2, X, s2)
        (R_dup, _, _) = R2.eval_for_poly(T)
        d = dup_discriminant(R_dup, ZZ)
        if d == 0:
            continue
        if is_square(d):
            return (S4TransitiveSubgroups.C4, False)
        else:
            return (S4TransitiveSubgroups.D4, False)
    raise MaxTriesException

def _galois_group_degree_4_lookup(T, max_tries=30, randomize=False):
    if False:
        i = 10
        return i + 15
    '\n    Compute the Galois group of a polynomial of degree 4.\n\n    Explanation\n    ===========\n\n    Based on Alg 6.3.6 of [1], but uses resolvent coeff lookup.\n\n    '
    from sympy.combinatorics.galois import S4TransitiveSubgroups
    history = set()
    for i in range(max_tries):
        R_dup = get_resolvent_by_lookup(T, 0)
        if dup_sqf_p(R_dup, ZZ):
            break
        (_, T) = tschirnhausen_transformation(T, max_tries=max_tries, history=history, fixed_order=not randomize)
    else:
        raise MaxTriesException
    fl = dup_factor_list(R_dup, ZZ)
    L = sorted(sum([[len(r) - 1] * e for (r, e) in fl[1]], []))
    if L == [6]:
        return (S4TransitiveSubgroups.A4, True) if has_square_disc(T) else (S4TransitiveSubgroups.S4, False)
    if L == [1, 1, 4]:
        return (S4TransitiveSubgroups.C4, False)
    if L == [2, 2, 2]:
        return (S4TransitiveSubgroups.V, True)
    assert L == [2, 4]
    return (S4TransitiveSubgroups.D4, False)

def _galois_group_degree_5_hybrid(T, max_tries=30, randomize=False):
    if False:
        print('Hello World!')
    '\n    Compute the Galois group of a polynomial of degree 5.\n\n    Explanation\n    ===========\n\n    Based on Alg 6.3.9 of [1], but uses a hybrid approach, combining resolvent\n    coeff lookup, with root approximation.\n\n    '
    from sympy.combinatorics.galois import S5TransitiveSubgroups
    from sympy.combinatorics.permutations import Permutation
    X5 = symbols('X0,X1,X2,X3,X4')
    res = define_resolvents()
    (F51, _, s51) = res[5, 1]
    F51 = F51.as_expr(*X5)
    R51 = Resolvent(F51, X5, s51)
    history = set()
    reached_second_stage = False
    for i in range(max_tries):
        if i > 0:
            (_, T) = tschirnhausen_transformation(T, max_tries=max_tries, history=history, fixed_order=not randomize)
        R51_dup = get_resolvent_by_lookup(T, 1)
        if not dup_sqf_p(R51_dup, ZZ):
            continue
        if not reached_second_stage:
            sq_disc = has_square_disc(T)
            if dup_irreducible_p(R51_dup, ZZ):
                return (S5TransitiveSubgroups.A5, True) if sq_disc else (S5TransitiveSubgroups.S5, False)
            if not sq_disc:
                return (S5TransitiveSubgroups.M20, False)
        reached_second_stage = True
        rounded_roots = R51.round_roots_to_integers_for_poly(T)
        for (permutation_index, candidate_root) in rounded_roots.items():
            if not dup_eval(R51_dup, candidate_root, ZZ):
                break
        X = X5
        F2_pre = X[0] * X[1] ** 2 + X[1] * X[2] ** 2 + X[2] * X[3] ** 2 + X[3] * X[4] ** 2 + X[4] * X[0] ** 2
        s2_pre = [Permutation(4), Permutation(4)(0, 1)(2, 4)]
        i0 = permutation_index
        sigma = s51[i0]
        F2 = F2_pre.subs(zip(X, sigma(X)), simultaneous=True)
        s2 = [sigma * tau * sigma for tau in s2_pre]
        R2 = Resolvent(F2, X, s2)
        (R_dup, _, _) = R2.eval_for_poly(T)
        d = dup_discriminant(R_dup, ZZ)
        if d == 0:
            continue
        if is_square(d):
            return (S5TransitiveSubgroups.C5, True)
        else:
            return (S5TransitiveSubgroups.D5, True)
    raise MaxTriesException

def _galois_group_degree_5_lookup_ext_factor(T, max_tries=30, randomize=False):
    if False:
        for i in range(10):
            print('nop')
    '\n    Compute the Galois group of a polynomial of degree 5.\n\n    Explanation\n    ===========\n\n    Based on Alg 6.3.9 of [1], but uses resolvent coeff lookup, plus\n    factorization over an algebraic extension.\n\n    '
    from sympy.combinatorics.galois import S5TransitiveSubgroups
    _T = T
    history = set()
    for i in range(max_tries):
        R_dup = get_resolvent_by_lookup(T, 1)
        if dup_sqf_p(R_dup, ZZ):
            break
        (_, T) = tschirnhausen_transformation(T, max_tries=max_tries, history=history, fixed_order=not randomize)
    else:
        raise MaxTriesException
    sq_disc = has_square_disc(T)
    if dup_irreducible_p(R_dup, ZZ):
        return (S5TransitiveSubgroups.A5, True) if sq_disc else (S5TransitiveSubgroups.S5, False)
    if not sq_disc:
        return (S5TransitiveSubgroups.M20, False)
    fl = Poly(_T, domain=ZZ.alg_field_from_poly(_T)).factor_list()[1]
    if len(fl) == 5:
        return (S5TransitiveSubgroups.C5, True)
    else:
        return (S5TransitiveSubgroups.D5, True)

def _galois_group_degree_6_lookup(T, max_tries=30, randomize=False):
    if False:
        for i in range(10):
            print('nop')
    '\n    Compute the Galois group of a polynomial of degree 6.\n\n    Explanation\n    ===========\n\n    Based on Alg 6.3.10 of [1], but uses resolvent coeff lookup.\n\n    '
    from sympy.combinatorics.galois import S6TransitiveSubgroups
    history = set()
    for i in range(max_tries):
        R_dup = get_resolvent_by_lookup(T, 1)
        if dup_sqf_p(R_dup, ZZ):
            break
        (_, T) = tschirnhausen_transformation(T, max_tries=max_tries, history=history, fixed_order=not randomize)
    else:
        raise MaxTriesException
    fl = dup_factor_list(R_dup, ZZ)
    factors_by_deg = defaultdict(list)
    for (r, _) in fl[1]:
        factors_by_deg[len(r) - 1].append(r)
    L = sorted(sum([[d] * len(ff) for (d, ff) in factors_by_deg.items()], []))
    T_has_sq_disc = has_square_disc(T)
    if L == [1, 2, 3]:
        f1 = factors_by_deg[3][0]
        return (S6TransitiveSubgroups.C6, False) if has_square_disc(f1) else (S6TransitiveSubgroups.D6, False)
    elif L == [3, 3]:
        (f1, f2) = factors_by_deg[3]
        any_square = has_square_disc(f1) or has_square_disc(f2)
        return (S6TransitiveSubgroups.G18, False) if any_square else (S6TransitiveSubgroups.G36m, False)
    elif L == [2, 4]:
        if T_has_sq_disc:
            return (S6TransitiveSubgroups.S4p, True)
        else:
            f1 = factors_by_deg[4][0]
            return (S6TransitiveSubgroups.A4xC2, False) if has_square_disc(f1) else (S6TransitiveSubgroups.S4xC2, False)
    elif L == [1, 1, 4]:
        return (S6TransitiveSubgroups.A4, True) if T_has_sq_disc else (S6TransitiveSubgroups.S4m, False)
    elif L == [1, 5]:
        return (S6TransitiveSubgroups.PSL2F5, True) if T_has_sq_disc else (S6TransitiveSubgroups.PGL2F5, False)
    elif L == [1, 1, 1, 3]:
        return (S6TransitiveSubgroups.S3, False)
    assert L == [6]
    history = set()
    for i in range(max_tries):
        R_dup = get_resolvent_by_lookup(T, 2)
        if dup_sqf_p(R_dup, ZZ):
            break
        (_, T) = tschirnhausen_transformation(T, max_tries=max_tries, history=history, fixed_order=not randomize)
    else:
        raise MaxTriesException
    T_has_sq_disc = has_square_disc(T)
    if dup_irreducible_p(R_dup, ZZ):
        return (S6TransitiveSubgroups.A6, True) if T_has_sq_disc else (S6TransitiveSubgroups.S6, False)
    else:
        return (S6TransitiveSubgroups.G36p, True) if T_has_sq_disc else (S6TransitiveSubgroups.G72, False)

@public
def galois_group(f, *gens, by_name=False, max_tries=30, randomize=False, **args):
    if False:
        return 10
    "\n    Compute the Galois group for polynomials *f* up to degree 6.\n\n    Examples\n    ========\n\n    >>> from sympy import galois_group\n    >>> from sympy.abc import x\n    >>> f = x**4 + 1\n    >>> G, alt = galois_group(f)\n    >>> print(G)\n    PermutationGroup([\n    (0 1)(2 3),\n    (0 2)(1 3)])\n\n    The group is returned along with a boolean, indicating whether it is\n    contained in the alternating group $A_n$, where $n$ is the degree of *T*.\n    Along with other group properties, this can help determine which group it\n    is:\n\n    >>> alt\n    True\n    >>> G.order()\n    4\n\n    Alternatively, the group can be returned by name:\n\n    >>> G_name, _ = galois_group(f, by_name=True)\n    >>> print(G_name)\n    S4TransitiveSubgroups.V\n\n    The group itself can then be obtained by calling the name's\n    ``get_perm_group()`` method:\n\n    >>> G_name.get_perm_group()\n    PermutationGroup([\n    (0 1)(2 3),\n    (0 2)(1 3)])\n\n    Group names are values of the enum classes\n    :py:class:`sympy.combinatorics.galois.S1TransitiveSubgroups`,\n    :py:class:`sympy.combinatorics.galois.S2TransitiveSubgroups`,\n    etc.\n\n    Parameters\n    ==========\n\n    f : Expr\n        Irreducible polynomial over :ref:`ZZ` or :ref:`QQ`, whose Galois group\n        is to be determined.\n    gens : optional list of symbols\n        For converting *f* to Poly, and will be passed on to the\n        :py:func:`~.poly_from_expr` function.\n    by_name : bool, default False\n        If ``True``, the Galois group will be returned by name.\n        Otherwise it will be returned as a :py:class:`~.PermutationGroup`.\n    max_tries : int, default 30\n        Make at most this many attempts in those steps that involve\n        generating Tschirnhausen transformations.\n    randomize : bool, default False\n        If ``True``, then use random coefficients when generating Tschirnhausen\n        transformations. Otherwise try transformations in a fixed order. Both\n        approaches start with small coefficients and degrees and work upward.\n    args : optional\n        For converting *f* to Poly, and will be passed on to the\n        :py:func:`~.poly_from_expr` function.\n\n    Returns\n    =======\n\n    Pair ``(G, alt)``\n        The first element ``G`` indicates the Galois group. It is an instance\n        of one of the :py:class:`sympy.combinatorics.galois.S1TransitiveSubgroups`\n        :py:class:`sympy.combinatorics.galois.S2TransitiveSubgroups`, etc. enum\n        classes if *by_name* was ``True``, and a :py:class:`~.PermutationGroup`\n        if ``False``.\n\n        The second element is a boolean, saying whether the group is contained\n        in the alternating group $A_n$ ($n$ the degree of *T*).\n\n    Raises\n    ======\n\n    ValueError\n        if *f* is of an unsupported degree.\n\n    MaxTriesException\n        if could not complete before exceeding *max_tries* in those steps\n        that involve generating Tschirnhausen transformations.\n\n    See Also\n    ========\n\n    .Poly.galois_group\n\n    "
    gens = gens or []
    args = args or {}
    try:
        (F, opt) = poly_from_expr(f, *gens, **args)
    except PolificationFailed as exc:
        raise ComputationFailed('galois_group', 1, exc)
    return F.galois_group(by_name=by_name, max_tries=max_tries, randomize=randomize)