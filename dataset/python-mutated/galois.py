"""
Construct transitive subgroups of symmetric groups, useful in Galois theory.

Besides constructing instances of the :py:class:`~.PermutationGroup` class to
represent the transitive subgroups of $S_n$ for small $n$, this module provides
*names* for these groups.

In some applications, it may be preferable to know the name of a group,
rather than receive an instance of the :py:class:`~.PermutationGroup`
class, and then have to do extra work to determine which group it is, by
checking various properties.

Names are instances of ``Enum`` classes defined in this module. With a name in
hand, the name's ``get_perm_group`` method can then be used to retrieve a
:py:class:`~.PermutationGroup`.

The names used for groups in this module are taken from [1].

References
==========

.. [1] Cohen, H. *A Course in Computational Algebraic Number Theory*.

"""
from collections import defaultdict
from enum import Enum
import itertools
from sympy.combinatorics.named_groups import SymmetricGroup, AlternatingGroup, CyclicGroup, DihedralGroup, set_symmetric_group_properties, set_alternating_group_properties
from sympy.combinatorics.perm_groups import PermutationGroup
from sympy.combinatorics.permutations import Permutation

class S1TransitiveSubgroups(Enum):
    """
    Names for the transitive subgroups of S1.
    """
    S1 = 'S1'

    def get_perm_group(self):
        if False:
            while True:
                i = 10
        return SymmetricGroup(1)

class S2TransitiveSubgroups(Enum):
    """
    Names for the transitive subgroups of S2.
    """
    S2 = 'S2'

    def get_perm_group(self):
        if False:
            return 10
        return SymmetricGroup(2)

class S3TransitiveSubgroups(Enum):
    """
    Names for the transitive subgroups of S3.
    """
    A3 = 'A3'
    S3 = 'S3'

    def get_perm_group(self):
        if False:
            while True:
                i = 10
        if self == S3TransitiveSubgroups.A3:
            return AlternatingGroup(3)
        elif self == S3TransitiveSubgroups.S3:
            return SymmetricGroup(3)

class S4TransitiveSubgroups(Enum):
    """
    Names for the transitive subgroups of S4.
    """
    C4 = 'C4'
    V = 'V'
    D4 = 'D4'
    A4 = 'A4'
    S4 = 'S4'

    def get_perm_group(self):
        if False:
            i = 10
            return i + 15
        if self == S4TransitiveSubgroups.C4:
            return CyclicGroup(4)
        elif self == S4TransitiveSubgroups.V:
            return four_group()
        elif self == S4TransitiveSubgroups.D4:
            return DihedralGroup(4)
        elif self == S4TransitiveSubgroups.A4:
            return AlternatingGroup(4)
        elif self == S4TransitiveSubgroups.S4:
            return SymmetricGroup(4)

class S5TransitiveSubgroups(Enum):
    """
    Names for the transitive subgroups of S5.
    """
    C5 = 'C5'
    D5 = 'D5'
    M20 = 'M20'
    A5 = 'A5'
    S5 = 'S5'

    def get_perm_group(self):
        if False:
            for i in range(10):
                print('nop')
        if self == S5TransitiveSubgroups.C5:
            return CyclicGroup(5)
        elif self == S5TransitiveSubgroups.D5:
            return DihedralGroup(5)
        elif self == S5TransitiveSubgroups.M20:
            return M20()
        elif self == S5TransitiveSubgroups.A5:
            return AlternatingGroup(5)
        elif self == S5TransitiveSubgroups.S5:
            return SymmetricGroup(5)

class S6TransitiveSubgroups(Enum):
    """
    Names for the transitive subgroups of S6.
    """
    C6 = 'C6'
    S3 = 'S3'
    D6 = 'D6'
    A4 = 'A4'
    G18 = 'G18'
    A4xC2 = 'A4 x C2'
    S4m = 'S4-'
    S4p = 'S4+'
    G36m = 'G36-'
    G36p = 'G36+'
    S4xC2 = 'S4 x C2'
    PSL2F5 = 'PSL2(F5)'
    G72 = 'G72'
    PGL2F5 = 'PGL2(F5)'
    A6 = 'A6'
    S6 = 'S6'

    def get_perm_group(self):
        if False:
            return 10
        if self == S6TransitiveSubgroups.C6:
            return CyclicGroup(6)
        elif self == S6TransitiveSubgroups.S3:
            return S3_in_S6()
        elif self == S6TransitiveSubgroups.D6:
            return DihedralGroup(6)
        elif self == S6TransitiveSubgroups.A4:
            return A4_in_S6()
        elif self == S6TransitiveSubgroups.G18:
            return G18()
        elif self == S6TransitiveSubgroups.A4xC2:
            return A4xC2()
        elif self == S6TransitiveSubgroups.S4m:
            return S4m()
        elif self == S6TransitiveSubgroups.S4p:
            return S4p()
        elif self == S6TransitiveSubgroups.G36m:
            return G36m()
        elif self == S6TransitiveSubgroups.G36p:
            return G36p()
        elif self == S6TransitiveSubgroups.S4xC2:
            return S4xC2()
        elif self == S6TransitiveSubgroups.PSL2F5:
            return PSL2F5()
        elif self == S6TransitiveSubgroups.G72:
            return G72()
        elif self == S6TransitiveSubgroups.PGL2F5:
            return PGL2F5()
        elif self == S6TransitiveSubgroups.A6:
            return AlternatingGroup(6)
        elif self == S6TransitiveSubgroups.S6:
            return SymmetricGroup(6)

def four_group():
    if False:
        print('Hello World!')
    '\n    Return a representation of the Klein four-group as a transitive subgroup\n    of S4.\n    '
    return PermutationGroup(Permutation(0, 1)(2, 3), Permutation(0, 2)(1, 3))

def M20():
    if False:
        return 10
    '\n    Return a representation of the metacyclic group M20, a transitive subgroup\n    of S5 that is one of the possible Galois groups for polys of degree 5.\n\n    Notes\n    =====\n\n    See [1], Page 323.\n\n    '
    G = PermutationGroup(Permutation(0, 1, 2, 3, 4), Permutation(1, 2, 4, 3))
    G._degree = 5
    G._order = 20
    G._is_transitive = True
    G._is_sym = False
    G._is_alt = False
    G._is_cyclic = False
    G._is_dihedral = False
    return G

def S3_in_S6():
    if False:
        i = 10
        return i + 15
    '\n    Return a representation of S3 as a transitive subgroup of S6.\n\n    Notes\n    =====\n\n    The representation is found by viewing the group as the symmetries of a\n    triangular prism.\n\n    '
    G = PermutationGroup(Permutation(0, 1, 2)(3, 4, 5), Permutation(0, 3)(2, 4)(1, 5))
    set_symmetric_group_properties(G, 3, 6)
    return G

def A4_in_S6():
    if False:
        i = 10
        return i + 15
    '\n    Return a representation of A4 as a transitive subgroup of S6.\n\n    Notes\n    =====\n\n    This was computed using :py:func:`~.find_transitive_subgroups_of_S6`.\n\n    '
    G = PermutationGroup(Permutation(0, 4, 5)(1, 3, 2), Permutation(0, 1, 2)(3, 5, 4))
    set_alternating_group_properties(G, 4, 6)
    return G

def S4m():
    if False:
        print('Hello World!')
    '\n    Return a representation of the S4- transitive subgroup of S6.\n\n    Notes\n    =====\n\n    This was computed using :py:func:`~.find_transitive_subgroups_of_S6`.\n\n    '
    G = PermutationGroup(Permutation(1, 4, 5, 3), Permutation(0, 4)(1, 5)(2, 3))
    set_symmetric_group_properties(G, 4, 6)
    return G

def S4p():
    if False:
        print('Hello World!')
    '\n    Return a representation of the S4+ transitive subgroup of S6.\n\n    Notes\n    =====\n\n    This was computed using :py:func:`~.find_transitive_subgroups_of_S6`.\n\n    '
    G = PermutationGroup(Permutation(0, 2, 4, 1)(3, 5), Permutation(0, 3)(4, 5))
    set_symmetric_group_properties(G, 4, 6)
    return G

def A4xC2():
    if False:
        i = 10
        return i + 15
    '\n    Return a representation of the (A4 x C2) transitive subgroup of S6.\n\n    Notes\n    =====\n\n    This was computed using :py:func:`~.find_transitive_subgroups_of_S6`.\n\n    '
    return PermutationGroup(Permutation(0, 4, 5)(1, 3, 2), Permutation(0, 1, 2)(3, 5, 4), Permutation(5)(2, 4))

def S4xC2():
    if False:
        while True:
            i = 10
    '\n    Return a representation of the (S4 x C2) transitive subgroup of S6.\n\n    Notes\n    =====\n\n    This was computed using :py:func:`~.find_transitive_subgroups_of_S6`.\n\n    '
    return PermutationGroup(Permutation(1, 4, 5, 3), Permutation(0, 4)(1, 5)(2, 3), Permutation(1, 4)(3, 5))

def G18():
    if False:
        i = 10
        return i + 15
    '\n    Return a representation of the group G18, a transitive subgroup of S6\n    isomorphic to the semidirect product of C3^2 with C2.\n\n    Notes\n    =====\n\n    This was computed using :py:func:`~.find_transitive_subgroups_of_S6`.\n\n    '
    return PermutationGroup(Permutation(5)(0, 1, 2), Permutation(3, 4, 5), Permutation(0, 4)(1, 5)(2, 3))

def G36m():
    if False:
        while True:
            i = 10
    '\n    Return a representation of the group G36-, a transitive subgroup of S6\n    isomorphic to the semidirect product of C3^2 with C2^2.\n\n    Notes\n    =====\n\n    This was computed using :py:func:`~.find_transitive_subgroups_of_S6`.\n\n    '
    return PermutationGroup(Permutation(5)(0, 1, 2), Permutation(3, 4, 5), Permutation(1, 2)(3, 5), Permutation(0, 4)(1, 5)(2, 3))

def G36p():
    if False:
        for i in range(10):
            print('nop')
    '\n    Return a representation of the group G36+, a transitive subgroup of S6\n    isomorphic to the semidirect product of C3^2 with C4.\n\n    Notes\n    =====\n\n    This was computed using :py:func:`~.find_transitive_subgroups_of_S6`.\n\n    '
    return PermutationGroup(Permutation(5)(0, 1, 2), Permutation(3, 4, 5), Permutation(0, 5, 2, 3)(1, 4))

def G72():
    if False:
        for i in range(10):
            print('nop')
    '\n    Return a representation of the group G72, a transitive subgroup of S6\n    isomorphic to the semidirect product of C3^2 with D4.\n\n    Notes\n    =====\n\n    See [1], Page 325.\n\n    '
    return PermutationGroup(Permutation(5)(0, 1, 2), Permutation(0, 4, 1, 3)(2, 5), Permutation(0, 3)(1, 4)(2, 5))

def PSL2F5():
    if False:
        for i in range(10):
            print('nop')
    '\n    Return a representation of the group $PSL_2(\\mathbb{F}_5)$, as a transitive\n    subgroup of S6, isomorphic to $A_5$.\n\n    Notes\n    =====\n\n    This was computed using :py:func:`~.find_transitive_subgroups_of_S6`.\n\n    '
    G = PermutationGroup(Permutation(0, 4, 5)(1, 3, 2), Permutation(0, 4, 3, 1, 5))
    set_alternating_group_properties(G, 5, 6)
    return G

def PGL2F5():
    if False:
        return 10
    '\n    Return a representation of the group $PGL_2(\\mathbb{F}_5)$, as a transitive\n    subgroup of S6, isomorphic to $S_5$.\n\n    Notes\n    =====\n\n    See [1], Page 325.\n\n    '
    G = PermutationGroup(Permutation(0, 1, 2, 3, 4), Permutation(0, 5)(1, 2)(3, 4))
    set_symmetric_group_properties(G, 5, 6)
    return G

def find_transitive_subgroups_of_S6(*targets, print_report=False):
    if False:
        for i in range(10):
            print('nop')
    '\n    Search for certain transitive subgroups of $S_6$.\n\n    The symmetric group $S_6$ has 16 different transitive subgroups, up to\n    conjugacy. Some are more easily constructed than others. For example, the\n    dihedral group $D_6$ is immediately found, but it is not at all obvious how\n    to realize $S_4$ or $S_5$ *transitively* within $S_6$.\n\n    In some cases there are well-known constructions that can be used. For\n    example, $S_5$ is isomorphic to $PGL_2(\\mathbb{F}_5)$, which acts in a\n    natural way on the projective line $P^1(\\mathbb{F}_5)$, a set of order 6.\n\n    In absence of such special constructions however, we can simply search for\n    generators. For example, transitive instances of $A_4$ and $S_4$ can be\n    found within $S_6$ in this way.\n\n    Once we are engaged in such searches, it may then be easier (if less\n    elegant) to find even those groups like $S_5$ that do have special\n    constructions, by mere search.\n\n    This function locates generators for transitive instances in $S_6$ of the\n    following subgroups:\n\n    * $A_4$\n    * $S_4^-$ ($S_4$ not contained within $A_6$)\n    * $S_4^+$ ($S_4$ contained within $A_6$)\n    * $A_4 \\times C_2$\n    * $S_4 \\times C_2$\n    * $G_{18}   = C_3^2 \\rtimes C_2$\n    * $G_{36}^- = C_3^2 \\rtimes C_2^2$\n    * $G_{36}^+ = C_3^2 \\rtimes C_4$\n    * $G_{72}   = C_3^2 \\rtimes D_4$\n    * $A_5$\n    * $S_5$\n\n    Note: Each of these groups also has a dedicated function in this module\n    that returns the group immediately, using generators that were found by\n    this search procedure.\n\n    The search procedure serves as a record of how these generators were\n    found. Also, due to randomness in the generation of the elements of\n    permutation groups, it can be called again, in order to (probably) get\n    different generators for the same groups.\n\n    Parameters\n    ==========\n\n    targets : list of :py:class:`~.S6TransitiveSubgroups` values\n        The groups you want to find.\n\n    print_report : bool (default False)\n        If True, print to stdout the generators found for each group.\n\n    Returns\n    =======\n\n    dict\n        mapping each name in *targets* to the :py:class:`~.PermutationGroup`\n        that was found\n\n    References\n    ==========\n\n    .. [2] https://en.wikipedia.org/wiki/Projective_linear_group#Exceptional_isomorphisms\n    .. [3] https://en.wikipedia.org/wiki/Automorphisms_of_the_symmetric_and_alternating_groups#PGL%282,5%29\n\n    '

    def elts_by_order(G):
        if False:
            while True:
                i = 10
        'Sort the elements of a group by their order. '
        elts = defaultdict(list)
        for g in G.elements:
            elts[g.order()].append(g)
        return elts

    def order_profile(G, name=None):
        if False:
            return 10
        'Determine how many elements a group has, of each order. '
        elts = elts_by_order(G)
        profile = {o: len(e) for (o, e) in elts.items()}
        if name:
            print(f'{name}: ' + ' '.join((f'{len(profile[r])}@{r}' for r in sorted(profile.keys()))))
        return profile
    S6 = SymmetricGroup(6)
    A6 = AlternatingGroup(6)
    S6_by_order = elts_by_order(S6)

    def search(existing_gens, needed_gen_orders, order, alt=None, profile=None, anti_profile=None):
        if False:
            while True:
                i = 10
        "\n        Find a transitive subgroup of S6.\n\n        Parameters\n        ==========\n\n        existing_gens : list of Permutation\n            Optionally empty list of generators that must be in the group.\n\n        needed_gen_orders : list of positive int\n            Nonempty list of the orders of the additional generators that are\n            to be found.\n\n        order: int\n            The order of the group being sought.\n\n        alt: bool, None\n            If True, require the group to be contained in A6.\n            If False, require the group not to be contained in A6.\n\n        profile : dict\n            If given, the group's order profile must equal this.\n\n        anti_profile : dict\n            If given, the group's order profile must *not* equal this.\n\n        "
        for gens in itertools.product(*[S6_by_order[n] for n in needed_gen_orders]):
            if len(set(gens)) < len(gens):
                continue
            G = PermutationGroup(existing_gens + list(gens))
            if G.order() == order and G.is_transitive():
                if alt is not None and G.is_subgroup(A6) != alt:
                    continue
                if profile and order_profile(G) != profile:
                    continue
                if anti_profile and order_profile(G) == anti_profile:
                    continue
                return G

    def match_known_group(G, alt=None):
        if False:
            for i in range(10):
                print('nop')
        needed = [g.order() for g in G.generators]
        return search([], needed, G.order(), alt=alt, profile=order_profile(G))
    found = {}

    def finish_up(name, G):
        if False:
            for i in range(10):
                print('nop')
        found[name] = G
        if print_report:
            print('=' * 40)
            print(f'{name}:')
            print(G.generators)
    if S6TransitiveSubgroups.A4 in targets or S6TransitiveSubgroups.A4xC2 in targets:
        A4_in_S6 = match_known_group(AlternatingGroup(4))
        finish_up(S6TransitiveSubgroups.A4, A4_in_S6)
    if S6TransitiveSubgroups.S4m in targets or S6TransitiveSubgroups.S4xC2 in targets:
        S4m_in_S6 = match_known_group(SymmetricGroup(4), alt=False)
        finish_up(S6TransitiveSubgroups.S4m, S4m_in_S6)
    if S6TransitiveSubgroups.S4p in targets:
        S4p_in_S6 = match_known_group(SymmetricGroup(4), alt=True)
        finish_up(S6TransitiveSubgroups.S4p, S4p_in_S6)
    if S6TransitiveSubgroups.A4xC2 in targets:
        A4xC2_in_S6 = search(A4_in_S6.generators, [2], 24, anti_profile=order_profile(SymmetricGroup(4)))
        finish_up(S6TransitiveSubgroups.A4xC2, A4xC2_in_S6)
    if S6TransitiveSubgroups.S4xC2 in targets:
        S4xC2_in_S6 = search(S4m_in_S6.generators, [2], 48)
        finish_up(S6TransitiveSubgroups.S4xC2, S4xC2_in_S6)
    N_gens = [Permutation(5)(0, 1, 2), Permutation(5)(3, 4, 5)]
    if S6TransitiveSubgroups.G18 in targets:
        G18_in_S6 = search(N_gens, [2], 18)
        finish_up(S6TransitiveSubgroups.G18, G18_in_S6)
    if S6TransitiveSubgroups.G36m in targets:
        G36m_in_S6 = search(N_gens, [2, 2], 36, alt=False)
        finish_up(S6TransitiveSubgroups.G36m, G36m_in_S6)
    if S6TransitiveSubgroups.G36p in targets:
        G36p_in_S6 = search(N_gens, [4], 36, alt=True)
        finish_up(S6TransitiveSubgroups.G36p, G36p_in_S6)
    if S6TransitiveSubgroups.G72 in targets:
        G72_in_S6 = search(N_gens, [4, 2], 72)
        finish_up(S6TransitiveSubgroups.G72, G72_in_S6)
    if S6TransitiveSubgroups.PSL2F5 in targets:
        PSL2F5_in_S6 = match_known_group(AlternatingGroup(5))
        finish_up(S6TransitiveSubgroups.PSL2F5, PSL2F5_in_S6)
    if S6TransitiveSubgroups.PGL2F5 in targets:
        PGL2F5_in_S6 = match_known_group(SymmetricGroup(5))
        finish_up(S6TransitiveSubgroups.PGL2F5, PGL2F5_in_S6)
    if S6TransitiveSubgroups.C6 in targets:
        C6 = match_known_group(CyclicGroup(6))
        finish_up(S6TransitiveSubgroups.C6, C6)
    if S6TransitiveSubgroups.S3 in targets:
        S3 = match_known_group(SymmetricGroup(3))
        finish_up(S6TransitiveSubgroups.S3, S3)
    if S6TransitiveSubgroups.D6 in targets:
        D6 = match_known_group(DihedralGroup(6))
        finish_up(S6TransitiveSubgroups.D6, D6)
    if S6TransitiveSubgroups.A6 in targets:
        A6 = match_known_group(A6)
        finish_up(S6TransitiveSubgroups.A6, A6)
    if S6TransitiveSubgroups.S6 in targets:
        S6 = match_known_group(S6)
        finish_up(S6TransitiveSubgroups.S6, S6)
    return found