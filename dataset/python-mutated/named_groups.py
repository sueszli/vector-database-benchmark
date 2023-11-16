from sympy.combinatorics.group_constructs import DirectProduct
from sympy.combinatorics.perm_groups import PermutationGroup
from sympy.combinatorics.permutations import Permutation
_af_new = Permutation._af_new

def AbelianGroup(*cyclic_orders):
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns the direct product of cyclic groups with the given orders.\n\n    Explanation\n    ===========\n\n    According to the structure theorem for finite abelian groups ([1]),\n    every finite abelian group can be written as the direct product of\n    finitely many cyclic groups.\n\n    Examples\n    ========\n\n    >>> from sympy.combinatorics.named_groups import AbelianGroup\n    >>> AbelianGroup(3, 4)\n    PermutationGroup([\n            (6)(0 1 2),\n            (3 4 5 6)])\n    >>> _.is_group\n    True\n\n    See Also\n    ========\n\n    DirectProduct\n\n    References\n    ==========\n\n    .. [1] https://groupprops.subwiki.org/wiki/Structure_theorem_for_finitely_generated_abelian_groups\n\n    '
    groups = []
    degree = 0
    order = 1
    for size in cyclic_orders:
        degree += size
        order *= size
        groups.append(CyclicGroup(size))
    G = DirectProduct(*groups)
    G._is_abelian = True
    G._degree = degree
    G._order = order
    return G

def AlternatingGroup(n):
    if False:
        return 10
    '\n    Generates the alternating group on ``n`` elements as a permutation group.\n\n    Explanation\n    ===========\n\n    For ``n > 2``, the generators taken are ``(0 1 2), (0 1 2 ... n-1)`` for\n    ``n`` odd\n    and ``(0 1 2), (1 2 ... n-1)`` for ``n`` even (See [1], p.31, ex.6.9.).\n    After the group is generated, some of its basic properties are set.\n    The cases ``n = 1, 2`` are handled separately.\n\n    Examples\n    ========\n\n    >>> from sympy.combinatorics.named_groups import AlternatingGroup\n    >>> G = AlternatingGroup(4)\n    >>> G.is_group\n    True\n    >>> a = list(G.generate_dimino())\n    >>> len(a)\n    12\n    >>> all(perm.is_even for perm in a)\n    True\n\n    See Also\n    ========\n\n    SymmetricGroup, CyclicGroup, DihedralGroup\n\n    References\n    ==========\n\n    .. [1] Armstrong, M. "Groups and Symmetry"\n\n    '
    if n in (1, 2):
        return PermutationGroup([Permutation([0])])
    a = list(range(n))
    (a[0], a[1], a[2]) = (a[1], a[2], a[0])
    gen1 = a
    if n % 2:
        a = list(range(1, n))
        a.append(0)
        gen2 = a
    else:
        a = list(range(2, n))
        a.append(1)
        a.insert(0, 0)
        gen2 = a
    gens = [gen1, gen2]
    if gen1 == gen2:
        gens = gens[:1]
    G = PermutationGroup([_af_new(a) for a in gens], dups=False)
    set_alternating_group_properties(G, n, n)
    G._is_alt = True
    return G

def set_alternating_group_properties(G, n, degree):
    if False:
        i = 10
        return i + 15
    'Set known properties of an alternating group. '
    if n < 4:
        G._is_abelian = True
        G._is_nilpotent = True
    else:
        G._is_abelian = False
        G._is_nilpotent = False
    if n < 5:
        G._is_solvable = True
    else:
        G._is_solvable = False
    G._degree = degree
    G._is_transitive = True
    G._is_dihedral = False

def CyclicGroup(n):
    if False:
        for i in range(10):
            print('nop')
    '\n    Generates the cyclic group of order ``n`` as a permutation group.\n\n    Explanation\n    ===========\n\n    The generator taken is the ``n``-cycle ``(0 1 2 ... n-1)``\n    (in cycle notation). After the group is generated, some of its basic\n    properties are set.\n\n    Examples\n    ========\n\n    >>> from sympy.combinatorics.named_groups import CyclicGroup\n    >>> G = CyclicGroup(6)\n    >>> G.is_group\n    True\n    >>> G.order()\n    6\n    >>> list(G.generate_schreier_sims(af=True))\n    [[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0], [2, 3, 4, 5, 0, 1],\n    [3, 4, 5, 0, 1, 2], [4, 5, 0, 1, 2, 3], [5, 0, 1, 2, 3, 4]]\n\n    See Also\n    ========\n\n    SymmetricGroup, DihedralGroup, AlternatingGroup\n\n    '
    a = list(range(1, n))
    a.append(0)
    gen = _af_new(a)
    G = PermutationGroup([gen])
    G._is_abelian = True
    G._is_nilpotent = True
    G._is_solvable = True
    G._degree = n
    G._is_transitive = True
    G._order = n
    G._is_dihedral = n == 2
    return G

def DihedralGroup(n):
    if False:
        i = 10
        return i + 15
    '\n    Generates the dihedral group `D_n` as a permutation group.\n\n    Explanation\n    ===========\n\n    The dihedral group `D_n` is the group of symmetries of the regular\n    ``n``-gon. The generators taken are the ``n``-cycle ``a = (0 1 2 ... n-1)``\n    (a rotation of the ``n``-gon) and ``b = (0 n-1)(1 n-2)...``\n    (a reflection of the ``n``-gon) in cycle rotation. It is easy to see that\n    these satisfy ``a**n = b**2 = 1`` and ``bab = ~a`` so they indeed generate\n    `D_n` (See [1]). After the group is generated, some of its basic properties\n    are set.\n\n    Examples\n    ========\n\n    >>> from sympy.combinatorics.named_groups import DihedralGroup\n    >>> G = DihedralGroup(5)\n    >>> G.is_group\n    True\n    >>> a = list(G.generate_dimino())\n    >>> [perm.cyclic_form for perm in a]\n    [[], [[0, 1, 2, 3, 4]], [[0, 2, 4, 1, 3]],\n    [[0, 3, 1, 4, 2]], [[0, 4, 3, 2, 1]], [[0, 4], [1, 3]],\n    [[1, 4], [2, 3]], [[0, 1], [2, 4]], [[0, 2], [3, 4]],\n    [[0, 3], [1, 2]]]\n\n    See Also\n    ========\n\n    SymmetricGroup, CyclicGroup, AlternatingGroup\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Dihedral_group\n\n    '
    if n == 1:
        return PermutationGroup([Permutation([1, 0])])
    if n == 2:
        return PermutationGroup([Permutation([1, 0, 3, 2]), Permutation([2, 3, 0, 1]), Permutation([3, 2, 1, 0])])
    a = list(range(1, n))
    a.append(0)
    gen1 = _af_new(a)
    a = list(range(n))
    a.reverse()
    gen2 = _af_new(a)
    G = PermutationGroup([gen1, gen2])
    if n & n - 1 == 0:
        G._is_nilpotent = True
    else:
        G._is_nilpotent = False
    G._is_dihedral = True
    G._is_abelian = False
    G._is_solvable = True
    G._degree = n
    G._is_transitive = True
    G._order = 2 * n
    return G

def SymmetricGroup(n):
    if False:
        for i in range(10):
            print('nop')
    '\n    Generates the symmetric group on ``n`` elements as a permutation group.\n\n    Explanation\n    ===========\n\n    The generators taken are the ``n``-cycle\n    ``(0 1 2 ... n-1)`` and the transposition ``(0 1)`` (in cycle notation).\n    (See [1]). After the group is generated, some of its basic properties\n    are set.\n\n    Examples\n    ========\n\n    >>> from sympy.combinatorics.named_groups import SymmetricGroup\n    >>> G = SymmetricGroup(4)\n    >>> G.is_group\n    True\n    >>> G.order()\n    24\n    >>> list(G.generate_schreier_sims(af=True))\n    [[0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1], [3, 1, 2, 0], [0, 2, 3, 1],\n    [1, 3, 0, 2], [2, 0, 1, 3], [3, 2, 0, 1], [0, 3, 1, 2], [1, 0, 2, 3],\n    [2, 1, 3, 0], [3, 0, 1, 2], [0, 1, 3, 2], [1, 2, 0, 3], [2, 3, 1, 0],\n    [3, 1, 0, 2], [0, 2, 1, 3], [1, 3, 2, 0], [2, 0, 3, 1], [3, 2, 1, 0],\n    [0, 3, 2, 1], [1, 0, 3, 2], [2, 1, 0, 3], [3, 0, 2, 1]]\n\n    See Also\n    ========\n\n    CyclicGroup, DihedralGroup, AlternatingGroup\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Symmetric_group#Generators_and_relations\n\n    '
    if n == 1:
        G = PermutationGroup([Permutation([0])])
    elif n == 2:
        G = PermutationGroup([Permutation([1, 0])])
    else:
        a = list(range(1, n))
        a.append(0)
        gen1 = _af_new(a)
        a = list(range(n))
        (a[0], a[1]) = (a[1], a[0])
        gen2 = _af_new(a)
        G = PermutationGroup([gen1, gen2])
    set_symmetric_group_properties(G, n, n)
    G._is_sym = True
    return G

def set_symmetric_group_properties(G, n, degree):
    if False:
        return 10
    'Set known properties of a symmetric group. '
    if n < 3:
        G._is_abelian = True
        G._is_nilpotent = True
    else:
        G._is_abelian = False
        G._is_nilpotent = False
    if n < 5:
        G._is_solvable = True
    else:
        G._is_solvable = False
    G._degree = degree
    G._is_transitive = True
    G._is_dihedral = n in [2, 3]

def RubikGroup(n):
    if False:
        i = 10
        return i + 15
    "Return a group of Rubik's cube generators\n\n    >>> from sympy.combinatorics.named_groups import RubikGroup\n    >>> RubikGroup(2).is_group\n    True\n    "
    from sympy.combinatorics.generators import rubik
    if n <= 1:
        raise ValueError('Invalid cube. n has to be greater than 1')
    return PermutationGroup(rubik(n))