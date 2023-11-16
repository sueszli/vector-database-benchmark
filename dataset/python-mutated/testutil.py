from sympy.combinatorics import Permutation
from sympy.combinatorics.util import _distribute_gens_by_base
rmul = Permutation.rmul

def _cmp_perm_lists(first, second):
    if False:
        while True:
            i = 10
    '\n    Compare two lists of permutations as sets.\n\n    Explanation\n    ===========\n\n    This is used for testing purposes. Since the array form of a\n    permutation is currently a list, Permutation is not hashable\n    and cannot be put into a set.\n\n    Examples\n    ========\n\n    >>> from sympy.combinatorics.permutations import Permutation\n    >>> from sympy.combinatorics.testutil import _cmp_perm_lists\n    >>> a = Permutation([0, 2, 3, 4, 1])\n    >>> b = Permutation([1, 2, 0, 4, 3])\n    >>> c = Permutation([3, 4, 0, 1, 2])\n    >>> ls1 = [a, b, c]\n    >>> ls2 = [b, c, a]\n    >>> _cmp_perm_lists(ls1, ls2)\n    True\n\n    '
    return {tuple(a) for a in first} == {tuple(a) for a in second}

def _naive_list_centralizer(self, other, af=False):
    if False:
        i = 10
        return i + 15
    from sympy.combinatorics.perm_groups import PermutationGroup
    '\n    Return a list of elements for the centralizer of a subgroup/set/element.\n\n    Explanation\n    ===========\n\n    This is a brute force implementation that goes over all elements of the\n    group and checks for membership in the centralizer. It is used to\n    test ``.centralizer()`` from ``sympy.combinatorics.perm_groups``.\n\n    Examples\n    ========\n\n    >>> from sympy.combinatorics.testutil import _naive_list_centralizer\n    >>> from sympy.combinatorics.named_groups import DihedralGroup\n    >>> D = DihedralGroup(4)\n    >>> _naive_list_centralizer(D, D)\n    [Permutation([0, 1, 2, 3]), Permutation([2, 3, 0, 1])]\n\n    See Also\n    ========\n\n    sympy.combinatorics.perm_groups.centralizer\n\n    '
    from sympy.combinatorics.permutations import _af_commutes_with
    if hasattr(other, 'generators'):
        elements = list(self.generate_dimino(af=True))
        gens = [x._array_form for x in other.generators]
        commutes_with_gens = lambda x: all((_af_commutes_with(x, gen) for gen in gens))
        centralizer_list = []
        if not af:
            for element in elements:
                if commutes_with_gens(element):
                    centralizer_list.append(Permutation._af_new(element))
        else:
            for element in elements:
                if commutes_with_gens(element):
                    centralizer_list.append(element)
        return centralizer_list
    elif hasattr(other, 'getitem'):
        return _naive_list_centralizer(self, PermutationGroup(other), af)
    elif hasattr(other, 'array_form'):
        return _naive_list_centralizer(self, PermutationGroup([other]), af)

def _verify_bsgs(group, base, gens):
    if False:
        while True:
            i = 10
    '\n    Verify the correctness of a base and strong generating set.\n\n    Explanation\n    ===========\n\n    This is a naive implementation using the definition of a base and a strong\n    generating set relative to it. There are other procedures for\n    verifying a base and strong generating set, but this one will\n    serve for more robust testing.\n\n    Examples\n    ========\n\n    >>> from sympy.combinatorics.named_groups import AlternatingGroup\n    >>> from sympy.combinatorics.testutil import _verify_bsgs\n    >>> A = AlternatingGroup(4)\n    >>> A.schreier_sims()\n    >>> _verify_bsgs(A, A.base, A.strong_gens)\n    True\n\n    See Also\n    ========\n\n    sympy.combinatorics.perm_groups.PermutationGroup.schreier_sims\n\n    '
    from sympy.combinatorics.perm_groups import PermutationGroup
    strong_gens_distr = _distribute_gens_by_base(base, gens)
    current_stabilizer = group
    for i in range(len(base)):
        candidate = PermutationGroup(strong_gens_distr[i])
        if current_stabilizer.order() != candidate.order():
            return False
        current_stabilizer = current_stabilizer.stabilizer(base[i])
    if current_stabilizer.order() != 1:
        return False
    return True

def _verify_centralizer(group, arg, centr=None):
    if False:
        print('Hello World!')
    '\n    Verify the centralizer of a group/set/element inside another group.\n\n    This is used for testing ``.centralizer()`` from\n    ``sympy.combinatorics.perm_groups``\n\n    Examples\n    ========\n\n    >>> from sympy.combinatorics.named_groups import (SymmetricGroup,\n    ... AlternatingGroup)\n    >>> from sympy.combinatorics.perm_groups import PermutationGroup\n    >>> from sympy.combinatorics.permutations import Permutation\n    >>> from sympy.combinatorics.testutil import _verify_centralizer\n    >>> S = SymmetricGroup(5)\n    >>> A = AlternatingGroup(5)\n    >>> centr = PermutationGroup([Permutation([0, 1, 2, 3, 4])])\n    >>> _verify_centralizer(S, A, centr)\n    True\n\n    See Also\n    ========\n\n    _naive_list_centralizer,\n    sympy.combinatorics.perm_groups.PermutationGroup.centralizer,\n    _cmp_perm_lists\n\n    '
    if centr is None:
        centr = group.centralizer(arg)
    centr_list = list(centr.generate_dimino(af=True))
    centr_list_naive = _naive_list_centralizer(group, arg, af=True)
    return _cmp_perm_lists(centr_list, centr_list_naive)

def _verify_normal_closure(group, arg, closure=None):
    if False:
        while True:
            i = 10
    from sympy.combinatorics.perm_groups import PermutationGroup
    '\n    Verify the normal closure of a subgroup/subset/element in a group.\n\n    This is used to test\n    sympy.combinatorics.perm_groups.PermutationGroup.normal_closure\n\n    Examples\n    ========\n\n    >>> from sympy.combinatorics.named_groups import (SymmetricGroup,\n    ... AlternatingGroup)\n    >>> from sympy.combinatorics.testutil import _verify_normal_closure\n    >>> S = SymmetricGroup(3)\n    >>> A = AlternatingGroup(3)\n    >>> _verify_normal_closure(S, A, closure=A)\n    True\n\n    See Also\n    ========\n\n    sympy.combinatorics.perm_groups.PermutationGroup.normal_closure\n\n    '
    if closure is None:
        closure = group.normal_closure(arg)
    conjugates = set()
    if hasattr(arg, 'generators'):
        subgr_gens = arg.generators
    elif hasattr(arg, '__getitem__'):
        subgr_gens = arg
    elif hasattr(arg, 'array_form'):
        subgr_gens = [arg]
    for el in group.generate_dimino():
        for gen in subgr_gens:
            conjugates.add(gen ^ el)
    naive_closure = PermutationGroup(list(conjugates))
    return closure.is_subgroup(naive_closure)

def canonicalize_naive(g, dummies, sym, *v):
    if False:
        print('Hello World!')
    '\n    Canonicalize tensor formed by tensors of the different types.\n\n    Explanation\n    ===========\n\n    sym_i symmetry under exchange of two component tensors of type `i`\n          None  no symmetry\n          0     commuting\n          1     anticommuting\n\n    Parameters\n    ==========\n\n    g : Permutation representing the tensor.\n    dummies : List of dummy indices.\n    msym : Symmetry of the metric.\n    v : A list of (base_i, gens_i, n_i, sym_i) for tensors of type `i`.\n        base_i, gens_i BSGS for tensors of this type\n        n_i  number of tensors of type `i`\n\n    Returns\n    =======\n\n    Returns 0 if the tensor is zero, else returns the array form of\n    the permutation representing the canonical form of the tensor.\n\n    Examples\n    ========\n\n    >>> from sympy.combinatorics.testutil import canonicalize_naive\n    >>> from sympy.combinatorics.tensor_can import get_symmetric_group_sgs\n    >>> from sympy.combinatorics import Permutation\n    >>> g = Permutation([1, 3, 2, 0, 4, 5])\n    >>> base2, gens2 = get_symmetric_group_sgs(2)\n    >>> canonicalize_naive(g, [2, 3], 0, (base2, gens2, 2, 0))\n    [0, 2, 1, 3, 4, 5]\n    '
    from sympy.combinatorics.perm_groups import PermutationGroup
    from sympy.combinatorics.tensor_can import gens_products, dummy_sgs
    from sympy.combinatorics.permutations import _af_rmul
    v1 = []
    for i in range(len(v)):
        (base_i, gens_i, n_i, sym_i) = v[i]
        v1.append((base_i, gens_i, [[]] * n_i, sym_i))
    (size, sbase, sgens) = gens_products(*v1)
    dgens = dummy_sgs(dummies, sym, size - 2)
    if isinstance(sym, int):
        num_types = 1
        dummies = [dummies]
        sym = [sym]
    else:
        num_types = len(sym)
    dgens = []
    for i in range(num_types):
        dgens.extend(dummy_sgs(dummies[i], sym[i], size - 2))
    S = PermutationGroup(sgens)
    D = PermutationGroup([Permutation(x) for x in dgens])
    dlist = list(D.generate(af=True))
    g = g.array_form
    st = set()
    for s in S.generate(af=True):
        h = _af_rmul(g, s)
        for d in dlist:
            q = tuple(_af_rmul(d, h))
            st.add(q)
    a = list(st)
    a.sort()
    prev = (0,) * size
    for h in a:
        if h[:-2] == prev[:-2]:
            if h[-1] != prev[-1]:
                return 0
        prev = h
    return list(a[0])

def graph_certificate(gr):
    if False:
        while True:
            i = 10
    '\n    Return a certificate for the graph\n\n    Parameters\n    ==========\n\n    gr : adjacency list\n\n    Explanation\n    ===========\n\n    The graph is assumed to be unoriented and without\n    external lines.\n\n    Associate to each vertex of the graph a symmetric tensor with\n    number of indices equal to the degree of the vertex; indices\n    are contracted when they correspond to the same line of the graph.\n    The canonical form of the tensor gives a certificate for the graph.\n\n    This is not an efficient algorithm to get the certificate of a graph.\n\n    Examples\n    ========\n\n    >>> from sympy.combinatorics.testutil import graph_certificate\n    >>> gr1 = {0:[1, 2, 3, 5], 1:[0, 2, 4], 2:[0, 1, 3, 4], 3:[0, 2, 4], 4:[1, 2, 3, 5], 5:[0, 4]}\n    >>> gr2 = {0:[1, 5], 1:[0, 2, 3, 4], 2:[1, 3, 5], 3:[1, 2, 4, 5], 4:[1, 3, 5], 5:[0, 2, 3, 4]}\n    >>> c1 = graph_certificate(gr1)\n    >>> c2 = graph_certificate(gr2)\n    >>> c1\n    [0, 2, 4, 6, 1, 8, 10, 12, 3, 14, 16, 18, 5, 9, 15, 7, 11, 17, 13, 19, 20, 21]\n    >>> c1 == c2\n    True\n    '
    from sympy.combinatorics.permutations import _af_invert
    from sympy.combinatorics.tensor_can import get_symmetric_group_sgs, canonicalize
    items = list(gr.items())
    items.sort(key=lambda x: len(x[1]), reverse=True)
    pvert = [x[0] for x in items]
    pvert = _af_invert(pvert)
    num_indices = 0
    for (v, neigh) in items:
        num_indices += len(neigh)
    vertices = [[] for i in items]
    i = 0
    for (v, neigh) in items:
        for v2 in neigh:
            if pvert[v] < pvert[v2]:
                vertices[pvert[v]].append(i)
                vertices[pvert[v2]].append(i + 1)
                i += 2
    g = []
    for v in vertices:
        g.extend(v)
    assert len(g) == num_indices
    g += [num_indices, num_indices + 1]
    size = num_indices + 2
    assert sorted(g) == list(range(size))
    g = Permutation(g)
    vlen = [0] * (len(vertices[0]) + 1)
    for neigh in vertices:
        vlen[len(neigh)] += 1
    v = []
    for i in range(len(vlen)):
        n = vlen[i]
        if n:
            (base, gens) = get_symmetric_group_sgs(i)
            v.append((base, gens, n, 0))
    v.reverse()
    dummies = list(range(num_indices))
    can = canonicalize(g, dummies, 0, *v)
    return can