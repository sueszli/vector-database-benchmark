from itertools import zip_longest
from sympy.utilities.enumerative import list_visitor, MultisetPartitionTraverser, multiset_partitions_taocp
from sympy.utilities.iterables import _set_partitions

def part_range_filter(partition_iterator, lb, ub):
    if False:
        while True:
            i = 10
    '\n    Filters (on the number of parts) a multiset partition enumeration\n\n    Arguments\n    =========\n\n    lb, and ub are a range (in the Python slice sense) on the lpart\n    variable returned from a multiset partition enumeration.  Recall\n    that lpart is 0-based (it points to the topmost part on the part\n    stack), so if you want to return parts of sizes 2,3,4,5 you would\n    use lb=1 and ub=5.\n    '
    for state in partition_iterator:
        (f, lpart, pstack) = state
        if lpart >= lb and lpart < ub:
            yield state

def multiset_partitions_baseline(multiplicities, components):
    if False:
        while True:
            i = 10
    'Enumerates partitions of a multiset\n\n    Parameters\n    ==========\n\n    multiplicities\n         list of integer multiplicities of the components of the multiset.\n\n    components\n         the components (elements) themselves\n\n    Returns\n    =======\n\n    Set of partitions.  Each partition is tuple of parts, and each\n    part is a tuple of components (with repeats to indicate\n    multiplicity)\n\n    Notes\n    =====\n\n    Multiset partitions can be created as equivalence classes of set\n    partitions, and this function does just that.  This approach is\n    slow and memory intensive compared to the more advanced algorithms\n    available, but the code is simple and easy to understand.  Hence\n    this routine is strictly for testing -- to provide a\n    straightforward baseline against which to regress the production\n    versions.  (This code is a simplified version of an earlier\n    production implementation.)\n    '
    canon = []
    for (ct, elem) in zip(multiplicities, components):
        canon.extend([elem] * ct)
    cache = set()
    n = len(canon)
    for (nc, q) in _set_partitions(n):
        rv = [[] for i in range(nc)]
        for i in range(n):
            rv[q[i]].append(canon[i])
        canonical = tuple(sorted([tuple(p) for p in rv]))
        cache.add(canonical)
    return cache

def compare_multiset_w_baseline(multiplicities):
    if False:
        i = 10
        return i + 15
    '\n    Enumerates the partitions of multiset with AOCP algorithm and\n    baseline implementation, and compare the results.\n\n    '
    letters = 'abcdefghijklmnopqrstuvwxyz'
    bl_partitions = multiset_partitions_baseline(multiplicities, letters)
    aocp_partitions = set()
    for state in multiset_partitions_taocp(multiplicities):
        p1 = tuple(sorted([tuple(p) for p in list_visitor(state, letters)]))
        aocp_partitions.add(p1)
    assert bl_partitions == aocp_partitions

def compare_multiset_states(s1, s2):
    if False:
        print('Hello World!')
    'compare for equality two instances of multiset partition states\n\n    This is useful for comparing different versions of the algorithm\n    to verify correctness.'
    (f1, lpart1, pstack1) = s1
    (f2, lpart2, pstack2) = s2
    if lpart1 == lpart2 and f1[0:lpart1 + 1] == f2[0:lpart2 + 1]:
        if pstack1[0:f1[lpart1 + 1]] == pstack2[0:f2[lpart2 + 1]]:
            return True
    return False

def test_multiset_partitions_taocp():
    if False:
        for i in range(10):
            print('nop')
    'Compares the output of multiset_partitions_taocp with a baseline\n    (set partition based) implementation.'
    multiplicities = [2, 2]
    compare_multiset_w_baseline(multiplicities)
    multiplicities = [4, 3, 1]
    compare_multiset_w_baseline(multiplicities)

def test_multiset_partitions_versions():
    if False:
        return 10
    'Compares Knuth-based versions of multiset_partitions'
    multiplicities = [5, 2, 2, 1]
    m = MultisetPartitionTraverser()
    for (s1, s2) in zip_longest(m.enum_all(multiplicities), multiset_partitions_taocp(multiplicities)):
        assert compare_multiset_states(s1, s2)

def subrange_exercise(mult, lb, ub):
    if False:
        print('Hello World!')
    'Compare filter-based and more optimized subrange implementations\n\n    Helper for tests, called with both small and larger multisets.\n    '
    m = MultisetPartitionTraverser()
    assert m.count_partitions(mult) == m.count_partitions_slow(mult)
    ma = MultisetPartitionTraverser()
    mc = MultisetPartitionTraverser()
    md = MultisetPartitionTraverser()
    a_it = ma.enum_range(mult, lb, ub)
    b_it = part_range_filter(multiset_partitions_taocp(mult), lb, ub)
    c_it = part_range_filter(mc.enum_small(mult, ub), lb, sum(mult))
    d_it = part_range_filter(md.enum_large(mult, lb), 0, ub)
    for (sa, sb, sc, sd) in zip_longest(a_it, b_it, c_it, d_it):
        assert compare_multiset_states(sa, sb)
        assert compare_multiset_states(sa, sc)
        assert compare_multiset_states(sa, sd)

def test_subrange():
    if False:
        i = 10
        return i + 15
    mult = [4, 4, 2, 1]
    lb = 1
    ub = 2
    subrange_exercise(mult, lb, ub)

def test_subrange_large():
    if False:
        for i in range(10):
            print('nop')
    mult = [6, 3, 2, 1]
    lb = 4
    ub = 7
    subrange_exercise(mult, lb, ub)