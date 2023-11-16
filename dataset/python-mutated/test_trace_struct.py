import itertools
import pytest
from pyro.poutine import Trace
from tests.common import assert_equal
EDGE_SETS = [[(1, 2), (1, 3), (3, 4), (3, 5), (4, 6), (4, 7)], [(1, 2), (3, 5), (1, 4), (1, 3), (5, 6), (6, 7)]]

@pytest.mark.parametrize('edges', [perm for edges in EDGE_SETS for perm in itertools.permutations(edges)])
def test_topological_sort(edges):
    if False:
        return 10
    tr = Trace()
    for (n1, n2) in edges:
        tr.add_edge(n1, n2)
    top_sort = tr.topological_sort()
    expected_nodes = set().union(*edges)
    assert len(top_sort) == len(expected_nodes)
    assert set(top_sort) == expected_nodes
    ranks = {n: rank for (rank, n) in enumerate(top_sort)}
    for (n1, n2) in edges:
        assert ranks[n1] < ranks[n2]

@pytest.mark.parametrize('edges', [perm for edges in EDGE_SETS for perm in itertools.permutations(edges)])
def test_connectivity_on_removal(edges):
    if False:
        i = 10
        return i + 15
    root = 1
    tr = Trace()
    for (e1, e2) in edges:
        tr.add_edge(e1, e2)
    top_sort = tr.topological_sort()
    while top_sort:
        num_nodes = len([n for n in tr._dfs(root, set())])
        num_expected = len(top_sort)
        assert_equal(num_nodes, num_expected)
        tr.remove_node(top_sort.pop())