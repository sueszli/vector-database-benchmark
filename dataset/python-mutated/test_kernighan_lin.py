"""Unit tests for the :mod:`networkx.algorithms.community.kernighan_lin`
module.
"""
from itertools import permutations
import pytest
import networkx as nx
from networkx.algorithms.community import kernighan_lin_bisection

def assert_partition_equal(x, y):
    if False:
        for i in range(10):
            print('nop')
    assert set(map(frozenset, x)) == set(map(frozenset, y))

def test_partition():
    if False:
        print('Hello World!')
    G = nx.barbell_graph(3, 0)
    C = kernighan_lin_bisection(G)
    assert_partition_equal(C, [{0, 1, 2}, {3, 4, 5}])

def test_partition_argument():
    if False:
        while True:
            i = 10
    G = nx.barbell_graph(3, 0)
    partition = [{0, 1, 2}, {3, 4, 5}]
    C = kernighan_lin_bisection(G, partition)
    assert_partition_equal(C, partition)

def test_partition_argument_non_integer_nodes():
    if False:
        i = 10
        return i + 15
    G = nx.Graph([('A', 'B'), ('A', 'C'), ('B', 'C'), ('C', 'D')])
    partition = ({'A', 'B'}, {'C', 'D'})
    C = kernighan_lin_bisection(G, partition)
    assert_partition_equal(C, partition)

def test_seed_argument():
    if False:
        i = 10
        return i + 15
    G = nx.barbell_graph(3, 0)
    C = kernighan_lin_bisection(G, seed=1)
    assert_partition_equal(C, [{0, 1, 2}, {3, 4, 5}])

def test_non_disjoint_partition():
    if False:
        while True:
            i = 10
    with pytest.raises(nx.NetworkXError):
        G = nx.barbell_graph(3, 0)
        partition = ({0, 1, 2}, {2, 3, 4, 5})
        kernighan_lin_bisection(G, partition)

def test_too_many_blocks():
    if False:
        i = 10
        return i + 15
    with pytest.raises(nx.NetworkXError):
        G = nx.barbell_graph(3, 0)
        partition = ({0, 1}, {2}, {3, 4, 5})
        kernighan_lin_bisection(G, partition)

def test_multigraph():
    if False:
        i = 10
        return i + 15
    G = nx.cycle_graph(4)
    M = nx.MultiGraph(G.edges())
    M.add_edges_from(G.edges())
    M.remove_edge(1, 2)
    for labels in permutations(range(4)):
        mapping = dict(zip(M, labels))
        (A, B) = kernighan_lin_bisection(nx.relabel_nodes(M, mapping), seed=0)
        assert_partition_equal([A, B], [{mapping[0], mapping[1]}, {mapping[2], mapping[3]}])

def test_max_iter_argument():
    if False:
        for i in range(10):
            print('nop')
    G = nx.Graph([('A', 'B', {'weight': 1}), ('A', 'C', {'weight': 2}), ('A', 'D', {'weight': 3}), ('A', 'E', {'weight': 2}), ('A', 'F', {'weight': 4}), ('B', 'C', {'weight': 1}), ('B', 'D', {'weight': 4}), ('B', 'E', {'weight': 2}), ('B', 'F', {'weight': 1}), ('C', 'D', {'weight': 3}), ('C', 'E', {'weight': 2}), ('C', 'F', {'weight': 1}), ('D', 'E', {'weight': 4}), ('D', 'F', {'weight': 3}), ('E', 'F', {'weight': 2})])
    partition = ({'A', 'B', 'C'}, {'D', 'E', 'F'})
    C = kernighan_lin_bisection(G, partition, max_iter=1)
    assert_partition_equal(C, ({'A', 'F', 'C'}, {'D', 'E', 'B'}))