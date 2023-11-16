"""Unit tests for the :mod:`networkx.algorithms.tournament` module."""
from itertools import combinations
import pytest
from networkx import DiGraph
from networkx.algorithms.tournament import hamiltonian_path, index_satisfying, is_reachable, is_strongly_connected, is_tournament, random_tournament, score_sequence, tournament_matrix

def test_condition_not_satisfied():
    if False:
        for i in range(10):
            print('nop')
    condition = lambda x: x > 0
    iter_in = [0]
    assert index_satisfying(iter_in, condition) == 1

def test_empty_iterable():
    if False:
        print('Hello World!')
    condition = lambda x: x > 0
    with pytest.raises(ValueError):
        index_satisfying([], condition)

def test_is_tournament():
    if False:
        i = 10
        return i + 15
    G = DiGraph()
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0), (1, 3), (0, 2)])
    assert is_tournament(G)

def test_self_loops():
    if False:
        print('Hello World!')
    'A tournament must have no self-loops.'
    G = DiGraph()
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0), (1, 3), (0, 2)])
    G.add_edge(0, 0)
    assert not is_tournament(G)

def test_missing_edges():
    if False:
        return 10
    'A tournament must not have any pair of nodes without at least\n    one edge joining the pair.\n\n    '
    G = DiGraph()
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0), (1, 3)])
    assert not is_tournament(G)

def test_bidirectional_edges():
    if False:
        for i in range(10):
            print('nop')
    'A tournament must not have any pair of nodes with greater\n    than one edge joining the pair.\n\n    '
    G = DiGraph()
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0), (1, 3), (0, 2)])
    G.add_edge(1, 0)
    assert not is_tournament(G)

def test_graph_is_tournament():
    if False:
        for i in range(10):
            print('nop')
    for _ in range(10):
        G = random_tournament(5)
        assert is_tournament(G)

def test_graph_is_tournament_seed():
    if False:
        return 10
    for _ in range(10):
        G = random_tournament(5, seed=1)
        assert is_tournament(G)

def test_graph_is_tournament_one_node():
    if False:
        return 10
    G = random_tournament(1)
    assert is_tournament(G)

def test_graph_is_tournament_zero_node():
    if False:
        for i in range(10):
            print('nop')
    G = random_tournament(0)
    assert is_tournament(G)

def test_hamiltonian_empty_graph():
    if False:
        for i in range(10):
            print('nop')
    path = hamiltonian_path(DiGraph())
    assert len(path) == 0

def test_path_is_hamiltonian():
    if False:
        i = 10
        return i + 15
    G = DiGraph()
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0), (1, 3), (0, 2)])
    path = hamiltonian_path(G)
    assert len(path) == 4
    assert all((v in G[u] for (u, v) in zip(path, path[1:])))

def test_hamiltonian_cycle():
    if False:
        while True:
            i = 10
    'Tests that :func:`networkx.tournament.hamiltonian_path`\n    returns a Hamiltonian cycle when provided a strongly connected\n    tournament.\n\n    '
    G = DiGraph()
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0), (1, 3), (0, 2)])
    path = hamiltonian_path(G)
    assert len(path) == 4
    assert all((v in G[u] for (u, v) in zip(path, path[1:])))
    assert path[0] in G[path[-1]]

def test_score_sequence_edge():
    if False:
        i = 10
        return i + 15
    G = DiGraph([(0, 1)])
    assert score_sequence(G) == [0, 1]

def test_score_sequence_triangle():
    if False:
        print('Hello World!')
    G = DiGraph([(0, 1), (1, 2), (2, 0)])
    assert score_sequence(G) == [1, 1, 1]

def test_tournament_matrix():
    if False:
        return 10
    np = pytest.importorskip('numpy')
    pytest.importorskip('scipy')
    npt = np.testing
    G = DiGraph([(0, 1)])
    m = tournament_matrix(G)
    npt.assert_array_equal(m.todense(), np.array([[0, 1], [-1, 0]]))

def test_reachable_pair():
    if False:
        i = 10
        return i + 15
    'Tests for a reachable pair of nodes.'
    G = DiGraph([(0, 1), (1, 2), (2, 0)])
    assert is_reachable(G, 0, 2)

def test_same_node_is_reachable():
    if False:
        while True:
            i = 10
    'Tests that a node is always reachable from it.'
    G = DiGraph((sorted(p) for p in combinations(range(10), 2)))
    assert all((is_reachable(G, v, v) for v in G))

def test_unreachable_pair():
    if False:
        i = 10
        return i + 15
    'Tests for an unreachable pair of nodes.'
    G = DiGraph([(0, 1), (0, 2), (1, 2)])
    assert not is_reachable(G, 1, 0)

def test_is_strongly_connected():
    if False:
        for i in range(10):
            print('nop')
    'Tests for a strongly connected tournament.'
    G = DiGraph([(0, 1), (1, 2), (2, 0)])
    assert is_strongly_connected(G)

def test_not_strongly_connected():
    if False:
        print('Hello World!')
    'Tests for a tournament that is not strongly connected.'
    G = DiGraph([(0, 1), (0, 2), (1, 2)])
    assert not is_strongly_connected(G)