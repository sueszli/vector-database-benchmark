"""Unit tests for the sparsifier computation functions."""
import pytest
import networkx as nx
from networkx.utils import py_random_state
_seed = 2

def _test_spanner(G, spanner, stretch, weight=None):
    if False:
        i = 10
        return i + 15
    'Test whether a spanner is valid.\n\n    This function tests whether the given spanner is a subgraph of the\n    given graph G with the same node set. It also tests for all shortest\n    paths whether they adhere to the given stretch.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n        The original graph for which the spanner was constructed.\n\n    spanner : NetworkX graph\n        The spanner to be tested.\n\n    stretch : float\n        The proclaimed stretch of the spanner.\n\n    weight : object\n        The edge attribute to use as distance.\n    '
    assert set(G.nodes()) == set(spanner.nodes())
    for (u, v) in spanner.edges():
        assert G.has_edge(u, v)
        if weight:
            assert spanner[u][v][weight] == G[u][v][weight]
    original_length = dict(nx.shortest_path_length(G, weight=weight))
    spanner_length = dict(nx.shortest_path_length(spanner, weight=weight))
    for u in G.nodes():
        for v in G.nodes():
            if u in original_length and v in original_length[u]:
                assert spanner_length[u][v] <= stretch * original_length[u][v]

@py_random_state(1)
def _assign_random_weights(G, seed=None):
    if False:
        while True:
            i = 10
    'Assigns random weights to the edges of a graph.\n\n    Parameters\n    ----------\n\n    G : NetworkX graph\n        The original graph for which the spanner was constructed.\n\n    seed : integer, random_state, or None (default)\n        Indicator of random number generation state.\n        See :ref:`Randomness<randomness>`.\n    '
    for (u, v) in G.edges():
        G[u][v]['weight'] = seed.random()

def test_spanner_trivial():
    if False:
        i = 10
        return i + 15
    'Test a trivial spanner with stretch 1.'
    G = nx.complete_graph(20)
    spanner = nx.spanner(G, 1, seed=_seed)
    for (u, v) in G.edges:
        assert spanner.has_edge(u, v)

def test_spanner_unweighted_complete_graph():
    if False:
        print('Hello World!')
    'Test spanner construction on a complete unweighted graph.'
    G = nx.complete_graph(20)
    spanner = nx.spanner(G, 4, seed=_seed)
    _test_spanner(G, spanner, 4)
    spanner = nx.spanner(G, 10, seed=_seed)
    _test_spanner(G, spanner, 10)

def test_spanner_weighted_complete_graph():
    if False:
        return 10
    'Test spanner construction on a complete weighted graph.'
    G = nx.complete_graph(20)
    _assign_random_weights(G, seed=_seed)
    spanner = nx.spanner(G, 4, weight='weight', seed=_seed)
    _test_spanner(G, spanner, 4, weight='weight')
    spanner = nx.spanner(G, 10, weight='weight', seed=_seed)
    _test_spanner(G, spanner, 10, weight='weight')

def test_spanner_unweighted_gnp_graph():
    if False:
        for i in range(10):
            print('nop')
    'Test spanner construction on an unweighted gnp graph.'
    G = nx.gnp_random_graph(20, 0.4, seed=_seed)
    spanner = nx.spanner(G, 4, seed=_seed)
    _test_spanner(G, spanner, 4)
    spanner = nx.spanner(G, 10, seed=_seed)
    _test_spanner(G, spanner, 10)

def test_spanner_weighted_gnp_graph():
    if False:
        i = 10
        return i + 15
    'Test spanner construction on an weighted gnp graph.'
    G = nx.gnp_random_graph(20, 0.4, seed=_seed)
    _assign_random_weights(G, seed=_seed)
    spanner = nx.spanner(G, 4, weight='weight', seed=_seed)
    _test_spanner(G, spanner, 4, weight='weight')
    spanner = nx.spanner(G, 10, weight='weight', seed=_seed)
    _test_spanner(G, spanner, 10, weight='weight')

def test_spanner_unweighted_disconnected_graph():
    if False:
        return 10
    'Test spanner construction on a disconnected graph.'
    G = nx.disjoint_union(nx.complete_graph(10), nx.complete_graph(10))
    spanner = nx.spanner(G, 4, seed=_seed)
    _test_spanner(G, spanner, 4)
    spanner = nx.spanner(G, 10, seed=_seed)
    _test_spanner(G, spanner, 10)

def test_spanner_invalid_stretch():
    if False:
        while True:
            i = 10
    'Check whether an invalid stretch is caught.'
    with pytest.raises(ValueError):
        G = nx.empty_graph()
        nx.spanner(G, 0)