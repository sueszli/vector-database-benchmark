import pytest
import networkx as nx

def test_dominating_set():
    if False:
        for i in range(10):
            print('nop')
    G = nx.gnp_random_graph(100, 0.1)
    D = nx.dominating_set(G)
    assert nx.is_dominating_set(G, D)
    D = nx.dominating_set(G, start_with=0)
    assert nx.is_dominating_set(G, D)

def test_complete():
    if False:
        for i in range(10):
            print('nop')
    'In complete graphs each node is a dominating set.\n    Thus the dominating set has to be of cardinality 1.\n    '
    K4 = nx.complete_graph(4)
    assert len(nx.dominating_set(K4)) == 1
    K5 = nx.complete_graph(5)
    assert len(nx.dominating_set(K5)) == 1

def test_raise_dominating_set():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(nx.NetworkXError):
        G = nx.path_graph(4)
        D = nx.dominating_set(G, start_with=10)

def test_is_dominating_set():
    if False:
        for i in range(10):
            print('nop')
    G = nx.path_graph(4)
    d = {1, 3}
    assert nx.is_dominating_set(G, d)
    d = {0, 2}
    assert nx.is_dominating_set(G, d)
    d = {1}
    assert not nx.is_dominating_set(G, d)

def test_wikipedia_is_dominating_set():
    if False:
        return 10
    'Example from https://en.wikipedia.org/wiki/Dominating_set'
    G = nx.cycle_graph(4)
    G.add_edges_from([(0, 4), (1, 4), (2, 5)])
    assert nx.is_dominating_set(G, {4, 3, 5})
    assert nx.is_dominating_set(G, {0, 2})
    assert nx.is_dominating_set(G, {1, 2})