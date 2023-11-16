import warnings
import pytest
import networkx as nx

def test_smetric():
    if False:
        for i in range(10):
            print('nop')
    g = nx.Graph()
    g.add_edge(1, 2)
    g.add_edge(2, 3)
    g.add_edge(2, 4)
    g.add_edge(1, 4)
    sm = nx.s_metric(g, normalized=False)
    assert sm == 19.0

def test_normalized_deprecation_warning():
    if False:
        while True:
            i = 10
    'Test that a deprecation warning is raised when s_metric is called with\n    a `normalized` kwarg.'
    G = nx.cycle_graph(7)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert nx.s_metric(G) == 28
    with pytest.deprecated_call():
        nx.s_metric(G, normalized=True)
    with pytest.raises(TypeError):
        nx.s_metric(G, normalize=True)