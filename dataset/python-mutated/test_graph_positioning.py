import networkx as nx
import pytest
from tribler.core.components.bandwidth_accounting.trust_calculation.graph_positioning import GraphPositioning

def test_graph_positioning_not_tree():
    if False:
        print('Hello World!')
    '\n    Test whether we get an error if we do not pass a tree to the graph positioning logic\n    '
    G = nx.DiGraph()
    G.add_edge('a', 'b')
    G.add_edge('b', 'a')
    with pytest.raises(TypeError):
        GraphPositioning.hierarchy_pos(G)

def test_graph_positioning():
    if False:
        while True:
            i = 10
    '\n    Test whether we get a tree layout\n    '
    G = nx.DiGraph()
    G.add_edge('a', 'b')
    G.add_edge('a', 'd')
    G.add_edge('b', 'c')
    result = GraphPositioning.hierarchy_pos(G)
    assert len(result.keys()) == 4