import copy
import json
import pytest
import networkx as nx
from networkx.readwrite.json_graph import cytoscape_data, cytoscape_graph

def test_graph():
    if False:
        i = 10
        return i + 15
    G = nx.path_graph(4)
    H = cytoscape_graph(cytoscape_data(G))
    assert nx.is_isomorphic(G, H)

def test_input_data_is_not_modified_when_building_graph():
    if False:
        print('Hello World!')
    G = nx.path_graph(4)
    input_data = cytoscape_data(G)
    orig_data = copy.deepcopy(input_data)
    cytoscape_graph(input_data)
    assert input_data == orig_data

def test_graph_attributes():
    if False:
        return 10
    G = nx.path_graph(4)
    G.add_node(1, color='red')
    G.add_edge(1, 2, width=7)
    G.graph['foo'] = 'bar'
    G.graph[1] = 'one'
    G.add_node(3, name='node', id='123')
    H = cytoscape_graph(cytoscape_data(G))
    assert H.graph['foo'] == 'bar'
    assert H.nodes[1]['color'] == 'red'
    assert H[1][2]['width'] == 7
    assert H.nodes[3]['name'] == 'node'
    assert H.nodes[3]['id'] == '123'
    d = json.dumps(cytoscape_data(G))
    H = cytoscape_graph(json.loads(d))
    assert H.graph['foo'] == 'bar'
    assert H.graph[1] == 'one'
    assert H.nodes[1]['color'] == 'red'
    assert H[1][2]['width'] == 7
    assert H.nodes[3]['name'] == 'node'
    assert H.nodes[3]['id'] == '123'

def test_digraph():
    if False:
        return 10
    G = nx.DiGraph()
    nx.add_path(G, [1, 2, 3])
    H = cytoscape_graph(cytoscape_data(G))
    assert H.is_directed()
    assert nx.is_isomorphic(G, H)

def test_multidigraph():
    if False:
        for i in range(10):
            print('nop')
    G = nx.MultiDiGraph()
    nx.add_path(G, [1, 2, 3])
    H = cytoscape_graph(cytoscape_data(G))
    assert H.is_directed()
    assert H.is_multigraph()

def test_multigraph():
    if False:
        i = 10
        return i + 15
    G = nx.MultiGraph()
    G.add_edge(1, 2, key='first')
    G.add_edge(1, 2, key='second', color='blue')
    H = cytoscape_graph(cytoscape_data(G))
    assert nx.is_isomorphic(G, H)
    assert H[1][2]['second']['color'] == 'blue'

def test_exception():
    if False:
        print('Hello World!')
    with pytest.raises(nx.NetworkXError):
        G = nx.MultiDiGraph()
        cytoscape_data(G, name='foo', ident='foo')