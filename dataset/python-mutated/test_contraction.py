"""Unit tests for the :mod:`networkx.algorithms.minors.contraction` module."""
import pytest
import networkx as nx
from networkx.utils import arbitrary_element, edges_equal, nodes_equal

def test_quotient_graph_complete_multipartite():
    if False:
        return 10
    'Tests that the quotient graph of the complete *n*-partite graph\n    under the "same neighbors" node relation is the complete graph on *n*\n    nodes.\n\n    '
    G = nx.complete_multipartite_graph(2, 3, 4)

    def same_neighbors(u, v):
        if False:
            i = 10
            return i + 15
        return u not in G[v] and v not in G[u] and (G[u] == G[v])
    expected = nx.complete_graph(3)
    actual = nx.quotient_graph(G, same_neighbors)
    assert nx.is_isomorphic(expected, actual)

def test_quotient_graph_complete_bipartite():
    if False:
        i = 10
        return i + 15
    'Tests that the quotient graph of the complete bipartite graph under\n    the "same neighbors" node relation is `K_2`.\n\n    '
    G = nx.complete_bipartite_graph(2, 3)

    def same_neighbors(u, v):
        if False:
            print('Hello World!')
        return u not in G[v] and v not in G[u] and (G[u] == G[v])
    expected = nx.complete_graph(2)
    actual = nx.quotient_graph(G, same_neighbors)
    assert nx.is_isomorphic(expected, actual)

def test_quotient_graph_edge_relation():
    if False:
        print('Hello World!')
    'Tests for specifying an alternate edge relation for the quotient\n    graph.\n\n    '
    G = nx.path_graph(5)

    def identity(u, v):
        if False:
            while True:
                i = 10
        return u == v

    def same_parity(b, c):
        if False:
            for i in range(10):
                print('nop')
        return arbitrary_element(b) % 2 == arbitrary_element(c) % 2
    actual = nx.quotient_graph(G, identity, same_parity)
    expected = nx.Graph()
    expected.add_edges_from([(0, 2), (0, 4), (2, 4)])
    expected.add_edge(1, 3)
    assert nx.is_isomorphic(actual, expected)

def test_condensation_as_quotient():
    if False:
        i = 10
        return i + 15
    'This tests that the condensation of a graph can be viewed as the\n    quotient graph under the "in the same connected component" equivalence\n    relation.\n\n    '
    G = nx.DiGraph()
    G.add_edges_from([(1, 2), (2, 3), (2, 11), (2, 12), (3, 4), (4, 3), (4, 5), (5, 6), (6, 5), (6, 7), (7, 8), (7, 9), (7, 10), (8, 9), (9, 7), (10, 6), (11, 2), (11, 4), (11, 6), (12, 6), (12, 11)])
    scc = list(nx.strongly_connected_components(G))
    C = nx.condensation(G, scc)
    component_of = C.graph['mapping']

    def same_component(u, v):
        if False:
            for i in range(10):
                print('nop')
        return component_of[u] == component_of[v]
    Q = nx.quotient_graph(G, same_component)
    assert nx.is_isomorphic(C, Q)

def test_path():
    if False:
        print('Hello World!')
    G = nx.path_graph(6)
    partition = [{0, 1}, {2, 3}, {4, 5}]
    M = nx.quotient_graph(G, partition, relabel=True)
    assert nodes_equal(M, [0, 1, 2])
    assert edges_equal(M.edges(), [(0, 1), (1, 2)])
    for n in M:
        assert M.nodes[n]['nedges'] == 1
        assert M.nodes[n]['nnodes'] == 2
        assert M.nodes[n]['density'] == 1

def test_path__partition_provided_as_dict_of_lists():
    if False:
        i = 10
        return i + 15
    G = nx.path_graph(6)
    partition = {0: [0, 1], 2: [2, 3], 4: [4, 5]}
    M = nx.quotient_graph(G, partition, relabel=True)
    assert nodes_equal(M, [0, 1, 2])
    assert edges_equal(M.edges(), [(0, 1), (1, 2)])
    for n in M:
        assert M.nodes[n]['nedges'] == 1
        assert M.nodes[n]['nnodes'] == 2
        assert M.nodes[n]['density'] == 1

def test_path__partition_provided_as_dict_of_tuples():
    if False:
        while True:
            i = 10
    G = nx.path_graph(6)
    partition = {0: (0, 1), 2: (2, 3), 4: (4, 5)}
    M = nx.quotient_graph(G, partition, relabel=True)
    assert nodes_equal(M, [0, 1, 2])
    assert edges_equal(M.edges(), [(0, 1), (1, 2)])
    for n in M:
        assert M.nodes[n]['nedges'] == 1
        assert M.nodes[n]['nnodes'] == 2
        assert M.nodes[n]['density'] == 1

def test_path__partition_provided_as_dict_of_sets():
    if False:
        while True:
            i = 10
    G = nx.path_graph(6)
    partition = {0: {0, 1}, 2: {2, 3}, 4: {4, 5}}
    M = nx.quotient_graph(G, partition, relabel=True)
    assert nodes_equal(M, [0, 1, 2])
    assert edges_equal(M.edges(), [(0, 1), (1, 2)])
    for n in M:
        assert M.nodes[n]['nedges'] == 1
        assert M.nodes[n]['nnodes'] == 2
        assert M.nodes[n]['density'] == 1

def test_multigraph_path():
    if False:
        print('Hello World!')
    G = nx.MultiGraph(nx.path_graph(6))
    partition = [{0, 1}, {2, 3}, {4, 5}]
    M = nx.quotient_graph(G, partition, relabel=True)
    assert nodes_equal(M, [0, 1, 2])
    assert edges_equal(M.edges(), [(0, 1), (1, 2)])
    for n in M:
        assert M.nodes[n]['nedges'] == 1
        assert M.nodes[n]['nnodes'] == 2
        assert M.nodes[n]['density'] == 1

def test_directed_path():
    if False:
        return 10
    G = nx.DiGraph()
    nx.add_path(G, range(6))
    partition = [{0, 1}, {2, 3}, {4, 5}]
    M = nx.quotient_graph(G, partition, relabel=True)
    assert nodes_equal(M, [0, 1, 2])
    assert edges_equal(M.edges(), [(0, 1), (1, 2)])
    for n in M:
        assert M.nodes[n]['nedges'] == 1
        assert M.nodes[n]['nnodes'] == 2
        assert M.nodes[n]['density'] == 0.5

def test_directed_multigraph_path():
    if False:
        return 10
    G = nx.MultiDiGraph()
    nx.add_path(G, range(6))
    partition = [{0, 1}, {2, 3}, {4, 5}]
    M = nx.quotient_graph(G, partition, relabel=True)
    assert nodes_equal(M, [0, 1, 2])
    assert edges_equal(M.edges(), [(0, 1), (1, 2)])
    for n in M:
        assert M.nodes[n]['nedges'] == 1
        assert M.nodes[n]['nnodes'] == 2
        assert M.nodes[n]['density'] == 0.5

def test_overlapping_blocks():
    if False:
        print('Hello World!')
    with pytest.raises(nx.NetworkXException):
        G = nx.path_graph(6)
        partition = [{0, 1, 2}, {2, 3}, {4, 5}]
        nx.quotient_graph(G, partition)

def test_weighted_path():
    if False:
        i = 10
        return i + 15
    G = nx.path_graph(6)
    for i in range(5):
        G[i][i + 1]['w'] = i + 1
    partition = [{0, 1}, {2, 3}, {4, 5}]
    M = nx.quotient_graph(G, partition, weight='w', relabel=True)
    assert nodes_equal(M, [0, 1, 2])
    assert edges_equal(M.edges(), [(0, 1), (1, 2)])
    assert M[0][1]['weight'] == 2
    assert M[1][2]['weight'] == 4
    for n in M:
        assert M.nodes[n]['nedges'] == 1
        assert M.nodes[n]['nnodes'] == 2
        assert M.nodes[n]['density'] == 1

def test_barbell():
    if False:
        for i in range(10):
            print('nop')
    G = nx.barbell_graph(3, 0)
    partition = [{0, 1, 2}, {3, 4, 5}]
    M = nx.quotient_graph(G, partition, relabel=True)
    assert nodes_equal(M, [0, 1])
    assert edges_equal(M.edges(), [(0, 1)])
    for n in M:
        assert M.nodes[n]['nedges'] == 3
        assert M.nodes[n]['nnodes'] == 3
        assert M.nodes[n]['density'] == 1

def test_barbell_plus():
    if False:
        i = 10
        return i + 15
    G = nx.barbell_graph(3, 0)
    G.add_edge(0, 5)
    partition = [{0, 1, 2}, {3, 4, 5}]
    M = nx.quotient_graph(G, partition, relabel=True)
    assert nodes_equal(M, [0, 1])
    assert edges_equal(M.edges(), [(0, 1)])
    assert M[0][1]['weight'] == 2
    for n in M:
        assert M.nodes[n]['nedges'] == 3
        assert M.nodes[n]['nnodes'] == 3
        assert M.nodes[n]['density'] == 1

def test_blockmodel():
    if False:
        return 10
    G = nx.path_graph(6)
    partition = [[0, 1], [2, 3], [4, 5]]
    M = nx.quotient_graph(G, partition, relabel=True)
    assert nodes_equal(M.nodes(), [0, 1, 2])
    assert edges_equal(M.edges(), [(0, 1), (1, 2)])
    for n in M.nodes():
        assert M.nodes[n]['nedges'] == 1
        assert M.nodes[n]['nnodes'] == 2
        assert M.nodes[n]['density'] == 1.0

def test_multigraph_blockmodel():
    if False:
        i = 10
        return i + 15
    G = nx.MultiGraph(nx.path_graph(6))
    partition = [[0, 1], [2, 3], [4, 5]]
    M = nx.quotient_graph(G, partition, create_using=nx.MultiGraph(), relabel=True)
    assert nodes_equal(M.nodes(), [0, 1, 2])
    assert edges_equal(M.edges(), [(0, 1), (1, 2)])
    for n in M.nodes():
        assert M.nodes[n]['nedges'] == 1
        assert M.nodes[n]['nnodes'] == 2
        assert M.nodes[n]['density'] == 1.0

def test_quotient_graph_incomplete_partition():
    if False:
        i = 10
        return i + 15
    G = nx.path_graph(6)
    partition = []
    H = nx.quotient_graph(G, partition, relabel=True)
    assert nodes_equal(H.nodes(), [])
    assert edges_equal(H.edges(), [])
    partition = [[0, 1], [2, 3], [5]]
    H = nx.quotient_graph(G, partition, relabel=True)
    assert nodes_equal(H.nodes(), [0, 1, 2])
    assert edges_equal(H.edges(), [(0, 1)])

def test_undirected_node_contraction():
    if False:
        print('Hello World!')
    'Tests for node contraction in an undirected graph.'
    G = nx.cycle_graph(4)
    actual = nx.contracted_nodes(G, 0, 1)
    expected = nx.cycle_graph(3)
    expected.add_edge(0, 0)
    assert nx.is_isomorphic(actual, expected)

def test_directed_node_contraction():
    if False:
        print('Hello World!')
    'Tests for node contraction in a directed graph.'
    G = nx.DiGraph(nx.cycle_graph(4))
    actual = nx.contracted_nodes(G, 0, 1)
    expected = nx.DiGraph(nx.cycle_graph(3))
    expected.add_edge(0, 0)
    expected.add_edge(0, 0)
    assert nx.is_isomorphic(actual, expected)

def test_undirected_node_contraction_no_copy():
    if False:
        return 10
    'Tests for node contraction in an undirected graph\n    by making changes in place.'
    G = nx.cycle_graph(4)
    actual = nx.contracted_nodes(G, 0, 1, copy=False)
    expected = nx.cycle_graph(3)
    expected.add_edge(0, 0)
    assert nx.is_isomorphic(actual, G)
    assert nx.is_isomorphic(actual, expected)

def test_directed_node_contraction_no_copy():
    if False:
        for i in range(10):
            print('nop')
    'Tests for node contraction in a directed graph\n    by making changes in place.'
    G = nx.DiGraph(nx.cycle_graph(4))
    actual = nx.contracted_nodes(G, 0, 1, copy=False)
    expected = nx.DiGraph(nx.cycle_graph(3))
    expected.add_edge(0, 0)
    expected.add_edge(0, 0)
    assert nx.is_isomorphic(actual, G)
    assert nx.is_isomorphic(actual, expected)

def test_create_multigraph():
    if False:
        for i in range(10):
            print('nop')
    'Tests that using a MultiGraph creates multiple edges.'
    G = nx.path_graph(3, create_using=nx.MultiGraph())
    G.add_edge(0, 1)
    G.add_edge(0, 0)
    G.add_edge(0, 2)
    actual = nx.contracted_nodes(G, 0, 2)
    expected = nx.MultiGraph()
    expected.add_edge(0, 1)
    expected.add_edge(0, 1)
    expected.add_edge(0, 1)
    expected.add_edge(0, 0)
    expected.add_edge(0, 0)
    assert edges_equal(actual.edges, expected.edges)

def test_multigraph_keys():
    if False:
        for i in range(10):
            print('nop')
    'Tests that multiedge keys are reset in new graph.'
    G = nx.path_graph(3, create_using=nx.MultiGraph())
    G.add_edge(0, 1, 5)
    G.add_edge(0, 0, 0)
    G.add_edge(0, 2, 5)
    actual = nx.contracted_nodes(G, 0, 2)
    expected = nx.MultiGraph()
    expected.add_edge(0, 1, 0)
    expected.add_edge(0, 1, 5)
    expected.add_edge(0, 1, 2)
    expected.add_edge(0, 0, 0)
    expected.add_edge(0, 0, 1)
    assert edges_equal(actual.edges, expected.edges)

def test_node_attributes():
    if False:
        while True:
            i = 10
    'Tests that node contraction preserves node attributes.'
    G = nx.cycle_graph(4)
    G.nodes[0]['foo'] = 'bar'
    G.nodes[1]['baz'] = 'xyzzy'
    actual = nx.contracted_nodes(G, 0, 1)
    expected = nx.complete_graph(3)
    expected = nx.relabel_nodes(expected, {1: 2, 2: 3})
    expected.add_edge(0, 0)
    cdict = {1: {'baz': 'xyzzy'}}
    expected.nodes[0].update({'foo': 'bar', 'contraction': cdict})
    assert nx.is_isomorphic(actual, expected)
    assert actual.nodes == expected.nodes

def test_edge_attributes():
    if False:
        while True:
            i = 10
    'Tests that node contraction preserves edge attributes.'
    G = nx.DiGraph([('src1', 'dest'), ('src2', 'dest')])
    G['src1']['dest']['value'] = 'src1-->dest'
    G['src2']['dest']['value'] = 'src2-->dest'
    H = nx.MultiDiGraph(G)
    G = nx.contracted_nodes(G, 'src1', 'src2')
    assert G.edges['src1', 'dest']['value'] == 'src1-->dest'
    assert G.edges['src1', 'dest']['contraction']['src2', 'dest']['value'] == 'src2-->dest'
    H = nx.contracted_nodes(H, 'src1', 'src2')
    assert len(H.edges(('src1', 'dest'))) == 2

def test_without_self_loops():
    if False:
        while True:
            i = 10
    'Tests for node contraction without preserving -loops.'
    G = nx.cycle_graph(4)
    actual = nx.contracted_nodes(G, 0, 1, self_loops=False)
    expected = nx.complete_graph(3)
    assert nx.is_isomorphic(actual, expected)

def test_contract_loop_graph():
    if False:
        return 10
    'Tests for node contraction when nodes have loops.'
    G = nx.cycle_graph(4)
    G.add_edge(0, 0)
    actual = nx.contracted_nodes(G, 0, 1)
    expected = nx.complete_graph([0, 2, 3])
    expected.add_edge(0, 0)
    expected.add_edge(0, 0)
    assert edges_equal(actual.edges, expected.edges)
    actual = nx.contracted_nodes(G, 1, 0)
    expected = nx.complete_graph([1, 2, 3])
    expected.add_edge(1, 1)
    expected.add_edge(1, 1)
    assert edges_equal(actual.edges, expected.edges)

def test_undirected_edge_contraction():
    if False:
        for i in range(10):
            print('nop')
    'Tests for edge contraction in an undirected graph.'
    G = nx.cycle_graph(4)
    actual = nx.contracted_edge(G, (0, 1))
    expected = nx.complete_graph(3)
    expected.add_edge(0, 0)
    assert nx.is_isomorphic(actual, expected)

def test_multigraph_edge_contraction():
    if False:
        i = 10
        return i + 15
    'Tests for edge contraction in a multigraph'
    G = nx.cycle_graph(4)
    actual = nx.contracted_edge(G, (0, 1, 0))
    expected = nx.complete_graph(3)
    expected.add_edge(0, 0)
    assert nx.is_isomorphic(actual, expected)

def test_nonexistent_edge():
    if False:
        print('Hello World!')
    'Tests that attempting to contract a nonexistent edge raises an\n    exception.\n\n    '
    with pytest.raises(ValueError):
        G = nx.cycle_graph(4)
        nx.contracted_edge(G, (0, 2))