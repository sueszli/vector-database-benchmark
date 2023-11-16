import pytest
np = pytest.importorskip('numpy')
import networkx as nx

def test_attr_matrix():
    if False:
        while True:
            i = 10
    G = nx.Graph()
    G.add_edge(0, 1, thickness=1, weight=3)
    G.add_edge(0, 1, thickness=1, weight=3)
    G.add_edge(0, 2, thickness=2)
    G.add_edge(1, 2, thickness=3)

    def node_attr(u):
        if False:
            while True:
                i = 10
        return G.nodes[u].get('size', 0.5) * 3

    def edge_attr(u, v):
        if False:
            while True:
                i = 10
        return G[u][v].get('thickness', 0.5)
    M = nx.attr_matrix(G, edge_attr=edge_attr, node_attr=node_attr)
    np.testing.assert_equal(M[0], np.array([[6.0]]))
    assert M[1] == [1.5]

def test_attr_matrix_directed():
    if False:
        while True:
            i = 10
    G = nx.DiGraph()
    G.add_edge(0, 1, thickness=1, weight=3)
    G.add_edge(0, 1, thickness=1, weight=3)
    G.add_edge(0, 2, thickness=2)
    G.add_edge(1, 2, thickness=3)
    M = nx.attr_matrix(G, rc_order=[0, 1, 2])
    data = np.array([[0.0, 1.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
    np.testing.assert_equal(M, np.array(data))

def test_attr_matrix_multigraph():
    if False:
        print('Hello World!')
    G = nx.MultiGraph()
    G.add_edge(0, 1, thickness=1, weight=3)
    G.add_edge(0, 1, thickness=1, weight=3)
    G.add_edge(0, 1, thickness=1, weight=3)
    G.add_edge(0, 2, thickness=2)
    G.add_edge(1, 2, thickness=3)
    M = nx.attr_matrix(G, rc_order=[0, 1, 2])
    data = np.array([[0.0, 3.0, 1.0], [3.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    np.testing.assert_equal(M, np.array(data))
    M = nx.attr_matrix(G, edge_attr='weight', rc_order=[0, 1, 2])
    data = np.array([[0.0, 9.0, 1.0], [9.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    np.testing.assert_equal(M, np.array(data))
    M = nx.attr_matrix(G, edge_attr='thickness', rc_order=[0, 1, 2])
    data = np.array([[0.0, 3.0, 2.0], [3.0, 0.0, 3.0], [2.0, 3.0, 0.0]])
    np.testing.assert_equal(M, np.array(data))

def test_attr_sparse_matrix():
    if False:
        while True:
            i = 10
    pytest.importorskip('scipy')
    G = nx.Graph()
    G.add_edge(0, 1, thickness=1, weight=3)
    G.add_edge(0, 2, thickness=2)
    G.add_edge(1, 2, thickness=3)
    M = nx.attr_sparse_matrix(G)
    mtx = M[0]
    data = np.ones((3, 3), float)
    np.fill_diagonal(data, 0)
    np.testing.assert_equal(mtx.todense(), np.array(data))
    assert M[1] == [0, 1, 2]

def test_attr_sparse_matrix_directed():
    if False:
        i = 10
        return i + 15
    pytest.importorskip('scipy')
    G = nx.DiGraph()
    G.add_edge(0, 1, thickness=1, weight=3)
    G.add_edge(0, 1, thickness=1, weight=3)
    G.add_edge(0, 2, thickness=2)
    G.add_edge(1, 2, thickness=3)
    M = nx.attr_sparse_matrix(G, rc_order=[0, 1, 2])
    data = np.array([[0.0, 1.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
    np.testing.assert_equal(M.todense(), np.array(data))