import pytest
import networkx as nx
from networkx.algorithms.planarity import check_planarity_recursive, get_counterexample, get_counterexample_recursive

class TestLRPlanarity:
    """Nose Unit tests for the :mod:`networkx.algorithms.planarity` module.

    Tests three things:
    1. Check that the result is correct
        (returns planar if and only if the graph is actually planar)
    2. In case a counter example is returned: Check if it is correct
    3. In case an embedding is returned: Check if its actually an embedding
    """

    @staticmethod
    def check_graph(G, is_planar=None):
        if False:
            return 10
        'Raises an exception if the lr_planarity check returns a wrong result\n\n        Parameters\n        ----------\n        G : NetworkX graph\n        is_planar : bool\n            The expected result of the planarity check.\n            If set to None only counter example or embedding are verified.\n\n        '
        (is_planar_lr, result) = nx.check_planarity(G, True)
        (is_planar_lr_rec, result_rec) = check_planarity_recursive(G, True)
        if is_planar is not None:
            if is_planar:
                msg = 'Wrong planarity check result. Should be planar.'
            else:
                msg = 'Wrong planarity check result. Should be non-planar.'
            assert is_planar == is_planar_lr, msg
            assert is_planar == is_planar_lr_rec, msg
        if is_planar_lr:
            check_embedding(G, result)
            check_embedding(G, result_rec)
        else:
            check_counterexample(G, result)
            check_counterexample(G, result_rec)

    def test_simple_planar_graph(self):
        if False:
            return 10
        e = [(1, 2), (2, 3), (3, 4), (4, 6), (6, 7), (7, 1), (1, 5), (5, 2), (2, 4), (4, 5), (5, 7)]
        self.check_graph(nx.Graph(e), is_planar=True)

    def test_planar_with_selfloop(self):
        if False:
            i = 10
            return i + 15
        e = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (1, 2), (1, 3), (1, 5), (2, 5), (2, 4), (3, 4), (3, 5), (4, 5)]
        self.check_graph(nx.Graph(e), is_planar=True)

    def test_k3_3(self):
        if False:
            print('Hello World!')
        self.check_graph(nx.complete_bipartite_graph(3, 3), is_planar=False)

    def test_k5(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_graph(nx.complete_graph(5), is_planar=False)

    def test_multiple_components_planar(self):
        if False:
            print('Hello World!')
        e = [(1, 2), (2, 3), (3, 1), (4, 5), (5, 6), (6, 4)]
        self.check_graph(nx.Graph(e), is_planar=True)

    def test_multiple_components_non_planar(self):
        if False:
            for i in range(10):
                print('nop')
        G = nx.complete_graph(5)
        G.add_edges_from([(6, 7), (7, 8), (8, 6)])
        self.check_graph(G, is_planar=False)

    def test_non_planar_with_selfloop(self):
        if False:
            print('Hello World!')
        G = nx.complete_graph(5)
        for i in range(5):
            G.add_edge(i, i)
        self.check_graph(G, is_planar=False)

    def test_non_planar1(self):
        if False:
            while True:
                i = 10
        e = [(1, 5), (1, 6), (1, 7), (2, 6), (2, 3), (3, 5), (3, 7), (4, 5), (4, 6), (4, 7)]
        self.check_graph(nx.Graph(e), is_planar=False)

    def test_loop(self):
        if False:
            i = 10
            return i + 15
        e = [(1, 2), (2, 2)]
        G = nx.Graph(e)
        self.check_graph(G, is_planar=True)

    def test_comp(self):
        if False:
            for i in range(10):
                print('nop')
        e = [(1, 2), (3, 4)]
        G = nx.Graph(e)
        G.remove_edge(1, 2)
        self.check_graph(G, is_planar=True)

    def test_goldner_harary(self):
        if False:
            while True:
                i = 10
        e = [(1, 2), (1, 3), (1, 4), (1, 5), (1, 7), (1, 8), (1, 10), (1, 11), (2, 3), (2, 4), (2, 6), (2, 7), (2, 9), (2, 10), (2, 11), (3, 4), (4, 5), (4, 6), (4, 7), (5, 7), (6, 7), (7, 8), (7, 9), (7, 10), (8, 10), (9, 10), (10, 11)]
        G = nx.Graph(e)
        self.check_graph(G, is_planar=True)

    def test_planar_multigraph(self):
        if False:
            print('Hello World!')
        G = nx.MultiGraph([(1, 2), (1, 2), (1, 2), (1, 2), (2, 3), (3, 1)])
        self.check_graph(G, is_planar=True)

    def test_non_planar_multigraph(self):
        if False:
            return 10
        G = nx.MultiGraph(nx.complete_graph(5))
        G.add_edges_from([(1, 2)] * 5)
        self.check_graph(G, is_planar=False)

    def test_planar_digraph(self):
        if False:
            for i in range(10):
                print('nop')
        G = nx.DiGraph([(1, 2), (2, 3), (2, 4), (4, 1), (4, 2), (1, 4), (3, 2)])
        self.check_graph(G, is_planar=True)

    def test_non_planar_digraph(self):
        if False:
            i = 10
            return i + 15
        G = nx.DiGraph(nx.complete_graph(5))
        G.remove_edge(1, 2)
        G.remove_edge(4, 1)
        self.check_graph(G, is_planar=False)

    def test_single_component(self):
        if False:
            return 10
        G = nx.Graph()
        G.add_node(1)
        self.check_graph(G, is_planar=True)

    def test_graph1(self):
        if False:
            while True:
                i = 10
        G = nx.Graph([(3, 10), (2, 13), (1, 13), (7, 11), (0, 8), (8, 13), (0, 2), (0, 7), (0, 10), (1, 7)])
        self.check_graph(G, is_planar=True)

    def test_graph2(self):
        if False:
            while True:
                i = 10
        G = nx.Graph([(1, 2), (4, 13), (0, 13), (4, 5), (7, 10), (1, 7), (0, 3), (2, 6), (5, 6), (7, 13), (4, 8), (0, 8), (0, 9), (2, 13), (6, 7), (3, 6), (2, 8)])
        self.check_graph(G, is_planar=False)

    def test_graph3(self):
        if False:
            for i in range(10):
                print('nop')
        G = nx.Graph([(0, 7), (3, 11), (3, 4), (8, 9), (4, 11), (1, 7), (1, 13), (1, 11), (3, 5), (5, 7), (1, 3), (0, 4), (5, 11), (5, 13)])
        self.check_graph(G, is_planar=False)

    def test_counterexample_planar(self):
        if False:
            i = 10
            return i + 15
        with pytest.raises(nx.NetworkXException):
            G = nx.Graph()
            G.add_node(1)
            get_counterexample(G)

    def test_counterexample_planar_recursive(self):
        if False:
            return 10
        with pytest.raises(nx.NetworkXException):
            G = nx.Graph()
            G.add_node(1)
            get_counterexample_recursive(G)

def check_embedding(G, embedding):
    if False:
        return 10
    "Raises an exception if the combinatorial embedding is not correct\n\n    Parameters\n    ----------\n    G : NetworkX graph\n    embedding : a dict mapping nodes to a list of edges\n        This specifies the ordering of the outgoing edges from a node for\n        a combinatorial embedding\n\n    Notes\n    -----\n    Checks the following things:\n        - The type of the embedding is correct\n        - The nodes and edges match the original graph\n        - Every half edge has its matching opposite half edge\n        - No intersections of edges (checked by Euler's formula)\n    "
    if not isinstance(embedding, nx.PlanarEmbedding):
        raise nx.NetworkXException('Bad embedding. Not of type nx.PlanarEmbedding')
    embedding.check_structure()
    assert set(G.nodes) == set(embedding.nodes), "Bad embedding. Nodes don't match the original graph."
    g_edges = set()
    for edge in G.edges:
        if edge[0] != edge[1]:
            g_edges.add((edge[0], edge[1]))
            g_edges.add((edge[1], edge[0]))
    assert g_edges == set(embedding.edges), "Bad embedding. Edges don't match the original graph."

def check_counterexample(G, sub_graph):
    if False:
        print('Hello World!')
    'Raises an exception if the counterexample is wrong.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n    subdivision_nodes : set\n        A set of nodes inducing a subgraph as a counterexample\n    '
    sub_graph = nx.Graph(sub_graph)
    for u in sub_graph:
        if sub_graph.has_edge(u, u):
            sub_graph.remove_edge(u, u)
    contract = list(sub_graph)
    while len(contract) > 0:
        contract_node = contract.pop()
        if contract_node not in sub_graph:
            continue
        degree = sub_graph.degree[contract_node]
        if degree == 2:
            neighbors = iter(sub_graph[contract_node])
            u = next(neighbors)
            v = next(neighbors)
            contract.append(u)
            contract.append(v)
            sub_graph.remove_node(contract_node)
            sub_graph.add_edge(u, v)
    if len(sub_graph) == 5:
        if not nx.is_isomorphic(nx.complete_graph(5), sub_graph):
            raise nx.NetworkXException('Bad counter example.')
    elif len(sub_graph) == 6:
        if not nx.is_isomorphic(nx.complete_bipartite_graph(3, 3), sub_graph):
            raise nx.NetworkXException('Bad counter example.')
    else:
        raise nx.NetworkXException('Bad counter example.')

class TestPlanarEmbeddingClass:

    def test_get_data(self):
        if False:
            i = 10
            return i + 15
        embedding = self.get_star_embedding(3)
        data = embedding.get_data()
        data_cmp = {0: [2, 1], 1: [0], 2: [0]}
        assert data == data_cmp

    def test_missing_edge_orientation(self):
        if False:
            i = 10
            return i + 15
        with pytest.raises(nx.NetworkXException):
            embedding = nx.PlanarEmbedding()
            embedding.add_edge(1, 2)
            embedding.add_edge(2, 1)
            embedding.check_structure()

    def test_invalid_edge_orientation(self):
        if False:
            i = 10
            return i + 15
        with pytest.raises(nx.NetworkXException):
            embedding = nx.PlanarEmbedding()
            embedding.add_half_edge_first(1, 2)
            embedding.add_half_edge_first(2, 1)
            embedding.add_edge(1, 3)
            embedding.check_structure()

    def test_missing_half_edge(self):
        if False:
            for i in range(10):
                print('nop')
        with pytest.raises(nx.NetworkXException):
            embedding = nx.PlanarEmbedding()
            embedding.add_half_edge_first(1, 2)
            embedding.check_structure()

    def test_not_fulfilling_euler_formula(self):
        if False:
            i = 10
            return i + 15
        with pytest.raises(nx.NetworkXException):
            embedding = nx.PlanarEmbedding()
            for i in range(5):
                for j in range(5):
                    if i != j:
                        embedding.add_half_edge_first(i, j)
            embedding.check_structure()

    def test_missing_reference(self):
        if False:
            return 10
        with pytest.raises(nx.NetworkXException):
            embedding = nx.PlanarEmbedding()
            embedding.add_half_edge_cw(1, 2, 3)

    def test_connect_components(self):
        if False:
            return 10
        embedding = nx.PlanarEmbedding()
        embedding.connect_components(1, 2)

    def test_successful_face_traversal(self):
        if False:
            for i in range(10):
                print('nop')
        embedding = nx.PlanarEmbedding()
        embedding.add_half_edge_first(1, 2)
        embedding.add_half_edge_first(2, 1)
        face = embedding.traverse_face(1, 2)
        assert face == [1, 2]

    def test_unsuccessful_face_traversal(self):
        if False:
            return 10
        with pytest.raises(nx.NetworkXException):
            embedding = nx.PlanarEmbedding()
            embedding.add_edge(1, 2, ccw=2, cw=3)
            embedding.add_edge(2, 1, ccw=1, cw=3)
            embedding.traverse_face(1, 2)

    @staticmethod
    def get_star_embedding(n):
        if False:
            while True:
                i = 10
        embedding = nx.PlanarEmbedding()
        for i in range(1, n):
            embedding.add_half_edge_first(0, i)
            embedding.add_half_edge_first(i, 0)
        return embedding