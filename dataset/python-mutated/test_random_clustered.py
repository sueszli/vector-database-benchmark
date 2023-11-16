import pytest
import networkx

class TestRandomClusteredGraph:

    def test_valid(self):
        if False:
            print('Hello World!')
        node = [1, 1, 1, 2, 1, 2, 0, 0]
        tri = [0, 0, 0, 0, 0, 1, 1, 1]
        joint_degree_sequence = zip(node, tri)
        G = networkx.random_clustered_graph(joint_degree_sequence)
        assert G.number_of_nodes() == 8
        assert G.number_of_edges() == 7

    def test_valid2(self):
        if False:
            i = 10
            return i + 15
        G = networkx.random_clustered_graph([(1, 2), (2, 1), (1, 1), (1, 1), (1, 1), (2, 0)])
        assert G.number_of_nodes() == 6
        assert G.number_of_edges() == 10

    def test_invalid1(self):
        if False:
            while True:
                i = 10
        pytest.raises((TypeError, networkx.NetworkXError), networkx.random_clustered_graph, [[1, 1], [2, 1], [0, 1]])

    def test_invalid2(self):
        if False:
            i = 10
            return i + 15
        pytest.raises((TypeError, networkx.NetworkXError), networkx.random_clustered_graph, [[1, 1], [1, 2], [0, 1]])