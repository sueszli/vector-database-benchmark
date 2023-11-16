import pytest
import networkx as nx

class TestReciprocity:

    def test_reciprocity_digraph(self):
        if False:
            while True:
                i = 10
        DG = nx.DiGraph([(1, 2), (2, 1)])
        reciprocity = nx.reciprocity(DG)
        assert reciprocity == 1.0

    def test_overall_reciprocity_empty_graph(self):
        if False:
            for i in range(10):
                print('nop')
        with pytest.raises(nx.NetworkXError):
            DG = nx.DiGraph()
            nx.overall_reciprocity(DG)

    def test_reciprocity_graph_nodes(self):
        if False:
            while True:
                i = 10
        DG = nx.DiGraph([(1, 2), (2, 3), (3, 2)])
        reciprocity = nx.reciprocity(DG, [1, 2])
        expected_reciprocity = {1: 0.0, 2: 0.6666666666666666}
        assert reciprocity == expected_reciprocity

    def test_reciprocity_graph_node(self):
        if False:
            print('Hello World!')
        DG = nx.DiGraph([(1, 2), (2, 3), (3, 2)])
        reciprocity = nx.reciprocity(DG, 2)
        assert reciprocity == 0.6666666666666666

    def test_reciprocity_graph_isolated_nodes(self):
        if False:
            i = 10
            return i + 15
        with pytest.raises(nx.NetworkXError):
            DG = nx.DiGraph([(1, 2)])
            DG.add_node(4)
            nx.reciprocity(DG, 4)