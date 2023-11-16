"""Unit tests for the :mod:`networkx.algorithms.efficiency` module."""
import networkx as nx

class TestEfficiency:

    def setup_method(self):
        if False:
            return 10
        self.G1 = nx.Graph()
        self.G1.add_nodes_from([1, 2, 3])
        self.G2 = nx.cycle_graph(4)
        self.G3 = nx.lollipop_graph(3, 1)

    def test_efficiency_disconnected_nodes(self):
        if False:
            while True:
                i = 10
        '\n        When nodes are disconnected, efficiency is 0\n        '
        assert nx.efficiency(self.G1, 1, 2) == 0

    def test_local_efficiency_disconnected_graph(self):
        if False:
            print('Hello World!')
        '\n        In a disconnected graph the efficiency is 0\n        '
        assert nx.local_efficiency(self.G1) == 0

    def test_efficiency(self):
        if False:
            print('Hello World!')
        assert nx.efficiency(self.G2, 0, 1) == 1
        assert nx.efficiency(self.G2, 0, 2) == 1 / 2

    def test_global_efficiency(self):
        if False:
            return 10
        assert nx.global_efficiency(self.G2) == 5 / 6

    def test_global_efficiency_complete_graph(self):
        if False:
            print('Hello World!')
        '\n        Tests that the average global efficiency of the complete graph is one.\n        '
        for n in range(2, 10):
            G = nx.complete_graph(n)
            assert nx.global_efficiency(G) == 1

    def test_local_efficiency_complete_graph(self):
        if False:
            i = 10
            return i + 15
        '\n        Test that the local efficiency for a complete graph with at least 3\n        nodes should be one. For a graph with only 2 nodes, the induced\n        subgraph has no edges.\n        '
        for n in range(3, 10):
            G = nx.complete_graph(n)
            assert nx.local_efficiency(G) == 1

    def test_using_ego_graph(self):
        if False:
            print('Hello World!')
        '\n        Test that the ego graph is used when computing local efficiency.\n        For more information, see GitHub issue #2710.\n        '
        assert nx.local_efficiency(self.G3) == 7 / 12