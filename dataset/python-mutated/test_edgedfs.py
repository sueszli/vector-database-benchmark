import pytest
import networkx as nx
from networkx.algorithms import edge_dfs
from networkx.algorithms.traversal.edgedfs import FORWARD, REVERSE

class TestEdgeDFS:

    @classmethod
    def setup_class(cls):
        if False:
            i = 10
            return i + 15
        cls.nodes = [0, 1, 2, 3]
        cls.edges = [(0, 1), (1, 0), (1, 0), (2, 1), (3, 1)]

    def test_empty(self):
        if False:
            while True:
                i = 10
        G = nx.Graph()
        edges = list(edge_dfs(G))
        assert edges == []

    def test_graph(self):
        if False:
            return 10
        G = nx.Graph(self.edges)
        x = list(edge_dfs(G, self.nodes))
        x_ = [(0, 1), (1, 2), (1, 3)]
        assert x == x_

    def test_digraph(self):
        if False:
            while True:
                i = 10
        G = nx.DiGraph(self.edges)
        x = list(edge_dfs(G, self.nodes))
        x_ = [(0, 1), (1, 0), (2, 1), (3, 1)]
        assert x == x_

    def test_digraph_orientation_invalid(self):
        if False:
            for i in range(10):
                print('nop')
        G = nx.DiGraph(self.edges)
        edge_iterator = edge_dfs(G, self.nodes, orientation='hello')
        pytest.raises(nx.NetworkXError, list, edge_iterator)

    def test_digraph_orientation_none(self):
        if False:
            while True:
                i = 10
        G = nx.DiGraph(self.edges)
        x = list(edge_dfs(G, self.nodes, orientation=None))
        x_ = [(0, 1), (1, 0), (2, 1), (3, 1)]
        assert x == x_

    def test_digraph_orientation_original(self):
        if False:
            for i in range(10):
                print('nop')
        G = nx.DiGraph(self.edges)
        x = list(edge_dfs(G, self.nodes, orientation='original'))
        x_ = [(0, 1, FORWARD), (1, 0, FORWARD), (2, 1, FORWARD), (3, 1, FORWARD)]
        assert x == x_

    def test_digraph2(self):
        if False:
            print('Hello World!')
        G = nx.DiGraph()
        nx.add_path(G, range(4))
        x = list(edge_dfs(G, [0]))
        x_ = [(0, 1), (1, 2), (2, 3)]
        assert x == x_

    def test_digraph_rev(self):
        if False:
            while True:
                i = 10
        G = nx.DiGraph(self.edges)
        x = list(edge_dfs(G, self.nodes, orientation='reverse'))
        x_ = [(1, 0, REVERSE), (0, 1, REVERSE), (2, 1, REVERSE), (3, 1, REVERSE)]
        assert x == x_

    def test_digraph_rev2(self):
        if False:
            while True:
                i = 10
        G = nx.DiGraph()
        nx.add_path(G, range(4))
        x = list(edge_dfs(G, [3], orientation='reverse'))
        x_ = [(2, 3, REVERSE), (1, 2, REVERSE), (0, 1, REVERSE)]
        assert x == x_

    def test_multigraph(self):
        if False:
            return 10
        G = nx.MultiGraph(self.edges)
        x = list(edge_dfs(G, self.nodes))
        x_ = [(0, 1, 0), (1, 0, 1), (0, 1, 2), (1, 2, 0), (1, 3, 0)]
        assert x == x_

    def test_multidigraph(self):
        if False:
            while True:
                i = 10
        G = nx.MultiDiGraph(self.edges)
        x = list(edge_dfs(G, self.nodes))
        x_ = [(0, 1, 0), (1, 0, 0), (1, 0, 1), (2, 1, 0), (3, 1, 0)]
        assert x == x_

    def test_multidigraph_rev(self):
        if False:
            for i in range(10):
                print('nop')
        G = nx.MultiDiGraph(self.edges)
        x = list(edge_dfs(G, self.nodes, orientation='reverse'))
        x_ = [(1, 0, 0, REVERSE), (0, 1, 0, REVERSE), (1, 0, 1, REVERSE), (2, 1, 0, REVERSE), (3, 1, 0, REVERSE)]
        assert x == x_

    def test_digraph_ignore(self):
        if False:
            return 10
        G = nx.DiGraph(self.edges)
        x = list(edge_dfs(G, self.nodes, orientation='ignore'))
        x_ = [(0, 1, FORWARD), (1, 0, FORWARD), (2, 1, REVERSE), (3, 1, REVERSE)]
        assert x == x_

    def test_digraph_ignore2(self):
        if False:
            return 10
        G = nx.DiGraph()
        nx.add_path(G, range(4))
        x = list(edge_dfs(G, [0], orientation='ignore'))
        x_ = [(0, 1, FORWARD), (1, 2, FORWARD), (2, 3, FORWARD)]
        assert x == x_

    def test_multidigraph_ignore(self):
        if False:
            print('Hello World!')
        G = nx.MultiDiGraph(self.edges)
        x = list(edge_dfs(G, self.nodes, orientation='ignore'))
        x_ = [(0, 1, 0, FORWARD), (1, 0, 0, FORWARD), (1, 0, 1, REVERSE), (2, 1, 0, REVERSE), (3, 1, 0, REVERSE)]
        assert x == x_