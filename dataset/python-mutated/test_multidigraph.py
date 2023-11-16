from collections import UserDict
import pytest
import networkx as nx
from networkx.utils import edges_equal
from .test_multigraph import BaseMultiGraphTester
from .test_multigraph import TestEdgeSubgraph as _TestMultiGraphEdgeSubgraph
from .test_multigraph import TestMultiGraph as _TestMultiGraph

class BaseMultiDiGraphTester(BaseMultiGraphTester):

    def test_edges(self):
        if False:
            while True:
                i = 10
        G = self.K3
        edges = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
        assert sorted(G.edges()) == edges
        assert sorted(G.edges(0)) == [(0, 1), (0, 2)]
        pytest.raises((KeyError, nx.NetworkXError), G.edges, -1)

    def test_edges_data(self):
        if False:
            i = 10
            return i + 15
        G = self.K3
        edges = [(0, 1, {}), (0, 2, {}), (1, 0, {}), (1, 2, {}), (2, 0, {}), (2, 1, {})]
        assert sorted(G.edges(data=True)) == edges
        assert sorted(G.edges(0, data=True)) == [(0, 1, {}), (0, 2, {})]
        pytest.raises((KeyError, nx.NetworkXError), G.neighbors, -1)

    def test_edges_multi(self):
        if False:
            i = 10
            return i + 15
        G = self.K3
        assert sorted(G.edges()) == [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
        assert sorted(G.edges(0)) == [(0, 1), (0, 2)]
        G.add_edge(0, 1)
        assert sorted(G.edges()) == [(0, 1), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]

    def test_out_edges(self):
        if False:
            for i in range(10):
                print('nop')
        G = self.K3
        assert sorted(G.out_edges()) == [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
        assert sorted(G.out_edges(0)) == [(0, 1), (0, 2)]
        pytest.raises((KeyError, nx.NetworkXError), G.out_edges, -1)
        assert sorted(G.out_edges(0, keys=True)) == [(0, 1, 0), (0, 2, 0)]

    def test_out_edges_multi(self):
        if False:
            while True:
                i = 10
        G = self.K3
        assert sorted(G.out_edges()) == [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
        assert sorted(G.out_edges(0)) == [(0, 1), (0, 2)]
        G.add_edge(0, 1, 2)
        assert sorted(G.out_edges()) == [(0, 1), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]

    def test_out_edges_data(self):
        if False:
            return 10
        G = self.K3
        assert sorted(G.edges(0, data=True)) == [(0, 1, {}), (0, 2, {})]
        G.remove_edge(0, 1)
        G.add_edge(0, 1, data=1)
        assert sorted(G.edges(0, data=True)) == [(0, 1, {'data': 1}), (0, 2, {})]
        assert sorted(G.edges(0, data='data')) == [(0, 1, 1), (0, 2, None)]
        assert sorted(G.edges(0, data='data', default=-1)) == [(0, 1, 1), (0, 2, -1)]

    def test_in_edges(self):
        if False:
            i = 10
            return i + 15
        G = self.K3
        assert sorted(G.in_edges()) == [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
        assert sorted(G.in_edges(0)) == [(1, 0), (2, 0)]
        pytest.raises((KeyError, nx.NetworkXError), G.in_edges, -1)
        G.add_edge(0, 1, 2)
        assert sorted(G.in_edges()) == [(0, 1), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
        assert sorted(G.in_edges(0, keys=True)) == [(1, 0, 0), (2, 0, 0)]

    def test_in_edges_no_keys(self):
        if False:
            print('Hello World!')
        G = self.K3
        assert sorted(G.in_edges()) == [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
        assert sorted(G.in_edges(0)) == [(1, 0), (2, 0)]
        G.add_edge(0, 1, 2)
        assert sorted(G.in_edges()) == [(0, 1), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
        assert sorted(G.in_edges(data=True, keys=False)) == [(0, 1, {}), (0, 1, {}), (0, 2, {}), (1, 0, {}), (1, 2, {}), (2, 0, {}), (2, 1, {})]

    def test_in_edges_data(self):
        if False:
            while True:
                i = 10
        G = self.K3
        assert sorted(G.in_edges(0, data=True)) == [(1, 0, {}), (2, 0, {})]
        G.remove_edge(1, 0)
        G.add_edge(1, 0, data=1)
        assert sorted(G.in_edges(0, data=True)) == [(1, 0, {'data': 1}), (2, 0, {})]
        assert sorted(G.in_edges(0, data='data')) == [(1, 0, 1), (2, 0, None)]
        assert sorted(G.in_edges(0, data='data', default=-1)) == [(1, 0, 1), (2, 0, -1)]

    def is_shallow(self, H, G):
        if False:
            i = 10
            return i + 15
        assert G.graph['foo'] == H.graph['foo']
        G.graph['foo'].append(1)
        assert G.graph['foo'] == H.graph['foo']
        assert G.nodes[0]['foo'] == H.nodes[0]['foo']
        G.nodes[0]['foo'].append(1)
        assert G.nodes[0]['foo'] == H.nodes[0]['foo']
        assert G[1][2][0]['foo'] == H[1][2][0]['foo']
        G[1][2][0]['foo'].append(1)
        assert G[1][2][0]['foo'] == H[1][2][0]['foo']

    def is_deep(self, H, G):
        if False:
            print('Hello World!')
        assert G.graph['foo'] == H.graph['foo']
        G.graph['foo'].append(1)
        assert G.graph['foo'] != H.graph['foo']
        assert G.nodes[0]['foo'] == H.nodes[0]['foo']
        G.nodes[0]['foo'].append(1)
        assert G.nodes[0]['foo'] != H.nodes[0]['foo']
        assert G[1][2][0]['foo'] == H[1][2][0]['foo']
        G[1][2][0]['foo'].append(1)
        assert G[1][2][0]['foo'] != H[1][2][0]['foo']

    def test_to_undirected(self):
        if False:
            return 10
        G = self.K3
        self.add_attributes(G)
        H = nx.MultiGraph(G)
        try:
            assert edges_equal(H.edges(), [(0, 1), (1, 2), (2, 0)])
        except AssertionError:
            assert edges_equal(H.edges(), [(0, 1), (1, 2), (1, 2), (2, 0)])
        H = G.to_undirected()
        self.is_deep(H, G)

    def test_has_successor(self):
        if False:
            print('Hello World!')
        G = self.K3
        assert G.has_successor(0, 1)
        assert not G.has_successor(0, -1)

    def test_successors(self):
        if False:
            for i in range(10):
                print('nop')
        G = self.K3
        assert sorted(G.successors(0)) == [1, 2]
        pytest.raises((KeyError, nx.NetworkXError), G.successors, -1)

    def test_has_predecessor(self):
        if False:
            return 10
        G = self.K3
        assert G.has_predecessor(0, 1)
        assert not G.has_predecessor(0, -1)

    def test_predecessors(self):
        if False:
            print('Hello World!')
        G = self.K3
        assert sorted(G.predecessors(0)) == [1, 2]
        pytest.raises((KeyError, nx.NetworkXError), G.predecessors, -1)

    def test_degree(self):
        if False:
            print('Hello World!')
        G = self.K3
        assert sorted(G.degree()) == [(0, 4), (1, 4), (2, 4)]
        assert dict(G.degree()) == {0: 4, 1: 4, 2: 4}
        assert G.degree(0) == 4
        assert list(G.degree(iter([0]))) == [(0, 4)]
        G.add_edge(0, 1, weight=0.3, other=1.2)
        assert sorted(G.degree(weight='weight')) == [(0, 4.3), (1, 4.3), (2, 4)]
        assert sorted(G.degree(weight='other')) == [(0, 5.2), (1, 5.2), (2, 4)]

    def test_in_degree(self):
        if False:
            print('Hello World!')
        G = self.K3
        assert sorted(G.in_degree()) == [(0, 2), (1, 2), (2, 2)]
        assert dict(G.in_degree()) == {0: 2, 1: 2, 2: 2}
        assert G.in_degree(0) == 2
        assert list(G.in_degree(iter([0]))) == [(0, 2)]
        assert G.in_degree(0, weight='weight') == 2

    def test_out_degree(self):
        if False:
            i = 10
            return i + 15
        G = self.K3
        assert sorted(G.out_degree()) == [(0, 2), (1, 2), (2, 2)]
        assert dict(G.out_degree()) == {0: 2, 1: 2, 2: 2}
        assert G.out_degree(0) == 2
        assert list(G.out_degree(iter([0]))) == [(0, 2)]
        assert G.out_degree(0, weight='weight') == 2

    def test_size(self):
        if False:
            i = 10
            return i + 15
        G = self.K3
        assert G.size() == 6
        assert G.number_of_edges() == 6
        G.add_edge(0, 1, weight=0.3, other=1.2)
        assert round(G.size(weight='weight'), 2) == 6.3
        assert round(G.size(weight='other'), 2) == 7.2

    def test_to_undirected_reciprocal(self):
        if False:
            while True:
                i = 10
        G = self.Graph()
        G.add_edge(1, 2)
        assert G.to_undirected().has_edge(1, 2)
        assert not G.to_undirected(reciprocal=True).has_edge(1, 2)
        G.add_edge(2, 1)
        assert G.to_undirected(reciprocal=True).has_edge(1, 2)

    def test_reverse_copy(self):
        if False:
            while True:
                i = 10
        G = nx.MultiDiGraph([(0, 1), (0, 1)])
        R = G.reverse()
        assert sorted(R.edges()) == [(1, 0), (1, 0)]
        R.remove_edge(1, 0)
        assert sorted(R.edges()) == [(1, 0)]
        assert sorted(G.edges()) == [(0, 1), (0, 1)]

    def test_reverse_nocopy(self):
        if False:
            for i in range(10):
                print('nop')
        G = nx.MultiDiGraph([(0, 1), (0, 1)])
        R = G.reverse(copy=False)
        assert sorted(R.edges()) == [(1, 0), (1, 0)]
        pytest.raises(nx.NetworkXError, R.remove_edge, 1, 0)

    def test_di_attributes_cached(self):
        if False:
            for i in range(10):
                print('nop')
        G = self.K3.copy()
        assert id(G.in_edges) == id(G.in_edges)
        assert id(G.out_edges) == id(G.out_edges)
        assert id(G.in_degree) == id(G.in_degree)
        assert id(G.out_degree) == id(G.out_degree)
        assert id(G.succ) == id(G.succ)
        assert id(G.pred) == id(G.pred)

class TestMultiDiGraph(BaseMultiDiGraphTester, _TestMultiGraph):

    def setup_method(self):
        if False:
            i = 10
            return i + 15
        self.Graph = nx.MultiDiGraph
        self.k3edges = [(0, 1), (0, 2), (1, 2)]
        self.k3nodes = [0, 1, 2]
        self.K3 = self.Graph()
        self.K3._succ = {0: {}, 1: {}, 2: {}}
        self.K3._pred = {0: {}, 1: {}, 2: {}}
        for u in self.k3nodes:
            for v in self.k3nodes:
                if u == v:
                    continue
                d = {0: {}}
                self.K3._succ[u][v] = d
                self.K3._pred[v][u] = d
        self.K3._node = {}
        self.K3._node[0] = {}
        self.K3._node[1] = {}
        self.K3._node[2] = {}

    def test_add_edge(self):
        if False:
            for i in range(10):
                print('nop')
        G = self.Graph()
        G.add_edge(0, 1)
        assert G._adj == {0: {1: {0: {}}}, 1: {}}
        assert G._succ == {0: {1: {0: {}}}, 1: {}}
        assert G._pred == {0: {}, 1: {0: {0: {}}}}
        G = self.Graph()
        G.add_edge(*(0, 1))
        assert G._adj == {0: {1: {0: {}}}, 1: {}}
        assert G._succ == {0: {1: {0: {}}}, 1: {}}
        assert G._pred == {0: {}, 1: {0: {0: {}}}}
        with pytest.raises(ValueError, match='None cannot be a node'):
            G.add_edge(None, 3)

    def test_add_edges_from(self):
        if False:
            i = 10
            return i + 15
        G = self.Graph()
        G.add_edges_from([(0, 1), (0, 1, {'weight': 3})])
        assert G._adj == {0: {1: {0: {}, 1: {'weight': 3}}}, 1: {}}
        assert G._succ == {0: {1: {0: {}, 1: {'weight': 3}}}, 1: {}}
        assert G._pred == {0: {}, 1: {0: {0: {}, 1: {'weight': 3}}}}
        G.add_edges_from([(0, 1), (0, 1, {'weight': 3})], weight=2)
        assert G._succ == {0: {1: {0: {}, 1: {'weight': 3}, 2: {'weight': 2}, 3: {'weight': 3}}}, 1: {}}
        assert G._pred == {0: {}, 1: {0: {0: {}, 1: {'weight': 3}, 2: {'weight': 2}, 3: {'weight': 3}}}}
        G = self.Graph()
        edges = [(0, 1, {'weight': 3}), (0, 1, (('weight', 2),)), (0, 1, 5), (0, 1, 's')]
        G.add_edges_from(edges)
        keydict = {0: {'weight': 3}, 1: {'weight': 2}, 5: {}, 's': {}}
        assert G._succ == {0: {1: keydict}, 1: {}}
        assert G._pred == {1: {0: keydict}, 0: {}}
        pytest.raises(nx.NetworkXError, G.add_edges_from, [(0,)])
        pytest.raises(nx.NetworkXError, G.add_edges_from, [(0, 1, 2, 3, 4)])
        pytest.raises(TypeError, G.add_edges_from, [0])
        with pytest.raises(ValueError, match='None cannot be a node'):
            G.add_edges_from([(None, 3), (3, 2)])

    def test_remove_edge(self):
        if False:
            for i in range(10):
                print('nop')
        G = self.K3
        G.remove_edge(0, 1)
        assert G._succ == {0: {2: {0: {}}}, 1: {0: {0: {}}, 2: {0: {}}}, 2: {0: {0: {}}, 1: {0: {}}}}
        assert G._pred == {0: {1: {0: {}}, 2: {0: {}}}, 1: {2: {0: {}}}, 2: {0: {0: {}}, 1: {0: {}}}}
        pytest.raises((KeyError, nx.NetworkXError), G.remove_edge, -1, 0)
        pytest.raises((KeyError, nx.NetworkXError), G.remove_edge, 0, 2, key=1)

    def test_remove_multiedge(self):
        if False:
            print('Hello World!')
        G = self.K3
        G.add_edge(0, 1, key='parallel edge')
        G.remove_edge(0, 1, key='parallel edge')
        assert G._adj == {0: {1: {0: {}}, 2: {0: {}}}, 1: {0: {0: {}}, 2: {0: {}}}, 2: {0: {0: {}}, 1: {0: {}}}}
        assert G._succ == {0: {1: {0: {}}, 2: {0: {}}}, 1: {0: {0: {}}, 2: {0: {}}}, 2: {0: {0: {}}, 1: {0: {}}}}
        assert G._pred == {0: {1: {0: {}}, 2: {0: {}}}, 1: {0: {0: {}}, 2: {0: {}}}, 2: {0: {0: {}}, 1: {0: {}}}}
        G.remove_edge(0, 1)
        assert G._succ == {0: {2: {0: {}}}, 1: {0: {0: {}}, 2: {0: {}}}, 2: {0: {0: {}}, 1: {0: {}}}}
        assert G._pred == {0: {1: {0: {}}, 2: {0: {}}}, 1: {2: {0: {}}}, 2: {0: {0: {}}, 1: {0: {}}}}
        pytest.raises((KeyError, nx.NetworkXError), G.remove_edge, -1, 0)

    def test_remove_edges_from(self):
        if False:
            return 10
        G = self.K3
        G.remove_edges_from([(0, 1)])
        assert G._succ == {0: {2: {0: {}}}, 1: {0: {0: {}}, 2: {0: {}}}, 2: {0: {0: {}}, 1: {0: {}}}}
        assert G._pred == {0: {1: {0: {}}, 2: {0: {}}}, 1: {2: {0: {}}}, 2: {0: {0: {}}, 1: {0: {}}}}
        G.remove_edges_from([(0, 0)])

class TestEdgeSubgraph(_TestMultiGraphEdgeSubgraph):
    """Unit tests for the :meth:`MultiDiGraph.edge_subgraph` method."""

    def setup_method(self):
        if False:
            i = 10
            return i + 15
        G = nx.MultiDiGraph()
        nx.add_path(G, range(5))
        nx.add_path(G, range(5))
        nx.add_path(G, reversed(range(5)))
        nx.add_path(G, reversed(range(5)))
        for i in range(5):
            G.nodes[i]['name'] = f'node{i}'
        G.adj[0][1][0]['name'] = 'edge010'
        G.adj[0][1][1]['name'] = 'edge011'
        G.adj[3][4][0]['name'] = 'edge340'
        G.adj[3][4][1]['name'] = 'edge341'
        G.graph['name'] = 'graph'
        self.G = G
        self.H = G.edge_subgraph([(0, 1, 0), (3, 4, 1)])

class CustomDictClass(UserDict):
    pass

class MultiDiGraphSubClass(nx.MultiDiGraph):
    node_dict_factory = CustomDictClass
    node_attr_dict_factory = CustomDictClass
    adjlist_outer_dict_factory = CustomDictClass
    adjlist_inner_dict_factory = CustomDictClass
    edge_key_dict_factory = CustomDictClass
    edge_attr_dict_factory = CustomDictClass
    graph_attr_dict_factory = CustomDictClass

class TestMultiDiGraphSubclass(TestMultiDiGraph):

    def setup_method(self):
        if False:
            i = 10
            return i + 15
        self.Graph = MultiDiGraphSubClass
        self.k3edges = [(0, 1), (0, 2), (1, 2)]
        self.k3nodes = [0, 1, 2]
        self.K3 = self.Graph()
        self.K3._succ = self.K3.adjlist_outer_dict_factory({0: self.K3.adjlist_inner_dict_factory(), 1: self.K3.adjlist_inner_dict_factory(), 2: self.K3.adjlist_inner_dict_factory()})
        self.K3._pred = {0: {}, 1: {}, 2: {}}
        for u in self.k3nodes:
            for v in self.k3nodes:
                if u == v:
                    continue
                d = {0: {}}
                self.K3._succ[u][v] = d
                self.K3._pred[v][u] = d
        self.K3._node = self.K3.node_dict_factory()
        self.K3._node[0] = self.K3.node_attr_dict_factory()
        self.K3._node[1] = self.K3.node_attr_dict_factory()
        self.K3._node[2] = self.K3.node_attr_dict_factory()