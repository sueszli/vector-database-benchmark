import pytest
import networkx as nx
from networkx.utils import edges_equal

class TestSubGraphView:
    gview = staticmethod(nx.subgraph_view)
    graph = nx.Graph
    hide_edges_filter = staticmethod(nx.filters.hide_edges)
    show_edges_filter = staticmethod(nx.filters.show_edges)

    @classmethod
    def setup_class(cls):
        if False:
            print('Hello World!')
        cls.G = nx.path_graph(9, create_using=cls.graph())
        cls.hide_edges_w_hide_nodes = {(3, 4), (4, 5), (5, 6)}

    def test_hidden_nodes(self):
        if False:
            while True:
                i = 10
        hide_nodes = [4, 5, 111]
        nodes_gone = nx.filters.hide_nodes(hide_nodes)
        gview = self.gview
        G = gview(self.G, filter_node=nodes_gone)
        assert self.G.nodes - G.nodes == {4, 5}
        assert self.G.edges - G.edges == self.hide_edges_w_hide_nodes
        if G.is_directed():
            assert list(G[3]) == []
            assert list(G[2]) == [3]
        else:
            assert list(G[3]) == [2]
            assert set(G[2]) == {1, 3}
        pytest.raises(KeyError, G.__getitem__, 4)
        pytest.raises(KeyError, G.__getitem__, 112)
        pytest.raises(KeyError, G.__getitem__, 111)
        assert G.degree(3) == (3 if G.is_multigraph() else 1)
        assert G.size() == (7 if G.is_multigraph() else 5)

    def test_hidden_edges(self):
        if False:
            while True:
                i = 10
        hide_edges = [(2, 3), (8, 7), (222, 223)]
        edges_gone = self.hide_edges_filter(hide_edges)
        gview = self.gview
        G = gview(self.G, filter_edge=edges_gone)
        assert self.G.nodes == G.nodes
        if G.is_directed():
            assert self.G.edges - G.edges == {(2, 3)}
            assert list(G[2]) == []
            assert list(G.pred[3]) == []
            assert list(G.pred[2]) == [1]
            assert G.size() == 7
        else:
            assert self.G.edges - G.edges == {(2, 3), (7, 8)}
            assert list(G[2]) == [1]
            assert G.size() == 6
        assert list(G[3]) == [4]
        pytest.raises(KeyError, G.__getitem__, 221)
        pytest.raises(KeyError, G.__getitem__, 222)
        assert G.degree(3) == 1

    def test_shown_node(self):
        if False:
            return 10
        induced_subgraph = nx.filters.show_nodes([2, 3, 111])
        gview = self.gview
        G = gview(self.G, filter_node=induced_subgraph)
        assert set(G.nodes) == {2, 3}
        if G.is_directed():
            assert list(G[3]) == []
        else:
            assert list(G[3]) == [2]
        assert list(G[2]) == [3]
        pytest.raises(KeyError, G.__getitem__, 4)
        pytest.raises(KeyError, G.__getitem__, 112)
        pytest.raises(KeyError, G.__getitem__, 111)
        assert G.degree(3) == (3 if G.is_multigraph() else 1)
        assert G.size() == (3 if G.is_multigraph() else 1)

    def test_shown_edges(self):
        if False:
            return 10
        show_edges = [(2, 3), (8, 7), (222, 223)]
        edge_subgraph = self.show_edges_filter(show_edges)
        G = self.gview(self.G, filter_edge=edge_subgraph)
        assert self.G.nodes == G.nodes
        if G.is_directed():
            assert G.edges == {(2, 3)}
            assert list(G[3]) == []
            assert list(G[2]) == [3]
            assert list(G.pred[3]) == [2]
            assert list(G.pred[2]) == []
            assert G.size() == 1
        else:
            assert G.edges == {(2, 3), (7, 8)}
            assert list(G[3]) == [2]
            assert list(G[2]) == [3]
            assert G.size() == 2
        pytest.raises(KeyError, G.__getitem__, 221)
        pytest.raises(KeyError, G.__getitem__, 222)
        assert G.degree(3) == 1

class TestSubDiGraphView(TestSubGraphView):
    gview = staticmethod(nx.subgraph_view)
    graph = nx.DiGraph
    hide_edges_filter = staticmethod(nx.filters.hide_diedges)
    show_edges_filter = staticmethod(nx.filters.show_diedges)
    hide_edges = [(2, 3), (8, 7), (222, 223)]
    excluded = {(2, 3), (3, 4), (4, 5), (5, 6)}

    def test_inoutedges(self):
        if False:
            for i in range(10):
                print('nop')
        edges_gone = self.hide_edges_filter(self.hide_edges)
        hide_nodes = [4, 5, 111]
        nodes_gone = nx.filters.hide_nodes(hide_nodes)
        G = self.gview(self.G, filter_node=nodes_gone, filter_edge=edges_gone)
        assert self.G.in_edges - G.in_edges == self.excluded
        assert self.G.out_edges - G.out_edges == self.excluded

    def test_pred(self):
        if False:
            while True:
                i = 10
        edges_gone = self.hide_edges_filter(self.hide_edges)
        hide_nodes = [4, 5, 111]
        nodes_gone = nx.filters.hide_nodes(hide_nodes)
        G = self.gview(self.G, filter_node=nodes_gone, filter_edge=edges_gone)
        assert list(G.pred[2]) == [1]
        assert list(G.pred[6]) == []

    def test_inout_degree(self):
        if False:
            i = 10
            return i + 15
        edges_gone = self.hide_edges_filter(self.hide_edges)
        hide_nodes = [4, 5, 111]
        nodes_gone = nx.filters.hide_nodes(hide_nodes)
        G = self.gview(self.G, filter_node=nodes_gone, filter_edge=edges_gone)
        assert G.degree(2) == 1
        assert G.out_degree(2) == 0
        assert G.in_degree(2) == 1
        assert G.size() == 4

class TestMultiGraphView(TestSubGraphView):
    gview = staticmethod(nx.subgraph_view)
    graph = nx.MultiGraph
    hide_edges_filter = staticmethod(nx.filters.hide_multiedges)
    show_edges_filter = staticmethod(nx.filters.show_multiedges)

    @classmethod
    def setup_class(cls):
        if False:
            while True:
                i = 10
        cls.G = nx.path_graph(9, create_using=cls.graph())
        multiedges = {(2, 3, 4), (2, 3, 5)}
        cls.G.add_edges_from(multiedges)
        cls.hide_edges_w_hide_nodes = {(3, 4, 0), (4, 5, 0), (5, 6, 0)}

    def test_hidden_edges(self):
        if False:
            i = 10
            return i + 15
        hide_edges = [(2, 3, 4), (2, 3, 3), (8, 7, 0), (222, 223, 0)]
        edges_gone = self.hide_edges_filter(hide_edges)
        G = self.gview(self.G, filter_edge=edges_gone)
        assert self.G.nodes == G.nodes
        if G.is_directed():
            assert self.G.edges - G.edges == {(2, 3, 4)}
            assert list(G[3]) == [4]
            assert list(G[2]) == [3]
            assert list(G.pred[3]) == [2]
            assert list(G.pred[2]) == [1]
            assert G.size() == 9
        else:
            assert self.G.edges - G.edges == {(2, 3, 4), (7, 8, 0)}
            assert list(G[3]) == [2, 4]
            assert list(G[2]) == [1, 3]
            assert G.size() == 8
        assert G.degree(3) == 3
        pytest.raises(KeyError, G.__getitem__, 221)
        pytest.raises(KeyError, G.__getitem__, 222)

    def test_shown_edges(self):
        if False:
            i = 10
            return i + 15
        show_edges = [(2, 3, 4), (2, 3, 3), (8, 7, 0), (222, 223, 0)]
        edge_subgraph = self.show_edges_filter(show_edges)
        G = self.gview(self.G, filter_edge=edge_subgraph)
        assert self.G.nodes == G.nodes
        if G.is_directed():
            assert G.edges == {(2, 3, 4)}
            assert list(G[3]) == []
            assert list(G.pred[3]) == [2]
            assert list(G.pred[2]) == []
            assert G.size() == 1
        else:
            assert G.edges == {(2, 3, 4), (7, 8, 0)}
            assert G.size() == 2
            assert list(G[3]) == [2]
        assert G.degree(3) == 1
        assert list(G[2]) == [3]
        pytest.raises(KeyError, G.__getitem__, 221)
        pytest.raises(KeyError, G.__getitem__, 222)

class TestMultiDiGraphView(TestMultiGraphView, TestSubDiGraphView):
    gview = staticmethod(nx.subgraph_view)
    graph = nx.MultiDiGraph
    hide_edges_filter = staticmethod(nx.filters.hide_multidiedges)
    show_edges_filter = staticmethod(nx.filters.show_multidiedges)
    hide_edges = [(2, 3, 0), (8, 7, 0), (222, 223, 0)]
    excluded = {(2, 3, 0), (3, 4, 0), (4, 5, 0), (5, 6, 0)}

    def test_inout_degree(self):
        if False:
            for i in range(10):
                print('nop')
        edges_gone = self.hide_edges_filter(self.hide_edges)
        hide_nodes = [4, 5, 111]
        nodes_gone = nx.filters.hide_nodes(hide_nodes)
        G = self.gview(self.G, filter_node=nodes_gone, filter_edge=edges_gone)
        assert G.degree(2) == 3
        assert G.out_degree(2) == 2
        assert G.in_degree(2) == 1
        assert G.size() == 6

class TestInducedSubGraph:

    @classmethod
    def setup_class(cls):
        if False:
            return 10
        cls.K3 = G = nx.complete_graph(3)
        G.graph['foo'] = []
        G.nodes[0]['foo'] = []
        G.remove_edge(1, 2)
        ll = []
        G.add_edge(1, 2, foo=ll)
        G.add_edge(2, 1, foo=ll)

    def test_full_graph(self):
        if False:
            for i in range(10):
                print('nop')
        G = self.K3
        H = nx.induced_subgraph(G, [0, 1, 2, 5])
        assert H.name == G.name
        self.graphs_equal(H, G)
        self.same_attrdict(H, G)

    def test_partial_subgraph(self):
        if False:
            i = 10
            return i + 15
        G = self.K3
        H = nx.induced_subgraph(G, 0)
        assert dict(H.adj) == {0: {}}
        assert dict(G.adj) != {0: {}}
        H = nx.induced_subgraph(G, [0, 1])
        assert dict(H.adj) == {0: {1: {}}, 1: {0: {}}}

    def same_attrdict(self, H, G):
        if False:
            while True:
                i = 10
        old_foo = H[1][2]['foo']
        H.edges[1, 2]['foo'] = 'baz'
        assert G.edges == H.edges
        H.edges[1, 2]['foo'] = old_foo
        assert G.edges == H.edges
        old_foo = H.nodes[0]['foo']
        H.nodes[0]['foo'] = 'baz'
        assert G.nodes == H.nodes
        H.nodes[0]['foo'] = old_foo
        assert G.nodes == H.nodes

    def graphs_equal(self, H, G):
        if False:
            while True:
                i = 10
        assert G._adj == H._adj
        assert G._node == H._node
        assert G.graph == H.graph
        assert G.name == H.name
        if not G.is_directed() and (not H.is_directed()):
            assert H._adj[1][2] is H._adj[2][1]
            assert G._adj[1][2] is G._adj[2][1]
        else:
            if not G.is_directed():
                G._pred = G._adj
                G._succ = G._adj
            if not H.is_directed():
                H._pred = H._adj
                H._succ = H._adj
            assert G._pred == H._pred
            assert G._succ == H._succ
            assert H._succ[1][2] is H._pred[2][1]
            assert G._succ[1][2] is G._pred[2][1]

class TestEdgeSubGraph:

    @classmethod
    def setup_class(cls):
        if False:
            for i in range(10):
                print('nop')
        cls.G = G = nx.path_graph(5)
        for i in range(5):
            G.nodes[i]['name'] = f'node{i}'
        G.edges[0, 1]['name'] = 'edge01'
        G.edges[3, 4]['name'] = 'edge34'
        G.graph['name'] = 'graph'
        cls.H = nx.edge_subgraph(G, [(0, 1), (3, 4)])

    def test_correct_nodes(self):
        if False:
            for i in range(10):
                print('nop')
        'Tests that the subgraph has the correct nodes.'
        assert [(0, 'node0'), (1, 'node1'), (3, 'node3'), (4, 'node4')] == sorted(self.H.nodes.data('name'))

    def test_correct_edges(self):
        if False:
            print('Hello World!')
        'Tests that the subgraph has the correct edges.'
        assert edges_equal([(0, 1, 'edge01'), (3, 4, 'edge34')], self.H.edges.data('name'))

    def test_add_node(self):
        if False:
            print('Hello World!')
        'Tests that adding a node to the original graph does not\n        affect the nodes of the subgraph.\n\n        '
        self.G.add_node(5)
        assert [0, 1, 3, 4] == sorted(self.H.nodes)
        self.G.remove_node(5)

    def test_remove_node(self):
        if False:
            i = 10
            return i + 15
        'Tests that removing a node in the original graph\n        removes the nodes of the subgraph.\n\n        '
        self.G.remove_node(0)
        assert [1, 3, 4] == sorted(self.H.nodes)
        self.G.add_node(0, name='node0')
        self.G.add_edge(0, 1, name='edge01')

    def test_node_attr_dict(self):
        if False:
            i = 10
            return i + 15
        'Tests that the node attribute dictionary of the two graphs is\n        the same object.\n\n        '
        for v in self.H:
            assert self.G.nodes[v] == self.H.nodes[v]
        self.G.nodes[0]['name'] = 'foo'
        assert self.G.nodes[0] == self.H.nodes[0]
        self.H.nodes[1]['name'] = 'bar'
        assert self.G.nodes[1] == self.H.nodes[1]
        self.G.nodes[0]['name'] = 'node0'
        self.H.nodes[1]['name'] = 'node1'

    def test_edge_attr_dict(self):
        if False:
            print('Hello World!')
        'Tests that the edge attribute dictionary of the two graphs is\n        the same object.\n\n        '
        for (u, v) in self.H.edges():
            assert self.G.edges[u, v] == self.H.edges[u, v]
        self.G.edges[0, 1]['name'] = 'foo'
        assert self.G.edges[0, 1]['name'] == self.H.edges[0, 1]['name']
        self.H.edges[3, 4]['name'] = 'bar'
        assert self.G.edges[3, 4]['name'] == self.H.edges[3, 4]['name']
        self.G.edges[0, 1]['name'] = 'edge01'
        self.H.edges[3, 4]['name'] = 'edge34'

    def test_graph_attr_dict(self):
        if False:
            i = 10
            return i + 15
        'Tests that the graph attribute dictionary of the two graphs\n        is the same object.\n\n        '
        assert self.G.graph is self.H.graph

    def test_readonly(self):
        if False:
            while True:
                i = 10
        'Tests that the subgraph cannot change the graph structure'
        pytest.raises(nx.NetworkXError, self.H.add_node, 5)
        pytest.raises(nx.NetworkXError, self.H.remove_node, 0)
        pytest.raises(nx.NetworkXError, self.H.add_edge, 5, 6)
        pytest.raises(nx.NetworkXError, self.H.remove_edge, 0, 1)