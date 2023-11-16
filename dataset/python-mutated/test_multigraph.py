from collections import UserDict
import pytest
import networkx as nx
from networkx.utils import edges_equal
from .test_graph import BaseAttrGraphTester
from .test_graph import TestGraph as _TestGraph

class BaseMultiGraphTester(BaseAttrGraphTester):

    def test_has_edge(self):
        if False:
            i = 10
            return i + 15
        G = self.K3
        assert G.has_edge(0, 1)
        assert not G.has_edge(0, -1)
        assert G.has_edge(0, 1, 0)
        assert not G.has_edge(0, 1, 1)

    def test_get_edge_data(self):
        if False:
            return 10
        G = self.K3
        assert G.get_edge_data(0, 1) == {0: {}}
        assert G[0][1] == {0: {}}
        assert G[0][1][0] == {}
        assert G.get_edge_data(10, 20) is None
        assert G.get_edge_data(0, 1, 0) == {}

    def test_adjacency(self):
        if False:
            i = 10
            return i + 15
        G = self.K3
        assert dict(G.adjacency()) == {0: {1: {0: {}}, 2: {0: {}}}, 1: {0: {0: {}}, 2: {0: {}}}, 2: {0: {0: {}}, 1: {0: {}}}}

    def deepcopy_edge_attr(self, H, G):
        if False:
            i = 10
            return i + 15
        assert G[1][2][0]['foo'] == H[1][2][0]['foo']
        G[1][2][0]['foo'].append(1)
        assert G[1][2][0]['foo'] != H[1][2][0]['foo']

    def shallow_copy_edge_attr(self, H, G):
        if False:
            return 10
        assert G[1][2][0]['foo'] == H[1][2][0]['foo']
        G[1][2][0]['foo'].append(1)
        assert G[1][2][0]['foo'] == H[1][2][0]['foo']

    def graphs_equal(self, H, G):
        if False:
            for i in range(10):
                print('nop')
        assert G._adj == H._adj
        assert G._node == H._node
        assert G.graph == H.graph
        assert G.name == H.name
        if not G.is_directed() and (not H.is_directed()):
            assert H._adj[1][2][0] is H._adj[2][1][0]
            assert G._adj[1][2][0] is G._adj[2][1][0]
        else:
            if not G.is_directed():
                G._pred = G._adj
                G._succ = G._adj
            if not H.is_directed():
                H._pred = H._adj
                H._succ = H._adj
            assert G._pred == H._pred
            assert G._succ == H._succ
            assert H._succ[1][2][0] is H._pred[2][1][0]
            assert G._succ[1][2][0] is G._pred[2][1][0]

    def same_attrdict(self, H, G):
        if False:
            while True:
                i = 10
        old_foo = H[1][2][0]['foo']
        H.adj[1][2][0]['foo'] = 'baz'
        assert G._adj == H._adj
        H.adj[1][2][0]['foo'] = old_foo
        assert G._adj == H._adj
        old_foo = H.nodes[0]['foo']
        H.nodes[0]['foo'] = 'baz'
        assert G._node == H._node
        H.nodes[0]['foo'] = old_foo
        assert G._node == H._node

    def different_attrdict(self, H, G):
        if False:
            print('Hello World!')
        old_foo = H[1][2][0]['foo']
        H.adj[1][2][0]['foo'] = 'baz'
        assert G._adj != H._adj
        H.adj[1][2][0]['foo'] = old_foo
        assert G._adj == H._adj
        old_foo = H.nodes[0]['foo']
        H.nodes[0]['foo'] = 'baz'
        assert G._node != H._node
        H.nodes[0]['foo'] = old_foo
        assert G._node == H._node

    def test_to_undirected(self):
        if False:
            print('Hello World!')
        G = self.K3
        self.add_attributes(G)
        H = nx.MultiGraph(G)
        self.is_shallow_copy(H, G)
        H = G.to_undirected()
        self.is_deepcopy(H, G)

    def test_to_directed(self):
        if False:
            for i in range(10):
                print('nop')
        G = self.K3
        self.add_attributes(G)
        H = nx.MultiDiGraph(G)
        self.is_shallow_copy(H, G)
        H = G.to_directed()
        self.is_deepcopy(H, G)

    def test_number_of_edges_selfloops(self):
        if False:
            for i in range(10):
                print('nop')
        G = self.K3
        G.add_edge(0, 0)
        G.add_edge(0, 0)
        G.add_edge(0, 0, key='parallel edge')
        G.remove_edge(0, 0, key='parallel edge')
        assert G.number_of_edges(0, 0) == 2
        G.remove_edge(0, 0)
        assert G.number_of_edges(0, 0) == 1

    def test_edge_lookup(self):
        if False:
            print('Hello World!')
        G = self.Graph()
        G.add_edge(1, 2, foo='bar')
        G.add_edge(1, 2, 'key', foo='biz')
        assert edges_equal(G.edges[1, 2, 0], {'foo': 'bar'})
        assert edges_equal(G.edges[1, 2, 'key'], {'foo': 'biz'})

    def test_edge_attr(self):
        if False:
            while True:
                i = 10
        G = self.Graph()
        G.add_edge(1, 2, key='k1', foo='bar')
        G.add_edge(1, 2, key='k2', foo='baz')
        assert isinstance(G.get_edge_data(1, 2), G.edge_key_dict_factory)
        assert all((isinstance(d, G.edge_attr_dict_factory) for (u, v, d) in G.edges(data=True)))
        assert edges_equal(G.edges(keys=True, data=True), [(1, 2, 'k1', {'foo': 'bar'}), (1, 2, 'k2', {'foo': 'baz'})])
        assert edges_equal(G.edges(keys=True, data='foo'), [(1, 2, 'k1', 'bar'), (1, 2, 'k2', 'baz')])

    def test_edge_attr4(self):
        if False:
            i = 10
            return i + 15
        G = self.Graph()
        G.add_edge(1, 2, key=0, data=7, spam='bar', bar='foo')
        assert edges_equal(G.edges(data=True), [(1, 2, {'data': 7, 'spam': 'bar', 'bar': 'foo'})])
        G[1][2][0]['data'] = 10
        assert edges_equal(G.edges(data=True), [(1, 2, {'data': 10, 'spam': 'bar', 'bar': 'foo'})])
        G.adj[1][2][0]['data'] = 20
        assert edges_equal(G.edges(data=True), [(1, 2, {'data': 20, 'spam': 'bar', 'bar': 'foo'})])
        G.edges[1, 2, 0]['data'] = 21
        assert edges_equal(G.edges(data=True), [(1, 2, {'data': 21, 'spam': 'bar', 'bar': 'foo'})])
        G.adj[1][2][0]['listdata'] = [20, 200]
        G.adj[1][2][0]['weight'] = 20
        assert edges_equal(G.edges(data=True), [(1, 2, {'data': 21, 'spam': 'bar', 'bar': 'foo', 'listdata': [20, 200], 'weight': 20})])

class TestMultiGraph(BaseMultiGraphTester, _TestGraph):

    def setup_method(self):
        if False:
            while True:
                i = 10
        self.Graph = nx.MultiGraph
        (ed1, ed2, ed3) = ({0: {}}, {0: {}}, {0: {}})
        self.k3adj = {0: {1: ed1, 2: ed2}, 1: {0: ed1, 2: ed3}, 2: {0: ed2, 1: ed3}}
        self.k3edges = [(0, 1), (0, 2), (1, 2)]
        self.k3nodes = [0, 1, 2]
        self.K3 = self.Graph()
        self.K3._adj = self.k3adj
        self.K3._node = {}
        self.K3._node[0] = {}
        self.K3._node[1] = {}
        self.K3._node[2] = {}

    def test_data_input(self):
        if False:
            print('Hello World!')
        G = self.Graph({1: [2], 2: [1]}, name='test')
        assert G.name == 'test'
        expected = [(1, {2: {0: {}}}), (2, {1: {0: {}}})]
        assert sorted(G.adj.items()) == expected

    def test_data_multigraph_input(self):
        if False:
            return 10
        edata0 = {'w': 200, 's': 'foo'}
        edata1 = {'w': 201, 's': 'bar'}
        keydict = {0: edata0, 1: edata1}
        dododod = {'a': {'b': keydict}}
        multiple_edge = [('a', 'b', 0, edata0), ('a', 'b', 1, edata1)]
        single_edge = [('a', 'b', 0, keydict)]
        G = self.Graph(dododod, multigraph_input=True)
        assert list(G.edges(keys=True, data=True)) == multiple_edge
        G = self.Graph(dododod, multigraph_input=None)
        assert list(G.edges(keys=True, data=True)) == multiple_edge
        G = self.Graph(dododod, multigraph_input=False)
        assert list(G.edges(keys=True, data=True)) == single_edge
        G = self.Graph(dododod, multigraph_input=True)
        H = self.Graph(nx.to_dict_of_dicts(G))
        assert nx.is_isomorphic(G, H) is True
        for mgi in [True, False]:
            H = self.Graph(nx.to_dict_of_dicts(G), multigraph_input=mgi)
            assert nx.is_isomorphic(G, H) == mgi
    etraits = {'w': 200, 's': 'foo'}
    egraphics = {'color': 'blue', 'shape': 'box'}
    edata = {'traits': etraits, 'graphics': egraphics}
    dodod1 = {'a': {'b': edata}}
    dodod2 = {'a': {'b': etraits}}
    dodod3 = {'a': {'b': {'traits': etraits, 's': 'foo'}}}
    dol = {'a': ['b']}
    multiple_edge = [('a', 'b', 'traits', etraits), ('a', 'b', 'graphics', egraphics)]
    single_edge = [('a', 'b', 0, {})]
    single_edge1 = [('a', 'b', 0, edata)]
    single_edge2 = [('a', 'b', 0, etraits)]
    single_edge3 = [('a', 'b', 0, {'traits': etraits, 's': 'foo'})]
    cases = [(dodod1, True, multiple_edge), (dodod1, False, single_edge1), (dodod2, False, single_edge2), (dodod3, False, single_edge3), (dol, False, single_edge)]

    @pytest.mark.parametrize('dod, mgi, edges', cases)
    def test_non_multigraph_input(self, dod, mgi, edges):
        if False:
            return 10
        G = self.Graph(dod, multigraph_input=mgi)
        assert list(G.edges(keys=True, data=True)) == edges
        G = nx.to_networkx_graph(dod, create_using=self.Graph, multigraph_input=mgi)
        assert list(G.edges(keys=True, data=True)) == edges
    mgi_none_cases = [(dodod1, multiple_edge), (dodod2, single_edge2), (dodod3, single_edge3)]

    @pytest.mark.parametrize('dod, edges', mgi_none_cases)
    def test_non_multigraph_input_mgi_none(self, dod, edges):
        if False:
            return 10
        G = self.Graph(dod)
        assert list(G.edges(keys=True, data=True)) == edges
    raise_cases = [dodod2, dodod3, dol]

    @pytest.mark.parametrize('dod', raise_cases)
    def test_non_multigraph_input_raise(self, dod):
        if False:
            return 10
        pytest.raises(nx.NetworkXError, self.Graph, dod, multigraph_input=True)
        pytest.raises(nx.NetworkXError, nx.to_networkx_graph, dod, create_using=self.Graph, multigraph_input=True)

    def test_getitem(self):
        if False:
            return 10
        G = self.K3
        assert G[0] == {1: {0: {}}, 2: {0: {}}}
        with pytest.raises(KeyError):
            G.__getitem__('j')
        with pytest.raises(TypeError):
            G.__getitem__(['A'])

    def test_remove_node(self):
        if False:
            return 10
        G = self.K3
        G.remove_node(0)
        assert G.adj == {1: {2: {0: {}}}, 2: {1: {0: {}}}}
        with pytest.raises(nx.NetworkXError):
            G.remove_node(-1)

    def test_add_edge(self):
        if False:
            print('Hello World!')
        G = self.Graph()
        G.add_edge(0, 1)
        assert G.adj == {0: {1: {0: {}}}, 1: {0: {0: {}}}}
        G = self.Graph()
        G.add_edge(*(0, 1))
        assert G.adj == {0: {1: {0: {}}}, 1: {0: {0: {}}}}
        G = self.Graph()
        with pytest.raises(ValueError):
            G.add_edge(None, 'anything')

    def test_add_edge_conflicting_key(self):
        if False:
            for i in range(10):
                print('nop')
        G = self.Graph()
        G.add_edge(0, 1, key=1)
        G.add_edge(0, 1)
        assert G.number_of_edges() == 2
        G = self.Graph()
        G.add_edges_from([(0, 1, 1, {})])
        G.add_edges_from([(0, 1)])
        assert G.number_of_edges() == 2

    def test_add_edges_from(self):
        if False:
            while True:
                i = 10
        G = self.Graph()
        G.add_edges_from([(0, 1), (0, 1, {'weight': 3})])
        assert G.adj == {0: {1: {0: {}, 1: {'weight': 3}}}, 1: {0: {0: {}, 1: {'weight': 3}}}}
        G.add_edges_from([(0, 1), (0, 1, {'weight': 3})], weight=2)
        assert G.adj == {0: {1: {0: {}, 1: {'weight': 3}, 2: {'weight': 2}, 3: {'weight': 3}}}, 1: {0: {0: {}, 1: {'weight': 3}, 2: {'weight': 2}, 3: {'weight': 3}}}}
        G = self.Graph()
        edges = [(0, 1, {'weight': 3}), (0, 1, (('weight', 2),)), (0, 1, 5), (0, 1, 's')]
        G.add_edges_from(edges)
        keydict = {0: {'weight': 3}, 1: {'weight': 2}, 5: {}, 's': {}}
        assert G._adj == {0: {1: keydict}, 1: {0: keydict}}
        with pytest.raises(nx.NetworkXError):
            G.add_edges_from([(0,)])
        with pytest.raises(nx.NetworkXError):
            G.add_edges_from([(0, 1, 2, 3, 4)])
        with pytest.raises(TypeError):
            G.add_edges_from([0])

    def test_multigraph_add_edges_from_four_tuple_misordered(self):
        if False:
            print('Hello World!')
        'add_edges_from expects 4-tuples of the format (u, v, key, data_dict).\n\n        Ensure 4-tuples of form (u, v, data_dict, key) raise exception.\n        '
        G = nx.MultiGraph()
        with pytest.raises(TypeError):
            G.add_edges_from([(0, 1, {'color': 'red'}, 0)])

    def test_remove_edge(self):
        if False:
            return 10
        G = self.K3
        G.remove_edge(0, 1)
        assert G.adj == {0: {2: {0: {}}}, 1: {2: {0: {}}}, 2: {0: {0: {}}, 1: {0: {}}}}
        with pytest.raises(nx.NetworkXError):
            G.remove_edge(-1, 0)
        with pytest.raises(nx.NetworkXError):
            G.remove_edge(0, 2, key=1)

    def test_remove_edges_from(self):
        if False:
            for i in range(10):
                print('nop')
        G = self.K3.copy()
        G.remove_edges_from([(0, 1)])
        kd = {0: {}}
        assert G.adj == {0: {2: kd}, 1: {2: kd}, 2: {0: kd, 1: kd}}
        G.remove_edges_from([(0, 0)])
        self.K3.add_edge(0, 1)
        G = self.K3.copy()
        G.remove_edges_from(list(G.edges(data=True, keys=True)))
        assert G.adj == {0: {}, 1: {}, 2: {}}
        G = self.K3.copy()
        G.remove_edges_from(list(G.edges(data=False, keys=True)))
        assert G.adj == {0: {}, 1: {}, 2: {}}
        G = self.K3.copy()
        G.remove_edges_from(list(G.edges(data=False, keys=False)))
        assert G.adj == {0: {}, 1: {}, 2: {}}
        G = self.K3.copy()
        G.remove_edges_from([(0, 1, 0), (0, 2, 0, {}), (1, 2)])
        assert G.adj == {0: {1: {1: {}}}, 1: {0: {1: {}}}, 2: {}}

    def test_remove_multiedge(self):
        if False:
            i = 10
            return i + 15
        G = self.K3
        G.add_edge(0, 1, key='parallel edge')
        G.remove_edge(0, 1, key='parallel edge')
        assert G.adj == {0: {1: {0: {}}, 2: {0: {}}}, 1: {0: {0: {}}, 2: {0: {}}}, 2: {0: {0: {}}, 1: {0: {}}}}
        G.remove_edge(0, 1)
        kd = {0: {}}
        assert G.adj == {0: {2: kd}, 1: {2: kd}, 2: {0: kd, 1: kd}}
        with pytest.raises(nx.NetworkXError):
            G.remove_edge(-1, 0)

class TestEdgeSubgraph:
    """Unit tests for the :meth:`MultiGraph.edge_subgraph` method."""

    def setup_method(self):
        if False:
            while True:
                i = 10
        G = nx.MultiGraph()
        nx.add_path(G, range(5))
        nx.add_path(G, range(5))
        for i in range(5):
            G.nodes[i]['name'] = f'node{i}'
        G.adj[0][1][0]['name'] = 'edge010'
        G.adj[0][1][1]['name'] = 'edge011'
        G.adj[3][4][0]['name'] = 'edge340'
        G.adj[3][4][1]['name'] = 'edge341'
        G.graph['name'] = 'graph'
        self.G = G
        self.H = G.edge_subgraph([(0, 1, 0), (3, 4, 1)])

    def test_correct_nodes(self):
        if False:
            print('Hello World!')
        'Tests that the subgraph has the correct nodes.'
        assert [0, 1, 3, 4] == sorted(self.H.nodes())

    def test_correct_edges(self):
        if False:
            return 10
        'Tests that the subgraph has the correct edges.'
        assert [(0, 1, 0, 'edge010'), (3, 4, 1, 'edge341')] == sorted(self.H.edges(keys=True, data='name'))

    def test_add_node(self):
        if False:
            return 10
        'Tests that adding a node to the original graph does not\n        affect the nodes of the subgraph.\n\n        '
        self.G.add_node(5)
        assert [0, 1, 3, 4] == sorted(self.H.nodes())

    def test_remove_node(self):
        if False:
            while True:
                i = 10
        'Tests that removing a node in the original graph does\n        affect the nodes of the subgraph.\n\n        '
        self.G.remove_node(0)
        assert [1, 3, 4] == sorted(self.H.nodes())

    def test_node_attr_dict(self):
        if False:
            for i in range(10):
                print('nop')
        'Tests that the node attribute dictionary of the two graphs is\n        the same object.\n\n        '
        for v in self.H:
            assert self.G.nodes[v] == self.H.nodes[v]
        self.G.nodes[0]['name'] = 'foo'
        assert self.G.nodes[0] == self.H.nodes[0]
        self.H.nodes[1]['name'] = 'bar'
        assert self.G.nodes[1] == self.H.nodes[1]

    def test_edge_attr_dict(self):
        if False:
            return 10
        'Tests that the edge attribute dictionary of the two graphs is\n        the same object.\n\n        '
        for (u, v, k) in self.H.edges(keys=True):
            assert self.G._adj[u][v][k] == self.H._adj[u][v][k]
        self.G._adj[0][1][0]['name'] = 'foo'
        assert self.G._adj[0][1][0]['name'] == self.H._adj[0][1][0]['name']
        self.H._adj[3][4][1]['name'] = 'bar'
        assert self.G._adj[3][4][1]['name'] == self.H._adj[3][4][1]['name']

    def test_graph_attr_dict(self):
        if False:
            for i in range(10):
                print('nop')
        'Tests that the graph attribute dictionary of the two graphs\n        is the same object.\n\n        '
        assert self.G.graph is self.H.graph

class CustomDictClass(UserDict):
    pass

class MultiGraphSubClass(nx.MultiGraph):
    node_dict_factory = CustomDictClass
    node_attr_dict_factory = CustomDictClass
    adjlist_outer_dict_factory = CustomDictClass
    adjlist_inner_dict_factory = CustomDictClass
    edge_key_dict_factory = CustomDictClass
    edge_attr_dict_factory = CustomDictClass
    graph_attr_dict_factory = CustomDictClass

class TestMultiGraphSubclass(TestMultiGraph):

    def setup_method(self):
        if False:
            while True:
                i = 10
        self.Graph = MultiGraphSubClass
        self.k3edges = [(0, 1), (0, 2), (1, 2)]
        self.k3nodes = [0, 1, 2]
        self.K3 = self.Graph()
        self.K3._adj = self.K3.adjlist_outer_dict_factory({0: self.K3.adjlist_inner_dict_factory(), 1: self.K3.adjlist_inner_dict_factory(), 2: self.K3.adjlist_inner_dict_factory()})
        self.K3._pred = {0: {}, 1: {}, 2: {}}
        for u in self.k3nodes:
            for v in self.k3nodes:
                if u != v:
                    d = {0: {}}
                    self.K3._adj[u][v] = d
                    self.K3._adj[v][u] = d
        self.K3._node = self.K3.node_dict_factory()
        self.K3._node[0] = self.K3.node_attr_dict_factory()
        self.K3._node[1] = self.K3.node_attr_dict_factory()
        self.K3._node[2] = self.K3.node_attr_dict_factory()