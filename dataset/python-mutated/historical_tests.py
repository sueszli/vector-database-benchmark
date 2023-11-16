"""Original NetworkX graph tests"""
import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.utils import edges_equal, nodes_equal

class HistoricalTests:

    @classmethod
    def setup_class(cls):
        if False:
            i = 10
            return i + 15
        cls.null = nx.null_graph()
        cls.P1 = cnlti(nx.path_graph(1), first_label=1)
        cls.P3 = cnlti(nx.path_graph(3), first_label=1)
        cls.P10 = cnlti(nx.path_graph(10), first_label=1)
        cls.K1 = cnlti(nx.complete_graph(1), first_label=1)
        cls.K3 = cnlti(nx.complete_graph(3), first_label=1)
        cls.K4 = cnlti(nx.complete_graph(4), first_label=1)
        cls.K5 = cnlti(nx.complete_graph(5), first_label=1)
        cls.K10 = cnlti(nx.complete_graph(10), first_label=1)
        cls.G = nx.Graph

    def test_name(self):
        if False:
            print('Hello World!')
        G = self.G(name='test')
        assert G.name == 'test'
        H = self.G()
        assert H.name == ''

    def test_add_remove_node(self):
        if False:
            i = 10
            return i + 15
        G = self.G()
        G.add_node('A')
        assert G.has_node('A')
        G.remove_node('A')
        assert not G.has_node('A')

    def test_nonhashable_node(self):
        if False:
            for i in range(10):
                print('nop')
        G = self.G()
        assert not G.has_node(['A'])
        assert not G.has_node({'A': 1})

    def test_add_nodes_from(self):
        if False:
            while True:
                i = 10
        G = self.G()
        G.add_nodes_from(list('ABCDEFGHIJKL'))
        assert G.has_node('L')
        G.remove_nodes_from(['H', 'I', 'J', 'K', 'L'])
        G.add_nodes_from([1, 2, 3, 4])
        assert sorted(G.nodes(), key=str) == [1, 2, 3, 4, 'A', 'B', 'C', 'D', 'E', 'F', 'G']
        assert sorted(G, key=str) == [1, 2, 3, 4, 'A', 'B', 'C', 'D', 'E', 'F', 'G']

    def test_contains(self):
        if False:
            for i in range(10):
                print('nop')
        G = self.G()
        G.add_node('A')
        assert 'A' in G
        assert [] not in G
        assert {1: 1} not in G

    def test_add_remove(self):
        if False:
            print('Hello World!')
        G = self.G()
        G.add_node('m')
        assert G.has_node('m')
        G.add_node('m')
        pytest.raises(nx.NetworkXError, G.remove_node, 'j')
        G.remove_node('m')
        assert list(G) == []

    def test_nbunch_is_list(self):
        if False:
            while True:
                i = 10
        G = self.G()
        G.add_nodes_from(list('ABCD'))
        G.add_nodes_from(self.P3)
        assert sorted(G.nodes(), key=str) == [1, 2, 3, 'A', 'B', 'C', 'D']
        G.remove_nodes_from(self.P3)
        assert sorted(G.nodes(), key=str) == ['A', 'B', 'C', 'D']

    def test_nbunch_is_set(self):
        if False:
            print('Hello World!')
        G = self.G()
        nbunch = set('ABCDEFGHIJKL')
        G.add_nodes_from(nbunch)
        assert G.has_node('L')

    def test_nbunch_dict(self):
        if False:
            print('Hello World!')
        G = self.G()
        nbunch = set('ABCDEFGHIJKL')
        G.add_nodes_from(nbunch)
        nbunch = {'I': 'foo', 'J': 2, 'K': True, 'L': 'spam'}
        G.remove_nodes_from(nbunch)
        assert sorted(G.nodes(), key=str), ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

    def test_nbunch_iterator(self):
        if False:
            i = 10
            return i + 15
        G = self.G()
        G.add_nodes_from(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'])
        n_iter = self.P3.nodes()
        G.add_nodes_from(n_iter)
        assert sorted(G.nodes(), key=str) == [1, 2, 3, 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        n_iter = self.P3.nodes()
        G.remove_nodes_from(n_iter)
        assert sorted(G.nodes(), key=str) == ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

    def test_nbunch_graph(self):
        if False:
            for i in range(10):
                print('nop')
        G = self.G()
        G.add_nodes_from(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'])
        nbunch = self.K3
        G.add_nodes_from(nbunch)
        assert sorted(G.nodes(), key=str), [1, 2, 3, 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

    def test_add_edge(self):
        if False:
            i = 10
            return i + 15
        G = self.G()
        pytest.raises(TypeError, G.add_edge, 'A')
        G.add_edge('A', 'B')
        G.add_edge('A', 'B')
        assert G.has_edge('A', 'B')
        assert not G.has_edge('A', 'C')
        assert G.has_edge(*('A', 'B'))
        if G.is_directed():
            assert not G.has_edge('B', 'A')
        else:
            assert G.has_edge('B', 'A')
        G.add_edge('A', 'C')
        G.add_edge('C', 'A')
        G.remove_edge('C', 'A')
        if G.is_directed():
            assert G.has_edge('A', 'C')
        else:
            assert not G.has_edge('A', 'C')
        assert not G.has_edge('C', 'A')

    def test_self_loop(self):
        if False:
            print('Hello World!')
        G = self.G()
        G.add_edge('A', 'A')
        assert G.has_edge('A', 'A')
        G.remove_edge('A', 'A')
        G.add_edge('X', 'X')
        assert G.has_node('X')
        G.remove_node('X')
        G.add_edge('A', 'Z')
        assert G.has_node('Z')

    def test_add_edges_from(self):
        if False:
            for i in range(10):
                print('nop')
        G = self.G()
        G.add_edges_from([('B', 'C')])
        assert G.has_edge('B', 'C')
        if G.is_directed():
            assert not G.has_edge('C', 'B')
        else:
            assert G.has_edge('C', 'B')
        G.add_edges_from([('D', 'F'), ('B', 'D')])
        assert G.has_edge('D', 'F')
        assert G.has_edge('B', 'D')
        if G.is_directed():
            assert not G.has_edge('D', 'B')
        else:
            assert G.has_edge('D', 'B')

    def test_add_edges_from2(self):
        if False:
            i = 10
            return i + 15
        G = self.G()
        G.add_edges_from([tuple('IJ'), list('KK'), tuple('JK')])
        assert G.has_edge(*('I', 'J'))
        assert G.has_edge(*('K', 'K'))
        assert G.has_edge(*('J', 'K'))
        if G.is_directed():
            assert not G.has_edge(*('K', 'J'))
        else:
            assert G.has_edge(*('K', 'J'))

    def test_add_edges_from3(self):
        if False:
            for i in range(10):
                print('nop')
        G = self.G()
        G.add_edges_from(zip(list('ACD'), list('CDE')))
        assert G.has_edge('D', 'E')
        assert not G.has_edge('E', 'C')

    def test_remove_edge(self):
        if False:
            print('Hello World!')
        G = self.G()
        G.add_nodes_from([1, 2, 3, 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'])
        G.add_edges_from(zip(list('MNOP'), list('NOPM')))
        assert G.has_edge('O', 'P')
        assert G.has_edge('P', 'M')
        G.remove_node('P')
        assert not G.has_edge('P', 'M')
        pytest.raises(TypeError, G.remove_edge, 'M')
        G.add_edge('N', 'M')
        assert G.has_edge('M', 'N')
        G.remove_edge('M', 'N')
        assert not G.has_edge('M', 'N')
        G.remove_edges_from([list('HI'), list('DF'), tuple('KK'), tuple('JK')])
        assert not G.has_edge('H', 'I')
        assert not G.has_edge('J', 'K')
        G.remove_edges_from([list('IJ'), list('KK'), list('JK')])
        assert not G.has_edge('I', 'J')
        G.remove_nodes_from(set('ZEFHIMNO'))
        G.add_edge('J', 'K')

    def test_edges_nbunch(self):
        if False:
            while True:
                i = 10
        G = self.G()
        G.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'D'), ('C', 'B'), ('C', 'D')])
        pytest.raises(nx.NetworkXError, G.edges, 6)
        assert list(G.edges('Z')) == []
        assert list(G.edges([])) == []
        if G.is_directed():
            elist = [('A', 'B'), ('A', 'C'), ('B', 'D')]
        else:
            elist = [('A', 'B'), ('A', 'C'), ('B', 'C'), ('B', 'D')]
        assert edges_equal(list(G.edges(['A', 'B'])), elist)
        assert edges_equal(G.edges({'A', 'B'}), elist)
        G1 = self.G()
        G1.add_nodes_from('AB')
        assert edges_equal(G.edges(G1), elist)
        ndict = {'A': 'thing1', 'B': 'thing2'}
        assert edges_equal(G.edges(ndict), elist)
        assert edges_equal(list(G.edges('A')), [('A', 'B'), ('A', 'C')])
        assert nodes_equal(sorted(G), ['A', 'B', 'C', 'D'])
        assert edges_equal(list(G.edges()), [('A', 'B'), ('A', 'C'), ('B', 'D'), ('C', 'B'), ('C', 'D')])

    def test_degree(self):
        if False:
            for i in range(10):
                print('nop')
        G = self.G()
        G.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'D'), ('C', 'B'), ('C', 'D')])
        assert G.degree('A') == 2
        assert list(G.degree(['A'])) == [('A', 2)]
        assert sorted((d for (n, d) in G.degree(['A', 'B']))) == [2, 3]
        assert sorted((d for (n, d) in G.degree())) == [2, 2, 3, 3]

    def test_degree2(self):
        if False:
            print('Hello World!')
        H = self.G()
        H.add_edges_from([(1, 24), (1, 2)])
        assert sorted((d for (n, d) in H.degree([1, 24]))) == [1, 2]

    def test_degree_graph(self):
        if False:
            return 10
        P3 = nx.path_graph(3)
        P5 = nx.path_graph(5)
        assert dict((d for (n, d) in P3.degree(['A', 'B']))) == {}
        assert sorted((d for (n, d) in P5.degree(P3))) == [1, 2, 2]
        assert sorted((d for (n, d) in P3.degree(P5))) == [1, 1, 2]
        assert list(P5.degree([])) == []
        assert dict(P5.degree([])) == {}

    def test_null(self):
        if False:
            print('Hello World!')
        null = nx.null_graph()
        assert list(null.degree()) == []
        assert dict(null.degree()) == {}

    def test_order_size(self):
        if False:
            print('Hello World!')
        G = self.G()
        G.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'D'), ('C', 'B'), ('C', 'D')])
        assert G.order() == 4
        assert G.size() == 5
        assert G.number_of_edges() == 5
        assert G.number_of_edges('A', 'B') == 1
        assert G.number_of_edges('A', 'D') == 0

    def test_copy(self):
        if False:
            while True:
                i = 10
        G = self.G()
        H = G.copy()
        assert H.adj == G.adj
        assert H.name == G.name
        assert H is not G

    def test_subgraph(self):
        if False:
            print('Hello World!')
        G = self.G()
        G.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'D'), ('C', 'B'), ('C', 'D')])
        SG = G.subgraph(['A', 'B', 'D'])
        assert nodes_equal(list(SG), ['A', 'B', 'D'])
        assert edges_equal(list(SG.edges()), [('A', 'B'), ('B', 'D')])

    def test_to_directed(self):
        if False:
            return 10
        G = self.G()
        if not G.is_directed():
            G.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'D'), ('C', 'B'), ('C', 'D')])
            DG = G.to_directed()
            assert DG is not G
            assert DG.is_directed()
            assert DG.name == G.name
            assert DG.adj == G.adj
            assert sorted(DG.out_edges(list('AB'))) == [('A', 'B'), ('A', 'C'), ('B', 'A'), ('B', 'C'), ('B', 'D')]
            DG.remove_edge('A', 'B')
            assert DG.has_edge('B', 'A')
            assert not DG.has_edge('A', 'B')

    def test_to_undirected(self):
        if False:
            print('Hello World!')
        G = self.G()
        if G.is_directed():
            G.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'D'), ('C', 'B'), ('C', 'D')])
            UG = G.to_undirected()
            assert UG is not G
            assert not UG.is_directed()
            assert G.is_directed()
            assert UG.name == G.name
            assert UG.adj != G.adj
            assert sorted(UG.edges(list('AB'))) == [('A', 'B'), ('A', 'C'), ('B', 'C'), ('B', 'D')]
            assert sorted(UG.edges(['A', 'B'])) == [('A', 'B'), ('A', 'C'), ('B', 'C'), ('B', 'D')]
            UG.remove_edge('A', 'B')
            assert not UG.has_edge('B', 'A')
            assert not UG.has_edge('A', 'B')

    def test_neighbors(self):
        if False:
            while True:
                i = 10
        G = self.G()
        G.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'D'), ('C', 'B'), ('C', 'D')])
        G.add_nodes_from('GJK')
        assert sorted(G['A']) == ['B', 'C']
        assert sorted(G.neighbors('A')) == ['B', 'C']
        assert sorted(G.neighbors('A')) == ['B', 'C']
        assert sorted(G.neighbors('G')) == []
        pytest.raises(nx.NetworkXError, G.neighbors, 'j')

    def test_iterators(self):
        if False:
            for i in range(10):
                print('nop')
        G = self.G()
        G.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'D'), ('C', 'B'), ('C', 'D')])
        G.add_nodes_from('GJK')
        assert sorted(G.nodes()) == ['A', 'B', 'C', 'D', 'G', 'J', 'K']
        assert edges_equal(G.edges(), [('A', 'B'), ('A', 'C'), ('B', 'D'), ('C', 'B'), ('C', 'D')])
        assert sorted((v for (k, v) in G.degree())) == [0, 0, 0, 2, 2, 3, 3]
        assert sorted(G.degree(), key=str) == [('A', 2), ('B', 3), ('C', 3), ('D', 2), ('G', 0), ('J', 0), ('K', 0)]
        assert sorted(G.neighbors('A')) == ['B', 'C']
        pytest.raises(nx.NetworkXError, G.neighbors, 'X')
        G.clear()
        assert nx.number_of_nodes(G) == 0
        assert nx.number_of_edges(G) == 0

    def test_null_subgraph(self):
        if False:
            return 10
        nullgraph = nx.null_graph()
        G = nx.null_graph()
        H = G.subgraph([])
        assert nx.is_isomorphic(H, nullgraph)

    def test_empty_subgraph(self):
        if False:
            i = 10
            return i + 15
        nullgraph = nx.null_graph()
        E5 = nx.empty_graph(5)
        E10 = nx.empty_graph(10)
        H = E10.subgraph([])
        assert nx.is_isomorphic(H, nullgraph)
        H = E10.subgraph([1, 2, 3, 4, 5])
        assert nx.is_isomorphic(H, E5)

    def test_complete_subgraph(self):
        if False:
            print('Hello World!')
        K1 = nx.complete_graph(1)
        K3 = nx.complete_graph(3)
        K5 = nx.complete_graph(5)
        H = K5.subgraph([1, 2, 3])
        assert nx.is_isomorphic(H, K3)

    def test_subgraph_nbunch(self):
        if False:
            print('Hello World!')
        nullgraph = nx.null_graph()
        K1 = nx.complete_graph(1)
        K3 = nx.complete_graph(3)
        K5 = nx.complete_graph(5)
        H = K5.subgraph(1)
        assert nx.is_isomorphic(H, K1)
        H = K5.subgraph({1})
        assert nx.is_isomorphic(H, K1)
        H = K5.subgraph(iter(K3))
        assert nx.is_isomorphic(H, K3)
        H = K5.subgraph(K3)
        assert nx.is_isomorphic(H, K3)
        H = K5.subgraph([9])
        assert nx.is_isomorphic(H, nullgraph)

    def test_node_tuple_issue(self):
        if False:
            for i in range(10):
                print('nop')
        H = self.G()
        pytest.raises(nx.NetworkXError, H.remove_node, (1, 2))
        H.remove_nodes_from([(1, 2)])
        pytest.raises(nx.NetworkXError, H.neighbors, (1, 2))