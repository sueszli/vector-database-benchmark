import pytest
np = pytest.importorskip('numpy')
pytest.importorskip('scipy')
import networkx as nx
from networkx.exception import NetworkXError
from networkx.generators.degree_seq import havel_hakimi_graph

def test_incidence_matrix_simple():
    if False:
        while True:
            i = 10
    deg = [3, 2, 2, 1, 0]
    G = havel_hakimi_graph(deg)
    deg = [(1, 0), (1, 0), (1, 0), (2, 0), (1, 0), (2, 1), (0, 1), (0, 1)]
    MG = nx.random_clustered_graph(deg, seed=42)
    I = nx.incidence_matrix(G, dtype=int).todense()
    expected = np.array([[1, 1, 1, 0], [0, 1, 0, 1], [1, 0, 0, 1], [0, 0, 1, 0], [0, 0, 0, 0]])
    np.testing.assert_equal(I, expected)
    I = nx.incidence_matrix(MG, dtype=int).todense()
    expected = np.array([[1, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1, 0], [0, 0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 1, 0, 1]])
    np.testing.assert_equal(I, expected)
    with pytest.raises(NetworkXError):
        nx.incidence_matrix(G, nodelist=[0, 1])

class TestGraphMatrix:

    @classmethod
    def setup_class(cls):
        if False:
            while True:
                i = 10
        deg = [3, 2, 2, 1, 0]
        cls.G = havel_hakimi_graph(deg)
        cls.OI = np.array([[-1, -1, -1, 0], [1, 0, 0, -1], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 0]])
        cls.A = np.array([[0, 1, 1, 1, 0], [1, 0, 1, 0, 0], [1, 1, 0, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
        cls.WG = havel_hakimi_graph(deg)
        cls.WG.add_edges_from(((u, v, {'weight': 0.5, 'other': 0.3}) for (u, v) in cls.G.edges()))
        cls.WA = np.array([[0, 0.5, 0.5, 0.5, 0], [0.5, 0, 0.5, 0, 0], [0.5, 0.5, 0, 0, 0], [0.5, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
        cls.MG = nx.MultiGraph(cls.G)
        cls.MG2 = cls.MG.copy()
        cls.MG2.add_edge(0, 1)
        cls.MG2A = np.array([[0, 2, 1, 1, 0], [2, 0, 1, 0, 0], [1, 1, 0, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
        cls.MGOI = np.array([[-1, -1, -1, -1, 0], [1, 1, 0, 0, -1], [0, 0, 1, 0, 1], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0]])
        cls.no_edges_G = nx.Graph([(1, 2), (3, 2, {'weight': 8})])
        cls.no_edges_A = np.array([[0, 0], [0, 0]])

    def test_incidence_matrix(self):
        if False:
            return 10
        'Conversion to incidence matrix'
        I = nx.incidence_matrix(self.G, nodelist=sorted(self.G), edgelist=sorted(self.G.edges()), oriented=True, dtype=int).todense()
        np.testing.assert_equal(I, self.OI)
        I = nx.incidence_matrix(self.G, nodelist=sorted(self.G), edgelist=sorted(self.G.edges()), oriented=False, dtype=int).todense()
        np.testing.assert_equal(I, np.abs(self.OI))
        I = nx.incidence_matrix(self.MG, nodelist=sorted(self.MG), edgelist=sorted(self.MG.edges()), oriented=True, dtype=int).todense()
        np.testing.assert_equal(I, self.OI)
        I = nx.incidence_matrix(self.MG, nodelist=sorted(self.MG), edgelist=sorted(self.MG.edges()), oriented=False, dtype=int).todense()
        np.testing.assert_equal(I, np.abs(self.OI))
        I = nx.incidence_matrix(self.MG2, nodelist=sorted(self.MG2), edgelist=sorted(self.MG2.edges()), oriented=True, dtype=int).todense()
        np.testing.assert_equal(I, self.MGOI)
        I = nx.incidence_matrix(self.MG2, nodelist=sorted(self.MG), edgelist=sorted(self.MG2.edges()), oriented=False, dtype=int).todense()
        np.testing.assert_equal(I, np.abs(self.MGOI))
        I = nx.incidence_matrix(self.G, dtype=np.uint8)
        assert I.dtype == np.uint8

    def test_weighted_incidence_matrix(self):
        if False:
            while True:
                i = 10
        I = nx.incidence_matrix(self.WG, nodelist=sorted(self.WG), edgelist=sorted(self.WG.edges()), oriented=True, dtype=int).todense()
        np.testing.assert_equal(I, self.OI)
        I = nx.incidence_matrix(self.WG, nodelist=sorted(self.WG), edgelist=sorted(self.WG.edges()), oriented=False, dtype=int).todense()
        np.testing.assert_equal(I, np.abs(self.OI))
        I = nx.incidence_matrix(self.WG, nodelist=sorted(self.WG), edgelist=sorted(self.WG.edges()), oriented=True, weight='weight').todense()
        np.testing.assert_equal(I, 0.5 * self.OI)
        I = nx.incidence_matrix(self.WG, nodelist=sorted(self.WG), edgelist=sorted(self.WG.edges()), oriented=False, weight='weight').todense()
        np.testing.assert_equal(I, np.abs(0.5 * self.OI))
        I = nx.incidence_matrix(self.WG, nodelist=sorted(self.WG), edgelist=sorted(self.WG.edges()), oriented=True, weight='other').todense()
        np.testing.assert_equal(I, 0.3 * self.OI)
        WMG = nx.MultiGraph(self.WG)
        WMG.add_edge(0, 1, weight=0.5, other=0.3)
        I = nx.incidence_matrix(WMG, nodelist=sorted(WMG), edgelist=sorted(WMG.edges(keys=True)), oriented=True, weight='weight').todense()
        np.testing.assert_equal(I, 0.5 * self.MGOI)
        I = nx.incidence_matrix(WMG, nodelist=sorted(WMG), edgelist=sorted(WMG.edges(keys=True)), oriented=False, weight='weight').todense()
        np.testing.assert_equal(I, np.abs(0.5 * self.MGOI))
        I = nx.incidence_matrix(WMG, nodelist=sorted(WMG), edgelist=sorted(WMG.edges(keys=True)), oriented=True, weight='other').todense()
        np.testing.assert_equal(I, 0.3 * self.MGOI)

    def test_adjacency_matrix(self):
        if False:
            i = 10
            return i + 15
        'Conversion to adjacency matrix'
        np.testing.assert_equal(nx.adjacency_matrix(self.G).todense(), self.A)
        np.testing.assert_equal(nx.adjacency_matrix(self.MG).todense(), self.A)
        np.testing.assert_equal(nx.adjacency_matrix(self.MG2).todense(), self.MG2A)
        np.testing.assert_equal(nx.adjacency_matrix(self.G, nodelist=[0, 1]).todense(), self.A[:2, :2])
        np.testing.assert_equal(nx.adjacency_matrix(self.WG).todense(), self.WA)
        np.testing.assert_equal(nx.adjacency_matrix(self.WG, weight=None).todense(), self.A)
        np.testing.assert_equal(nx.adjacency_matrix(self.MG2, weight=None).todense(), self.MG2A)
        np.testing.assert_equal(nx.adjacency_matrix(self.WG, weight='other').todense(), 0.6 * self.WA)
        np.testing.assert_equal(nx.adjacency_matrix(self.no_edges_G, nodelist=[1, 3]).todense(), self.no_edges_A)