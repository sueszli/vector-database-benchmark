import pytest
import networkx as nx

class TestLoadCentrality:

    @classmethod
    def setup_class(cls):
        if False:
            return 10
        G = nx.Graph()
        G.add_edge(0, 1, weight=3)
        G.add_edge(0, 2, weight=2)
        G.add_edge(0, 3, weight=6)
        G.add_edge(0, 4, weight=4)
        G.add_edge(1, 3, weight=5)
        G.add_edge(1, 5, weight=5)
        G.add_edge(2, 4, weight=1)
        G.add_edge(3, 4, weight=2)
        G.add_edge(3, 5, weight=1)
        G.add_edge(4, 5, weight=4)
        cls.G = G
        cls.exact_weighted = {0: 4.0, 1: 0.0, 2: 8.0, 3: 6.0, 4: 8.0, 5: 0.0}
        cls.K = nx.krackhardt_kite_graph()
        cls.P3 = nx.path_graph(3)
        cls.P4 = nx.path_graph(4)
        cls.K5 = nx.complete_graph(5)
        cls.P2 = nx.path_graph(2)
        cls.C4 = nx.cycle_graph(4)
        cls.T = nx.balanced_tree(r=2, h=2)
        cls.Gb = nx.Graph()
        cls.Gb.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3), (2, 4), (4, 5), (3, 5)])
        cls.F = nx.florentine_families_graph()
        cls.LM = nx.les_miserables_graph()
        cls.D = nx.cycle_graph(3, create_using=nx.DiGraph())
        cls.D.add_edges_from([(3, 0), (4, 3)])

    def test_not_strongly_connected(self):
        if False:
            return 10
        b = nx.load_centrality(self.D)
        result = {0: 5.0 / 12, 1: 1.0 / 4, 2: 1.0 / 12, 3: 1.0 / 4, 4: 0.0}
        for n in sorted(self.D):
            assert result[n] == pytest.approx(b[n], abs=0.001)
            assert result[n] == pytest.approx(nx.load_centrality(self.D, n), abs=0.001)

    def test_P2_normalized_load(self):
        if False:
            return 10
        G = self.P2
        c = nx.load_centrality(G, normalized=True)
        d = {0: 0.0, 1: 0.0}
        for n in sorted(G):
            assert c[n] == pytest.approx(d[n], abs=0.001)

    def test_weighted_load(self):
        if False:
            for i in range(10):
                print('nop')
        b = nx.load_centrality(self.G, weight='weight', normalized=False)
        for n in sorted(self.G):
            assert b[n] == self.exact_weighted[n]

    def test_k5_load(self):
        if False:
            print('Hello World!')
        G = self.K5
        c = nx.load_centrality(G)
        d = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
        for n in sorted(G):
            assert c[n] == pytest.approx(d[n], abs=0.001)

    def test_p3_load(self):
        if False:
            return 10
        G = self.P3
        c = nx.load_centrality(G)
        d = {0: 0.0, 1: 1.0, 2: 0.0}
        for n in sorted(G):
            assert c[n] == pytest.approx(d[n], abs=0.001)
        c = nx.load_centrality(G, v=1)
        assert c == pytest.approx(1.0, abs=1e-07)
        c = nx.load_centrality(G, v=1, normalized=True)
        assert c == pytest.approx(1.0, abs=1e-07)

    def test_p2_load(self):
        if False:
            i = 10
            return i + 15
        G = nx.path_graph(2)
        c = nx.load_centrality(G)
        d = {0: 0.0, 1: 0.0}
        for n in sorted(G):
            assert c[n] == pytest.approx(d[n], abs=0.001)

    def test_krackhardt_load(self):
        if False:
            while True:
                i = 10
        G = self.K
        c = nx.load_centrality(G)
        d = {0: 0.023, 1: 0.023, 2: 0.0, 3: 0.102, 4: 0.0, 5: 0.231, 6: 0.231, 7: 0.389, 8: 0.222, 9: 0.0}
        for n in sorted(G):
            assert c[n] == pytest.approx(d[n], abs=0.001)

    def test_florentine_families_load(self):
        if False:
            while True:
                i = 10
        G = self.F
        c = nx.load_centrality(G)
        d = {'Acciaiuoli': 0.0, 'Albizzi': 0.211, 'Barbadori': 0.093, 'Bischeri': 0.104, 'Castellani': 0.055, 'Ginori': 0.0, 'Guadagni': 0.251, 'Lamberteschi': 0.0, 'Medici': 0.522, 'Pazzi': 0.0, 'Peruzzi': 0.022, 'Ridolfi': 0.117, 'Salviati': 0.143, 'Strozzi': 0.106, 'Tornabuoni': 0.09}
        for n in sorted(G):
            assert c[n] == pytest.approx(d[n], abs=0.001)

    def test_les_miserables_load(self):
        if False:
            for i in range(10):
                print('nop')
        G = self.LM
        c = nx.load_centrality(G)
        d = {'Napoleon': 0.0, 'Myriel': 0.177, 'MlleBaptistine': 0.0, 'MmeMagloire': 0.0, 'CountessDeLo': 0.0, 'Geborand': 0.0, 'Champtercier': 0.0, 'Cravatte': 0.0, 'Count': 0.0, 'OldMan': 0.0, 'Valjean': 0.567, 'Labarre': 0.0, 'Marguerite': 0.0, 'MmeDeR': 0.0, 'Isabeau': 0.0, 'Gervais': 0.0, 'Listolier': 0.0, 'Tholomyes': 0.043, 'Fameuil': 0.0, 'Blacheville': 0.0, 'Favourite': 0.0, 'Dahlia': 0.0, 'Zephine': 0.0, 'Fantine': 0.128, 'MmeThenardier': 0.029, 'Thenardier': 0.075, 'Cosette': 0.024, 'Javert': 0.054, 'Fauchelevent': 0.026, 'Bamatabois': 0.008, 'Perpetue': 0.0, 'Simplice': 0.009, 'Scaufflaire': 0.0, 'Woman1': 0.0, 'Judge': 0.0, 'Champmathieu': 0.0, 'Brevet': 0.0, 'Chenildieu': 0.0, 'Cochepaille': 0.0, 'Pontmercy': 0.007, 'Boulatruelle': 0.0, 'Eponine': 0.012, 'Anzelma': 0.0, 'Woman2': 0.0, 'MotherInnocent': 0.0, 'Gribier': 0.0, 'MmeBurgon': 0.026, 'Jondrette': 0.0, 'Gavroche': 0.164, 'Gillenormand': 0.021, 'Magnon': 0.0, 'MlleGillenormand': 0.047, 'MmePontmercy': 0.0, 'MlleVaubois': 0.0, 'LtGillenormand': 0.0, 'Marius': 0.133, 'BaronessT': 0.0, 'Mabeuf': 0.028, 'Enjolras': 0.041, 'Combeferre': 0.001, 'Prouvaire': 0.0, 'Feuilly': 0.001, 'Courfeyrac': 0.006, 'Bahorel': 0.002, 'Bossuet': 0.032, 'Joly': 0.002, 'Grantaire': 0.0, 'MotherPlutarch': 0.0, 'Gueulemer': 0.005, 'Babet': 0.005, 'Claquesous': 0.005, 'Montparnasse': 0.004, 'Toussaint': 0.0, 'Child1': 0.0, 'Child2': 0.0, 'Brujon': 0.0, 'MmeHucheloup': 0.0}
        for n in sorted(G):
            assert c[n] == pytest.approx(d[n], abs=0.001)

    def test_unnormalized_k5_load(self):
        if False:
            while True:
                i = 10
        G = self.K5
        c = nx.load_centrality(G, normalized=False)
        d = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
        for n in sorted(G):
            assert c[n] == pytest.approx(d[n], abs=0.001)

    def test_unnormalized_p3_load(self):
        if False:
            return 10
        G = self.P3
        c = nx.load_centrality(G, normalized=False)
        d = {0: 0.0, 1: 2.0, 2: 0.0}
        for n in sorted(G):
            assert c[n] == pytest.approx(d[n], abs=0.001)

    def test_unnormalized_krackhardt_load(self):
        if False:
            return 10
        G = self.K
        c = nx.load_centrality(G, normalized=False)
        d = {0: 1.667, 1: 1.667, 2: 0.0, 3: 7.333, 4: 0.0, 5: 16.667, 6: 16.667, 7: 28.0, 8: 16.0, 9: 0.0}
        for n in sorted(G):
            assert c[n] == pytest.approx(d[n], abs=0.001)

    def test_unnormalized_florentine_families_load(self):
        if False:
            while True:
                i = 10
        G = self.F
        c = nx.load_centrality(G, normalized=False)
        d = {'Acciaiuoli': 0.0, 'Albizzi': 38.333, 'Barbadori': 17.0, 'Bischeri': 19.0, 'Castellani': 10.0, 'Ginori': 0.0, 'Guadagni': 45.667, 'Lamberteschi': 0.0, 'Medici': 95.0, 'Pazzi': 0.0, 'Peruzzi': 4.0, 'Ridolfi': 21.333, 'Salviati': 26.0, 'Strozzi': 19.333, 'Tornabuoni': 16.333}
        for n in sorted(G):
            assert c[n] == pytest.approx(d[n], abs=0.001)

    def test_load_betweenness_difference(self):
        if False:
            while True:
                i = 10
        B = nx.Graph()
        B.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3), (2, 4), (4, 5), (3, 5)])
        c = nx.load_centrality(B, normalized=False)
        d = {0: 1.75, 1: 1.75, 2: 6.5, 3: 6.5, 4: 1.75, 5: 1.75}
        for n in sorted(B):
            assert c[n] == pytest.approx(d[n], abs=0.001)

    def test_c4_edge_load(self):
        if False:
            i = 10
            return i + 15
        G = self.C4
        c = nx.edge_load_centrality(G)
        d = {(0, 1): 6.0, (0, 3): 6.0, (1, 2): 6.0, (2, 3): 6.0}
        for n in G.edges():
            assert c[n] == pytest.approx(d[n], abs=0.001)

    def test_p4_edge_load(self):
        if False:
            i = 10
            return i + 15
        G = self.P4
        c = nx.edge_load_centrality(G)
        d = {(0, 1): 6.0, (1, 2): 8.0, (2, 3): 6.0}
        for n in G.edges():
            assert c[n] == pytest.approx(d[n], abs=0.001)

    def test_k5_edge_load(self):
        if False:
            print('Hello World!')
        G = self.K5
        c = nx.edge_load_centrality(G)
        d = {(0, 1): 5.0, (0, 2): 5.0, (0, 3): 5.0, (0, 4): 5.0, (1, 2): 5.0, (1, 3): 5.0, (1, 4): 5.0, (2, 3): 5.0, (2, 4): 5.0, (3, 4): 5.0}
        for n in G.edges():
            assert c[n] == pytest.approx(d[n], abs=0.001)

    def test_tree_edge_load(self):
        if False:
            for i in range(10):
                print('nop')
        G = self.T
        c = nx.edge_load_centrality(G)
        d = {(0, 1): 24.0, (0, 2): 24.0, (1, 3): 12.0, (1, 4): 12.0, (2, 5): 12.0, (2, 6): 12.0}
        for n in G.edges():
            assert c[n] == pytest.approx(d[n], abs=0.001)