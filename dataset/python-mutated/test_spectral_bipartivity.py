import pytest
pytest.importorskip('scipy')
import networkx as nx
from networkx.algorithms.bipartite import spectral_bipartivity as sb

class TestSpectralBipartivity:

    def test_star_like(self):
        if False:
            i = 10
            return i + 15
        G = nx.star_graph(2)
        G.add_edge(1, 2)
        assert sb(G) == pytest.approx(0.843, abs=0.001)
        G = nx.star_graph(3)
        G.add_edge(1, 2)
        assert sb(G) == pytest.approx(0.871, abs=0.001)
        G = nx.star_graph(4)
        G.add_edge(1, 2)
        assert sb(G) == pytest.approx(0.89, abs=0.001)

    def test_k23_like(self):
        if False:
            while True:
                i = 10
        G = nx.complete_bipartite_graph(2, 3)
        G.add_edge(0, 1)
        assert sb(G) == pytest.approx(0.769, abs=0.001)
        G = nx.complete_bipartite_graph(2, 3)
        G.add_edge(2, 4)
        assert sb(G) == pytest.approx(0.829, abs=0.001)
        G = nx.complete_bipartite_graph(2, 3)
        G.add_edge(2, 4)
        G.add_edge(3, 4)
        assert sb(G) == pytest.approx(0.731, abs=0.001)
        G = nx.complete_bipartite_graph(2, 3)
        G.add_edge(0, 1)
        G.add_edge(2, 4)
        assert sb(G) == pytest.approx(0.692, abs=0.001)
        G = nx.complete_bipartite_graph(2, 3)
        G.add_edge(2, 4)
        G.add_edge(3, 4)
        G.add_edge(0, 1)
        assert sb(G) == pytest.approx(0.645, abs=0.001)
        G = nx.complete_bipartite_graph(2, 3)
        G.add_edge(2, 4)
        G.add_edge(3, 4)
        G.add_edge(2, 3)
        assert sb(G) == pytest.approx(0.645, abs=0.001)
        G = nx.complete_bipartite_graph(2, 3)
        G.add_edge(2, 4)
        G.add_edge(3, 4)
        G.add_edge(2, 3)
        G.add_edge(0, 1)
        assert sb(G) == pytest.approx(0.597, abs=0.001)

    def test_single_nodes(self):
        if False:
            return 10
        G = nx.complete_bipartite_graph(2, 3)
        G.add_edge(2, 4)
        sbn = sb(G, nodes=[1, 2])
        assert sbn[1] == pytest.approx(0.85, abs=0.01)
        assert sbn[2] == pytest.approx(0.77, abs=0.01)
        G = nx.complete_bipartite_graph(2, 3)
        G.add_edge(0, 1)
        sbn = sb(G, nodes=[1, 2])
        assert sbn[1] == pytest.approx(0.73, abs=0.01)
        assert sbn[2] == pytest.approx(0.82, abs=0.01)