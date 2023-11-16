from itertools import combinations
import pytest
import networkx as nx

def test_overlapping_K5():
    if False:
        i = 10
        return i + 15
    G = nx.Graph()
    G.add_edges_from(combinations(range(5), 2))
    G.add_edges_from(combinations(range(2, 7), 2))
    c = list(nx.community.k_clique_communities(G, 4))
    assert c == [frozenset(range(7))]
    c = set(nx.community.k_clique_communities(G, 5))
    assert c == {frozenset(range(5)), frozenset(range(2, 7))}

def test_isolated_K5():
    if False:
        for i in range(10):
            print('nop')
    G = nx.Graph()
    G.add_edges_from(combinations(range(5), 2))
    G.add_edges_from(combinations(range(5, 10), 2))
    c = set(nx.community.k_clique_communities(G, 5))
    assert c == {frozenset(range(5)), frozenset(range(5, 10))}

class TestZacharyKarateClub:

    def setup_method(self):
        if False:
            print('Hello World!')
        self.G = nx.karate_club_graph()

    def _check_communities(self, k, expected):
        if False:
            print('Hello World!')
        communities = set(nx.community.k_clique_communities(self.G, k))
        assert communities == expected

    def test_k2(self):
        if False:
            return 10
        expected = {frozenset(self.G)}
        self._check_communities(2, expected)

    def test_k3(self):
        if False:
            print('Hello World!')
        comm1 = [0, 1, 2, 3, 7, 8, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 26, 27, 28, 29, 30, 31, 32, 33]
        comm2 = [0, 4, 5, 6, 10, 16]
        comm3 = [24, 25, 31]
        expected = {frozenset(comm1), frozenset(comm2), frozenset(comm3)}
        self._check_communities(3, expected)

    def test_k4(self):
        if False:
            while True:
                i = 10
        expected = {frozenset([0, 1, 2, 3, 7, 13]), frozenset([8, 32, 30, 33]), frozenset([32, 33, 29, 23])}
        self._check_communities(4, expected)

    def test_k5(self):
        if False:
            while True:
                i = 10
        expected = {frozenset([0, 1, 2, 3, 7, 13])}
        self._check_communities(5, expected)

    def test_k6(self):
        if False:
            i = 10
            return i + 15
        expected = set()
        self._check_communities(6, expected)

def test_bad_k():
    if False:
        print('Hello World!')
    with pytest.raises(nx.NetworkXError):
        list(nx.community.k_clique_communities(nx.Graph(), 1))