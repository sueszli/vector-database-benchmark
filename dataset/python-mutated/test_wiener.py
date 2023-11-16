"""Unit tests for the :mod:`networkx.algorithms.wiener` module."""
from networkx import DiGraph, complete_graph, empty_graph, path_graph, wiener_index

class TestWienerIndex:
    """Unit tests for computing the Wiener index of a graph."""

    def test_disconnected_graph(self):
        if False:
            return 10
        'Tests that the Wiener index of a disconnected graph is\n        positive infinity.\n\n        '
        assert wiener_index(empty_graph(2)) == float('inf')

    def test_directed(self):
        if False:
            while True:
                i = 10
        'Tests that each pair of nodes in the directed graph is\n        counted once when computing the Wiener index.\n\n        '
        G = complete_graph(3)
        H = DiGraph(G)
        assert 2 * wiener_index(G) == wiener_index(H)

    def test_complete_graph(self):
        if False:
            return 10
        'Tests that the Wiener index of the complete graph is simply\n        the number of edges.\n\n        '
        n = 10
        G = complete_graph(n)
        assert wiener_index(G) == n * (n - 1) / 2

    def test_path_graph(self):
        if False:
            while True:
                i = 10
        'Tests that the Wiener index of the path graph is correctly\n        computed.\n\n        '
        n = 9
        G = path_graph(n)
        expected = 2 * sum((i * (n - i) for i in range(1, n // 2 + 1)))
        actual = wiener_index(G)
        assert expected == actual