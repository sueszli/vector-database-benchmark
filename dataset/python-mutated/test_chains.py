"""Unit tests for the chain decomposition functions."""
from itertools import cycle, islice
import pytest
import networkx as nx

def cycles(seq):
    if False:
        while True:
            i = 10
    'Yields cyclic permutations of the given sequence.\n\n    For example::\n\n        >>> list(cycles("abc"))\n        [(\'a\', \'b\', \'c\'), (\'b\', \'c\', \'a\'), (\'c\', \'a\', \'b\')]\n\n    '
    n = len(seq)
    cycled_seq = cycle(seq)
    for x in seq:
        yield tuple(islice(cycled_seq, n))
        next(cycled_seq)

def cyclic_equals(seq1, seq2):
    if False:
        return 10
    'Decide whether two sequences are equal up to cyclic permutations.\n\n    For example::\n\n        >>> cyclic_equals("xyz", "zxy")\n        True\n        >>> cyclic_equals("xyz", "zyx")\n        False\n\n    '
    seq2 = tuple(seq2)
    return any((x == tuple(seq2) for x in cycles(seq1)))

class TestChainDecomposition:
    """Unit tests for the chain decomposition function."""

    def assertContainsChain(self, chain, expected):
        if False:
            while True:
                i = 10
        reversed_chain = list(reversed([tuple(reversed(e)) for e in chain]))
        for candidate in expected:
            if cyclic_equals(chain, candidate):
                break
            if cyclic_equals(reversed_chain, candidate):
                break
        else:
            self.fail('chain not found')

    def test_decomposition(self):
        if False:
            for i in range(10):
                print('nop')
        edges = [(1, 2), (2, 3), (3, 4), (3, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (1, 3), (1, 4), (2, 5), (5, 10), (6, 8)]
        G = nx.Graph(edges)
        expected = [[(1, 3), (3, 2), (2, 1)], [(1, 4), (4, 3)], [(2, 5), (5, 3)], [(5, 10), (10, 9), (9, 5)], [(6, 8), (8, 7), (7, 6)]]
        chains = list(nx.chain_decomposition(G, root=1))
        assert len(chains) == len(expected)

    def test_barbell_graph(self):
        if False:
            while True:
                i = 10
        G = nx.barbell_graph(3, 0)
        chains = list(nx.chain_decomposition(G, root=0))
        expected = [[(0, 1), (1, 2), (2, 0)], [(3, 4), (4, 5), (5, 3)]]
        assert len(chains) == len(expected)
        for chain in chains:
            self.assertContainsChain(chain, expected)

    def test_disconnected_graph(self):
        if False:
            while True:
                i = 10
        'Test for a graph with multiple connected components.'
        G = nx.barbell_graph(3, 0)
        H = nx.barbell_graph(3, 0)
        mapping = dict(zip(range(6), 'abcdef'))
        nx.relabel_nodes(H, mapping, copy=False)
        G = nx.union(G, H)
        chains = list(nx.chain_decomposition(G))
        expected = [[(0, 1), (1, 2), (2, 0)], [(3, 4), (4, 5), (5, 3)], [('a', 'b'), ('b', 'c'), ('c', 'a')], [('d', 'e'), ('e', 'f'), ('f', 'd')]]
        assert len(chains) == len(expected)
        for chain in chains:
            self.assertContainsChain(chain, expected)

    def test_disconnected_graph_root_node(self):
        if False:
            while True:
                i = 10
        'Test for a single component of a disconnected graph.'
        G = nx.barbell_graph(3, 0)
        H = nx.barbell_graph(3, 0)
        mapping = dict(zip(range(6), 'abcdef'))
        nx.relabel_nodes(H, mapping, copy=False)
        G = nx.union(G, H)
        chains = list(nx.chain_decomposition(G, root='a'))
        expected = [[('a', 'b'), ('b', 'c'), ('c', 'a')], [('d', 'e'), ('e', 'f'), ('f', 'd')]]
        assert len(chains) == len(expected)
        for chain in chains:
            self.assertContainsChain(chain, expected)

    def test_chain_decomposition_root_not_in_G(self):
        if False:
            print('Hello World!')
        'Test chain decomposition when root is not in graph'
        G = nx.Graph()
        G.add_nodes_from([1, 2, 3])
        with pytest.raises(nx.NodeNotFound):
            nx.has_bridges(G, root=6)