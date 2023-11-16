import itertools
import networkx as nx
from networkx.algorithms.approximation import treewidth_min_degree, treewidth_min_fill_in
from networkx.algorithms.approximation.treewidth import MinDegreeHeuristic, min_fill_in_heuristic

def is_tree_decomp(graph, decomp):
    if False:
        print('Hello World!')
    'Check if the given tree decomposition is valid.'
    for x in graph.nodes():
        appear_once = False
        for bag in decomp.nodes():
            if x in bag:
                appear_once = True
                break
        assert appear_once
    for (x, y) in graph.edges():
        appear_together = False
        for bag in decomp.nodes():
            if x in bag and y in bag:
                appear_together = True
                break
        assert appear_together
    for v in graph.nodes():
        subset = []
        for bag in decomp.nodes():
            if v in bag:
                subset.append(bag)
        sub_graph = decomp.subgraph(subset)
        assert nx.is_connected(sub_graph)

class TestTreewidthMinDegree:
    """Unit tests for the min_degree function"""

    @classmethod
    def setup_class(cls):
        if False:
            i = 10
            return i + 15
        'Setup for different kinds of trees'
        cls.complete = nx.Graph()
        cls.complete.add_edge(1, 2)
        cls.complete.add_edge(2, 3)
        cls.complete.add_edge(1, 3)
        cls.small_tree = nx.Graph()
        cls.small_tree.add_edge(1, 3)
        cls.small_tree.add_edge(4, 3)
        cls.small_tree.add_edge(2, 3)
        cls.small_tree.add_edge(3, 5)
        cls.small_tree.add_edge(5, 6)
        cls.small_tree.add_edge(5, 7)
        cls.small_tree.add_edge(6, 7)
        cls.deterministic_graph = nx.Graph()
        cls.deterministic_graph.add_edge(0, 1)
        cls.deterministic_graph.add_edge(1, 2)
        cls.deterministic_graph.add_edge(2, 3)
        cls.deterministic_graph.add_edge(2, 4)
        cls.deterministic_graph.add_edge(3, 4)
        cls.deterministic_graph.add_edge(3, 5)
        cls.deterministic_graph.add_edge(3, 6)
        cls.deterministic_graph.add_edge(4, 5)
        cls.deterministic_graph.add_edge(4, 6)
        cls.deterministic_graph.add_edge(4, 7)
        cls.deterministic_graph.add_edge(5, 6)
        cls.deterministic_graph.add_edge(5, 7)
        cls.deterministic_graph.add_edge(5, 8)
        cls.deterministic_graph.add_edge(5, 9)
        cls.deterministic_graph.add_edge(6, 7)
        cls.deterministic_graph.add_edge(6, 8)
        cls.deterministic_graph.add_edge(6, 9)
        cls.deterministic_graph.add_edge(7, 8)
        cls.deterministic_graph.add_edge(7, 9)
        cls.deterministic_graph.add_edge(8, 9)

    def test_petersen_graph(self):
        if False:
            print('Hello World!')
        'Test Petersen graph tree decomposition result'
        G = nx.petersen_graph()
        (_, decomp) = treewidth_min_degree(G)
        is_tree_decomp(G, decomp)

    def test_small_tree_treewidth(self):
        if False:
            i = 10
            return i + 15
        'Test small tree\n\n        Test if the computed treewidth of the known self.small_tree is 2.\n        As we know which value we can expect from our heuristic, values other\n        than two are regressions\n        '
        G = self.small_tree
        (treewidth, _) = treewidth_min_fill_in(G)
        assert treewidth == 2

    def test_heuristic_abort(self):
        if False:
            while True:
                i = 10
        'Test heuristic abort condition for fully connected graph'
        graph = {}
        for u in self.complete:
            graph[u] = set()
            for v in self.complete[u]:
                if u != v:
                    graph[u].add(v)
        deg_heuristic = MinDegreeHeuristic(graph)
        node = deg_heuristic.best_node(graph)
        if node is None:
            pass
        else:
            assert False

    def test_empty_graph(self):
        if False:
            return 10
        'Test empty graph'
        G = nx.Graph()
        (_, _) = treewidth_min_degree(G)

    def test_two_component_graph(self):
        if False:
            print('Hello World!')
        G = nx.Graph()
        G.add_node(1)
        G.add_node(2)
        (treewidth, _) = treewidth_min_degree(G)
        assert treewidth == 0

    def test_not_sortable_nodes(self):
        if False:
            i = 10
            return i + 15
        G = nx.Graph([(0, 'a')])
        treewidth_min_degree(G)

    def test_heuristic_first_steps(self):
        if False:
            i = 10
            return i + 15
        'Test first steps of min_degree heuristic'
        graph = {n: set(self.deterministic_graph[n]) - {n} for n in self.deterministic_graph}
        deg_heuristic = MinDegreeHeuristic(graph)
        elim_node = deg_heuristic.best_node(graph)
        print(f'Graph {graph}:')
        steps = []
        while elim_node is not None:
            print(f'Removing {elim_node}:')
            steps.append(elim_node)
            nbrs = graph[elim_node]
            for (u, v) in itertools.permutations(nbrs, 2):
                if v not in graph[u]:
                    graph[u].add(v)
            for u in graph:
                if elim_node in graph[u]:
                    graph[u].remove(elim_node)
            del graph[elim_node]
            print(f'Graph {graph}:')
            elim_node = deg_heuristic.best_node(graph)
        assert steps[:5] == [0, 1, 2, 3, 4]

class TestTreewidthMinFillIn:
    """Unit tests for the treewidth_min_fill_in function."""

    @classmethod
    def setup_class(cls):
        if False:
            for i in range(10):
                print('nop')
        'Setup for different kinds of trees'
        cls.complete = nx.Graph()
        cls.complete.add_edge(1, 2)
        cls.complete.add_edge(2, 3)
        cls.complete.add_edge(1, 3)
        cls.small_tree = nx.Graph()
        cls.small_tree.add_edge(1, 2)
        cls.small_tree.add_edge(2, 3)
        cls.small_tree.add_edge(3, 4)
        cls.small_tree.add_edge(1, 4)
        cls.small_tree.add_edge(2, 4)
        cls.small_tree.add_edge(4, 5)
        cls.small_tree.add_edge(5, 6)
        cls.small_tree.add_edge(5, 7)
        cls.small_tree.add_edge(6, 7)
        cls.deterministic_graph = nx.Graph()
        cls.deterministic_graph.add_edge(1, 2)
        cls.deterministic_graph.add_edge(1, 3)
        cls.deterministic_graph.add_edge(3, 4)
        cls.deterministic_graph.add_edge(2, 4)
        cls.deterministic_graph.add_edge(3, 5)
        cls.deterministic_graph.add_edge(4, 5)
        cls.deterministic_graph.add_edge(3, 6)
        cls.deterministic_graph.add_edge(5, 6)

    def test_petersen_graph(self):
        if False:
            i = 10
            return i + 15
        'Test Petersen graph tree decomposition result'
        G = nx.petersen_graph()
        (_, decomp) = treewidth_min_fill_in(G)
        is_tree_decomp(G, decomp)

    def test_small_tree_treewidth(self):
        if False:
            while True:
                i = 10
        'Test if the computed treewidth of the known self.small_tree is 2'
        G = self.small_tree
        (treewidth, _) = treewidth_min_fill_in(G)
        assert treewidth == 2

    def test_heuristic_abort(self):
        if False:
            while True:
                i = 10
        'Test if min_fill_in returns None for fully connected graph'
        graph = {}
        for u in self.complete:
            graph[u] = set()
            for v in self.complete[u]:
                if u != v:
                    graph[u].add(v)
        next_node = min_fill_in_heuristic(graph)
        if next_node is None:
            pass
        else:
            assert False

    def test_empty_graph(self):
        if False:
            return 10
        'Test empty graph'
        G = nx.Graph()
        (_, _) = treewidth_min_fill_in(G)

    def test_two_component_graph(self):
        if False:
            i = 10
            return i + 15
        G = nx.Graph()
        G.add_node(1)
        G.add_node(2)
        (treewidth, _) = treewidth_min_fill_in(G)
        assert treewidth == 0

    def test_not_sortable_nodes(self):
        if False:
            while True:
                i = 10
        G = nx.Graph([(0, 'a')])
        treewidth_min_fill_in(G)

    def test_heuristic_first_steps(self):
        if False:
            while True:
                i = 10
        'Test first steps of min_fill_in heuristic'
        graph = {n: set(self.deterministic_graph[n]) - {n} for n in self.deterministic_graph}
        print(f'Graph {graph}:')
        elim_node = min_fill_in_heuristic(graph)
        steps = []
        while elim_node is not None:
            print(f'Removing {elim_node}:')
            steps.append(elim_node)
            nbrs = graph[elim_node]
            for (u, v) in itertools.permutations(nbrs, 2):
                if v not in graph[u]:
                    graph[u].add(v)
            for u in graph:
                if elim_node in graph[u]:
                    graph[u].remove(elim_node)
            del graph[elim_node]
            print(f'Graph {graph}:')
            elim_node = min_fill_in_heuristic(graph)
        assert steps[:2] == [6, 5]