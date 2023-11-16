"""Functions for computing treewidth decomposition.

Treewidth of an undirected graph is a number associated with the graph.
It can be defined as the size of the largest vertex set (bag) in a tree
decomposition of the graph minus one.

`Wikipedia: Treewidth <https://en.wikipedia.org/wiki/Treewidth>`_

The notions of treewidth and tree decomposition have gained their
attractiveness partly because many graph and network problems that are
intractable (e.g., NP-hard) on arbitrary graphs become efficiently
solvable (e.g., with a linear time algorithm) when the treewidth of the
input graphs is bounded by a constant [1]_ [2]_.

There are two different functions for computing a tree decomposition:
:func:`treewidth_min_degree` and :func:`treewidth_min_fill_in`.

.. [1] Hans L. Bodlaender and Arie M. C. A. Koster. 2010. "Treewidth
      computations I.Upper bounds". Inf. Comput. 208, 3 (March 2010),259-275.
      http://dx.doi.org/10.1016/j.ic.2009.03.008

.. [2] Hans L. Bodlaender. "Discovering Treewidth". Institute of Information
      and Computing Sciences, Utrecht University.
      Technical Report UU-CS-2005-018.
      http://www.cs.uu.nl

.. [3] K. Wang, Z. Lu, and J. Hicks *Treewidth*.
      https://web.archive.org/web/20210507025929/http://web.eecs.utk.edu/~cphill25/cs594_spring2015_projects/treewidth.pdf

"""
import itertools
import sys
from heapq import heapify, heappop, heappush
import networkx as nx
from networkx.utils import not_implemented_for
__all__ = ['treewidth_min_degree', 'treewidth_min_fill_in']

@not_implemented_for('directed')
@not_implemented_for('multigraph')
@nx._dispatch
def treewidth_min_degree(G):
    if False:
        return 10
    'Returns a treewidth decomposition using the Minimum Degree heuristic.\n\n    The heuristic chooses the nodes according to their degree, i.e., first\n    the node with the lowest degree is chosen, then the graph is updated\n    and the corresponding node is removed. Next, a new node with the lowest\n    degree is chosen, and so on.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n\n    Returns\n    -------\n    Treewidth decomposition : (int, Graph) tuple\n          2-tuple with treewidth and the corresponding decomposed tree.\n    '
    deg_heuristic = MinDegreeHeuristic(G)
    return treewidth_decomp(G, lambda graph: deg_heuristic.best_node(graph))

@not_implemented_for('directed')
@not_implemented_for('multigraph')
@nx._dispatch
def treewidth_min_fill_in(G):
    if False:
        while True:
            i = 10
    'Returns a treewidth decomposition using the Minimum Fill-in heuristic.\n\n    The heuristic chooses a node from the graph, where the number of edges\n    added turning the neighbourhood of the chosen node into clique is as\n    small as possible.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n\n    Returns\n    -------\n    Treewidth decomposition : (int, Graph) tuple\n        2-tuple with treewidth and the corresponding decomposed tree.\n    '
    return treewidth_decomp(G, min_fill_in_heuristic)

class MinDegreeHeuristic:
    """Implements the Minimum Degree heuristic.

    The heuristic chooses the nodes according to their degree
    (number of neighbours), i.e., first the node with the lowest degree is
    chosen, then the graph is updated and the corresponding node is
    removed. Next, a new node with the lowest degree is chosen, and so on.
    """

    def __init__(self, graph):
        if False:
            return 10
        self._graph = graph
        self._update_nodes = []
        self._degreeq = []
        self.count = itertools.count()
        for n in graph:
            self._degreeq.append((len(graph[n]), next(self.count), n))
        heapify(self._degreeq)

    def best_node(self, graph):
        if False:
            i = 10
            return i + 15
        for n in self._update_nodes:
            heappush(self._degreeq, (len(graph[n]), next(self.count), n))
        while self._degreeq:
            (min_degree, _, elim_node) = heappop(self._degreeq)
            if elim_node not in graph or len(graph[elim_node]) != min_degree:
                continue
            elif min_degree == len(graph) - 1:
                return None
            self._update_nodes = graph[elim_node]
            return elim_node
        return None

def min_fill_in_heuristic(graph):
    if False:
        while True:
            i = 10
    'Implements the Minimum Degree heuristic.\n\n    Returns the node from the graph, where the number of edges added when\n    turning the neighbourhood of the chosen node into clique is as small as\n    possible. This algorithm chooses the nodes using the Minimum Fill-In\n    heuristic. The running time of the algorithm is :math:`O(V^3)` and it uses\n    additional constant memory.'
    if len(graph) == 0:
        return None
    min_fill_in_node = None
    min_fill_in = sys.maxsize
    nodes_by_degree = sorted(graph, key=lambda x: len(graph[x]))
    min_degree = len(graph[nodes_by_degree[0]])
    if min_degree == len(graph) - 1:
        return None
    for node in nodes_by_degree:
        num_fill_in = 0
        nbrs = graph[node]
        for nbr in nbrs:
            num_fill_in += len(nbrs - graph[nbr]) - 1
            if num_fill_in >= 2 * min_fill_in:
                break
        num_fill_in /= 2
        if num_fill_in < min_fill_in:
            if num_fill_in == 0:
                return node
            min_fill_in = num_fill_in
            min_fill_in_node = node
    return min_fill_in_node

@nx._dispatch
def treewidth_decomp(G, heuristic=min_fill_in_heuristic):
    if False:
        return 10
    'Returns a treewidth decomposition using the passed heuristic.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n    heuristic : heuristic function\n\n    Returns\n    -------\n    Treewidth decomposition : (int, Graph) tuple\n        2-tuple with treewidth and the corresponding decomposed tree.\n    '
    graph = {n: set(G[n]) - {n} for n in G}
    node_stack = []
    elim_node = heuristic(graph)
    while elim_node is not None:
        nbrs = graph[elim_node]
        for (u, v) in itertools.permutations(nbrs, 2):
            if v not in graph[u]:
                graph[u].add(v)
        node_stack.append((elim_node, nbrs))
        for u in graph[elim_node]:
            graph[u].remove(elim_node)
        del graph[elim_node]
        elim_node = heuristic(graph)
    decomp = nx.Graph()
    first_bag = frozenset(graph.keys())
    decomp.add_node(first_bag)
    treewidth = len(first_bag) - 1
    while node_stack:
        (curr_node, nbrs) = node_stack.pop()
        old_bag = None
        for bag in decomp.nodes:
            if nbrs <= bag:
                old_bag = bag
                break
        if old_bag is None:
            old_bag = first_bag
        nbrs.add(curr_node)
        new_bag = frozenset(nbrs)
        treewidth = max(treewidth, len(new_bag) - 1)
        decomp.add_edge(old_bag, new_bag)
    return (treewidth, decomp)