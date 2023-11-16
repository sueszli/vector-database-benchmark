"""Functions for computing and verifying regular graphs."""
import networkx as nx
from networkx.utils import not_implemented_for
__all__ = ['is_regular', 'is_k_regular', 'k_factor']

@nx._dispatch
def is_regular(G):
    if False:
        while True:
            i = 10
    'Determines whether the graph ``G`` is a regular graph.\n\n    A regular graph is a graph where each vertex has the same degree. A\n    regular digraph is a graph where the indegree and outdegree of each\n    vertex are equal.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n\n    Returns\n    -------\n    bool\n        Whether the given graph or digraph is regular.\n\n    Examples\n    --------\n    >>> G = nx.DiGraph([(1, 2), (2, 3), (3, 4), (4, 1)])\n    >>> nx.is_regular(G)\n    True\n\n    '
    n1 = nx.utils.arbitrary_element(G)
    if not G.is_directed():
        d1 = G.degree(n1)
        return all((d1 == d for (_, d) in G.degree))
    else:
        d_in = G.in_degree(n1)
        in_regular = all((d_in == d for (_, d) in G.in_degree))
        d_out = G.out_degree(n1)
        out_regular = all((d_out == d for (_, d) in G.out_degree))
        return in_regular and out_regular

@not_implemented_for('directed')
@nx._dispatch
def is_k_regular(G, k):
    if False:
        i = 10
        return i + 15
    'Determines whether the graph ``G`` is a k-regular graph.\n\n    A k-regular graph is a graph where each vertex has degree k.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n\n    Returns\n    -------\n    bool\n        Whether the given graph is k-regular.\n\n    Examples\n    --------\n    >>> G = nx.Graph([(1, 2), (2, 3), (3, 4), (4, 1)])\n    >>> nx.is_k_regular(G, k=3)\n    False\n\n    '
    return all((d == k for (n, d) in G.degree))

@not_implemented_for('directed')
@not_implemented_for('multigraph')
@nx._dispatch(edge_attrs='matching_weight')
def k_factor(G, k, matching_weight='weight'):
    if False:
        print('Hello World!')
    'Compute a k-factor of G\n\n    A k-factor of a graph is a spanning k-regular subgraph.\n    A spanning k-regular subgraph of G is a subgraph that contains\n    each vertex of G and a subset of the edges of G such that each\n    vertex has degree k.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n      Undirected graph\n\n    matching_weight: string, optional (default=\'weight\')\n       Edge data key corresponding to the edge weight.\n       Used for finding the max-weighted perfect matching.\n       If key not found, uses 1 as weight.\n\n    Returns\n    -------\n    G2 : NetworkX graph\n        A k-factor of G\n\n    Examples\n    --------\n    >>> G = nx.Graph([(1, 2), (2, 3), (3, 4), (4, 1)])\n    >>> G2 = nx.k_factor(G, k=1)\n    >>> G2.edges()\n    EdgeView([(1, 2), (3, 4)])\n\n    References\n    ----------\n    .. [1] "An algorithm for computing simple k-factors.",\n       Meijer, Henk, Yurai Núñez-Rodríguez, and David Rappaport,\n       Information processing letters, 2009.\n    '
    from networkx.algorithms.matching import is_perfect_matching, max_weight_matching

    class LargeKGadget:

        def __init__(self, k, degree, node, g):
            if False:
                print('Hello World!')
            self.original = node
            self.g = g
            self.k = k
            self.degree = degree
            self.outer_vertices = [(node, x) for x in range(degree)]
            self.core_vertices = [(node, x + degree) for x in range(degree - k)]

        def replace_node(self):
            if False:
                print('Hello World!')
            adj_view = self.g[self.original]
            neighbors = list(adj_view.keys())
            edge_attrs = list(adj_view.values())
            for (outer, neighbor, edge_attrs) in zip(self.outer_vertices, neighbors, edge_attrs):
                self.g.add_edge(outer, neighbor, **edge_attrs)
            for core in self.core_vertices:
                for outer in self.outer_vertices:
                    self.g.add_edge(core, outer)
            self.g.remove_node(self.original)

        def restore_node(self):
            if False:
                return 10
            self.g.add_node(self.original)
            for outer in self.outer_vertices:
                adj_view = self.g[outer]
                for (neighbor, edge_attrs) in list(adj_view.items()):
                    if neighbor not in self.core_vertices:
                        self.g.add_edge(self.original, neighbor, **edge_attrs)
                        break
            g.remove_nodes_from(self.outer_vertices)
            g.remove_nodes_from(self.core_vertices)

    class SmallKGadget:

        def __init__(self, k, degree, node, g):
            if False:
                while True:
                    i = 10
            self.original = node
            self.k = k
            self.degree = degree
            self.g = g
            self.outer_vertices = [(node, x) for x in range(degree)]
            self.inner_vertices = [(node, x + degree) for x in range(degree)]
            self.core_vertices = [(node, x + 2 * degree) for x in range(k)]

        def replace_node(self):
            if False:
                for i in range(10):
                    print('nop')
            adj_view = self.g[self.original]
            for (outer, inner, (neighbor, edge_attrs)) in zip(self.outer_vertices, self.inner_vertices, list(adj_view.items())):
                self.g.add_edge(outer, inner)
                self.g.add_edge(outer, neighbor, **edge_attrs)
            for core in self.core_vertices:
                for inner in self.inner_vertices:
                    self.g.add_edge(core, inner)
            self.g.remove_node(self.original)

        def restore_node(self):
            if False:
                return 10
            self.g.add_node(self.original)
            for outer in self.outer_vertices:
                adj_view = self.g[outer]
                for (neighbor, edge_attrs) in adj_view.items():
                    if neighbor not in self.core_vertices:
                        self.g.add_edge(self.original, neighbor, **edge_attrs)
                        break
            self.g.remove_nodes_from(self.outer_vertices)
            self.g.remove_nodes_from(self.inner_vertices)
            self.g.remove_nodes_from(self.core_vertices)
    if any((d < k for (_, d) in G.degree)):
        raise nx.NetworkXUnfeasible('Graph contains a vertex with degree less than k')
    g = G.copy()
    gadgets = []
    for (node, degree) in list(g.degree):
        if k < degree / 2.0:
            gadget = SmallKGadget(k, degree, node, g)
        else:
            gadget = LargeKGadget(k, degree, node, g)
        gadget.replace_node()
        gadgets.append(gadget)
    matching = max_weight_matching(g, maxcardinality=True, weight=matching_weight)
    if not is_perfect_matching(g, matching):
        raise nx.NetworkXUnfeasible('Cannot find k-factor because no perfect matching exists')
    for edge in g.edges():
        if edge not in matching and (edge[1], edge[0]) not in matching:
            g.remove_edge(edge[0], edge[1])
    for gadget in gadgets:
        gadget.restore_node()
    return g