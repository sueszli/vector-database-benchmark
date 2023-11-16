""" Functions related to graph covers."""
from functools import partial
from itertools import chain
import networkx as nx
from networkx.utils import arbitrary_element, not_implemented_for
__all__ = ['min_edge_cover', 'is_edge_cover']

@not_implemented_for('directed')
@not_implemented_for('multigraph')
@nx._dispatch
def min_edge_cover(G, matching_algorithm=None):
    if False:
        print('Hello World!')
    'Returns the min cardinality edge cover of the graph as a set of edges.\n\n    A smallest edge cover can be found in polynomial time by finding\n    a maximum matching and extending it greedily so that all nodes\n    are covered. This function follows that process. A maximum matching\n    algorithm can be specified for the first step of the algorithm.\n    The resulting set may return a set with one 2-tuple for each edge,\n    (the usual case) or with both 2-tuples `(u, v)` and `(v, u)` for\n    each edge. The latter is only done when a bipartite matching algorithm\n    is specified as `matching_algorithm`.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n        An undirected graph.\n\n    matching_algorithm : function\n        A function that returns a maximum cardinality matching for `G`.\n        The function must take one input, the graph `G`, and return\n        either a set of edges (with only one direction for the pair of nodes)\n        or a dictionary mapping each node to its mate. If not specified,\n        :func:`~networkx.algorithms.matching.max_weight_matching` is used.\n        Common bipartite matching functions include\n        :func:`~networkx.algorithms.bipartite.matching.hopcroft_karp_matching`\n        or\n        :func:`~networkx.algorithms.bipartite.matching.eppstein_matching`.\n\n    Returns\n    -------\n    min_cover : set\n\n        A set of the edges in a minimum edge cover in the form of tuples.\n        It contains only one of the equivalent 2-tuples `(u, v)` and `(v, u)`\n        for each edge. If a bipartite method is used to compute the matching,\n        the returned set contains both the 2-tuples `(u, v)` and `(v, u)`\n        for each edge of a minimum edge cover.\n\n    Examples\n    --------\n    >>> G = nx.Graph([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3)])\n    >>> sorted(nx.min_edge_cover(G))\n    [(2, 1), (3, 0)]\n\n    Notes\n    -----\n    An edge cover of a graph is a set of edges such that every node of\n    the graph is incident to at least one edge of the set.\n    The minimum edge cover is an edge covering of smallest cardinality.\n\n    Due to its implementation, the worst-case running time of this algorithm\n    is bounded by the worst-case running time of the function\n    ``matching_algorithm``.\n\n    Minimum edge cover for `G` can also be found using the `min_edge_covering`\n    function in :mod:`networkx.algorithms.bipartite.covering` which is\n    simply this function with a default matching algorithm of\n    :func:`~networkx.algorithms.bipartite.matching.hopcraft_karp_matching`\n    '
    if len(G) == 0:
        return set()
    if nx.number_of_isolates(G) > 0:
        raise nx.NetworkXException('Graph has a node with no edge incident on it, so no edge cover exists.')
    if matching_algorithm is None:
        matching_algorithm = partial(nx.max_weight_matching, maxcardinality=True)
    maximum_matching = matching_algorithm(G)
    try:
        min_cover = set(maximum_matching.items())
        bipartite_cover = True
    except AttributeError:
        min_cover = maximum_matching
        bipartite_cover = False
    uncovered_nodes = set(G) - {v for (u, v) in min_cover} - {u for (u, v) in min_cover}
    for v in uncovered_nodes:
        u = arbitrary_element(G[v])
        min_cover.add((u, v))
        if bipartite_cover:
            min_cover.add((v, u))
    return min_cover

@not_implemented_for('directed')
@nx._dispatch
def is_edge_cover(G, cover):
    if False:
        i = 10
        return i + 15
    'Decides whether a set of edges is a valid edge cover of the graph.\n\n    Given a set of edges, whether it is an edge covering can\n    be decided if we just check whether all nodes of the graph\n    has an edge from the set, incident on it.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n        An undirected bipartite graph.\n\n    cover : set\n        Set of edges to be checked.\n\n    Returns\n    -------\n    bool\n        Whether the set of edges is a valid edge cover of the graph.\n\n    Examples\n    --------\n    >>> G = nx.Graph([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3)])\n    >>> cover = {(2, 1), (3, 0)}\n    >>> nx.is_edge_cover(G, cover)\n    True\n\n    Notes\n    -----\n    An edge cover of a graph is a set of edges such that every node of\n    the graph is incident to at least one edge of the set.\n    '
    return set(G) <= set(chain.from_iterable(cover))