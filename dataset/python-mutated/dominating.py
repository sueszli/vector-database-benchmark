"""Functions for computing dominating sets in a graph."""
from itertools import chain
import networkx as nx
from networkx.utils import arbitrary_element
__all__ = ['dominating_set', 'is_dominating_set']

@nx._dispatch
def dominating_set(G, start_with=None):
    if False:
        print('Hello World!')
    'Finds a dominating set for the graph G.\n\n    A *dominating set* for a graph with node set *V* is a subset *D* of\n    *V* such that every node not in *D* is adjacent to at least one\n    member of *D* [1]_.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n\n    start_with : node (default=None)\n        Node to use as a starting point for the algorithm.\n\n    Returns\n    -------\n    D : set\n        A dominating set for G.\n\n    Notes\n    -----\n    This function is an implementation of algorithm 7 in [2]_ which\n    finds some dominating set, not necessarily the smallest one.\n\n    See also\n    --------\n    is_dominating_set\n\n    References\n    ----------\n    .. [1] https://en.wikipedia.org/wiki/Dominating_set\n\n    .. [2] Abdol-Hossein Esfahanian. Connectivity Algorithms.\n        http://www.cse.msu.edu/~cse835/Papers/Graph_connectivity_revised.pdf\n\n    '
    all_nodes = set(G)
    if start_with is None:
        start_with = arbitrary_element(all_nodes)
    if start_with not in G:
        raise nx.NetworkXError(f'node {start_with} is not in G')
    dominating_set = {start_with}
    dominated_nodes = set(G[start_with])
    remaining_nodes = all_nodes - dominated_nodes - dominating_set
    while remaining_nodes:
        v = remaining_nodes.pop()
        undominated_neighbors = set(G[v]) - dominating_set
        dominating_set.add(v)
        dominated_nodes |= undominated_neighbors
        remaining_nodes -= undominated_neighbors
    return dominating_set

@nx._dispatch
def is_dominating_set(G, nbunch):
    if False:
        while True:
            i = 10
    'Checks if `nbunch` is a dominating set for `G`.\n\n    A *dominating set* for a graph with node set *V* is a subset *D* of\n    *V* such that every node not in *D* is adjacent to at least one\n    member of *D* [1]_.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n\n    nbunch : iterable\n        An iterable of nodes in the graph `G`.\n\n    See also\n    --------\n    dominating_set\n\n    References\n    ----------\n    .. [1] https://en.wikipedia.org/wiki/Dominating_set\n\n    '
    testset = {n for n in nbunch if n in G}
    nbrs = set(chain.from_iterable((G[n] for n in testset)))
    return len(set(G) - testset - nbrs) == 0