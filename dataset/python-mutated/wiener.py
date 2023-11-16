"""Functions related to the Wiener index of a graph."""
from itertools import chain
import networkx as nx
from .components import is_connected, is_strongly_connected
from .shortest_paths import shortest_path_length as spl
__all__ = ['wiener_index']
chaini = chain.from_iterable

@nx._dispatch(edge_attrs='weight')
def wiener_index(G, weight=None):
    if False:
        i = 10
        return i + 15
    'Returns the Wiener index of the given graph.\n\n    The *Wiener index* of a graph is the sum of the shortest-path\n    distances between each pair of reachable nodes. For pairs of nodes\n    in undirected graphs, only one orientation of the pair is counted.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n\n    weight : object\n        The edge attribute to use as distance when computing\n        shortest-path distances. This is passed directly to the\n        :func:`networkx.shortest_path_length` function.\n\n    Returns\n    -------\n    float\n        The Wiener index of the graph `G`.\n\n    Raises\n    ------\n    NetworkXError\n        If the graph `G` is not connected.\n\n    Notes\n    -----\n    If a pair of nodes is not reachable, the distance is assumed to be\n    infinity. This means that for graphs that are not\n    strongly-connected, this function returns ``inf``.\n\n    The Wiener index is not usually defined for directed graphs, however\n    this function uses the natural generalization of the Wiener index to\n    directed graphs.\n\n    Examples\n    --------\n    The Wiener index of the (unweighted) complete graph on *n* nodes\n    equals the number of pairs of the *n* nodes, since each pair of\n    nodes is at distance one::\n\n        >>> n = 10\n        >>> G = nx.complete_graph(n)\n        >>> nx.wiener_index(G) == n * (n - 1) / 2\n        True\n\n    Graphs that are not strongly-connected have infinite Wiener index::\n\n        >>> G = nx.empty_graph(2)\n        >>> nx.wiener_index(G)\n        inf\n\n    '
    is_directed = G.is_directed()
    if is_directed and (not is_strongly_connected(G)) or (not is_directed and (not is_connected(G))):
        return float('inf')
    total = sum(chaini((p.values() for (v, p) in spl(G, weight=weight))))
    return total if is_directed else total / 2