"""Provides functions for computing the efficiency of nodes and graphs."""
import networkx as nx
from networkx.exception import NetworkXNoPath
from ..utils import not_implemented_for
__all__ = ['efficiency', 'local_efficiency', 'global_efficiency']

@not_implemented_for('directed')
@nx._dispatch
def efficiency(G, u, v):
    if False:
        print('Hello World!')
    'Returns the efficiency of a pair of nodes in a graph.\n\n    The *efficiency* of a pair of nodes is the multiplicative inverse of the\n    shortest path distance between the nodes [1]_. Returns 0 if no path\n    between nodes.\n\n    Parameters\n    ----------\n    G : :class:`networkx.Graph`\n        An undirected graph for which to compute the average local efficiency.\n    u, v : node\n        Nodes in the graph ``G``.\n\n    Returns\n    -------\n    float\n        Multiplicative inverse of the shortest path distance between the nodes.\n\n    Examples\n    --------\n    >>> G = nx.Graph([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3)])\n    >>> nx.efficiency(G, 2, 3)  # this gives efficiency for node 2 and 3\n    0.5\n\n    Notes\n    -----\n    Edge weights are ignored when computing the shortest path distances.\n\n    See also\n    --------\n    local_efficiency\n    global_efficiency\n\n    References\n    ----------\n    .. [1] Latora, Vito, and Massimo Marchiori.\n           "Efficient behavior of small-world networks."\n           *Physical Review Letters* 87.19 (2001): 198701.\n           <https://doi.org/10.1103/PhysRevLett.87.198701>\n\n    '
    try:
        eff = 1 / nx.shortest_path_length(G, u, v)
    except NetworkXNoPath:
        eff = 0
    return eff

@not_implemented_for('directed')
@nx._dispatch
def global_efficiency(G):
    if False:
        while True:
            i = 10
    'Returns the average global efficiency of the graph.\n\n    The *efficiency* of a pair of nodes in a graph is the multiplicative\n    inverse of the shortest path distance between the nodes. The *average\n    global efficiency* of a graph is the average efficiency of all pairs of\n    nodes [1]_.\n\n    Parameters\n    ----------\n    G : :class:`networkx.Graph`\n        An undirected graph for which to compute the average global efficiency.\n\n    Returns\n    -------\n    float\n        The average global efficiency of the graph.\n\n    Examples\n    --------\n    >>> G = nx.Graph([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3)])\n    >>> round(nx.global_efficiency(G), 12)\n    0.916666666667\n\n    Notes\n    -----\n    Edge weights are ignored when computing the shortest path distances.\n\n    See also\n    --------\n    local_efficiency\n\n    References\n    ----------\n    .. [1] Latora, Vito, and Massimo Marchiori.\n           "Efficient behavior of small-world networks."\n           *Physical Review Letters* 87.19 (2001): 198701.\n           <https://doi.org/10.1103/PhysRevLett.87.198701>\n\n    '
    n = len(G)
    denom = n * (n - 1)
    if denom != 0:
        lengths = nx.all_pairs_shortest_path_length(G)
        g_eff = 0
        for (source, targets) in lengths:
            for (target, distance) in targets.items():
                if distance > 0:
                    g_eff += 1 / distance
        g_eff /= denom
    else:
        g_eff = 0
    return g_eff

@not_implemented_for('directed')
@nx._dispatch
def local_efficiency(G):
    if False:
        return 10
    'Returns the average local efficiency of the graph.\n\n    The *efficiency* of a pair of nodes in a graph is the multiplicative\n    inverse of the shortest path distance between the nodes. The *local\n    efficiency* of a node in the graph is the average global efficiency of the\n    subgraph induced by the neighbors of the node. The *average local\n    efficiency* is the average of the local efficiencies of each node [1]_.\n\n    Parameters\n    ----------\n    G : :class:`networkx.Graph`\n        An undirected graph for which to compute the average local efficiency.\n\n    Returns\n    -------\n    float\n        The average local efficiency of the graph.\n\n    Examples\n    --------\n    >>> G = nx.Graph([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3)])\n    >>> nx.local_efficiency(G)\n    0.9166666666666667\n\n    Notes\n    -----\n    Edge weights are ignored when computing the shortest path distances.\n\n    See also\n    --------\n    global_efficiency\n\n    References\n    ----------\n    .. [1] Latora, Vito, and Massimo Marchiori.\n           "Efficient behavior of small-world networks."\n           *Physical Review Letters* 87.19 (2001): 198701.\n           <https://doi.org/10.1103/PhysRevLett.87.198701>\n\n    '
    efficiency_list = (global_efficiency(G.subgraph(G[v])) for v in G)
    return sum(efficiency_list) / len(G)