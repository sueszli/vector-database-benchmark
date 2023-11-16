"""Bridge-finding algorithms."""
from itertools import chain
import networkx as nx
from networkx.utils import not_implemented_for
__all__ = ['bridges', 'has_bridges', 'local_bridges']

@not_implemented_for('directed')
@nx._dispatch
def bridges(G, root=None):
    if False:
        for i in range(10):
            print('nop')
    'Generate all bridges in a graph.\n\n    A *bridge* in a graph is an edge whose removal causes the number of\n    connected components of the graph to increase.  Equivalently, a bridge is an\n    edge that does not belong to any cycle. Bridges are also known as cut-edges,\n    isthmuses, or cut arcs.\n\n    Parameters\n    ----------\n    G : undirected graph\n\n    root : node (optional)\n       A node in the graph `G`. If specified, only the bridges in the\n       connected component containing this node will be returned.\n\n    Yields\n    ------\n    e : edge\n       An edge in the graph whose removal disconnects the graph (or\n       causes the number of connected components to increase).\n\n    Raises\n    ------\n    NodeNotFound\n       If `root` is not in the graph `G`.\n\n    NetworkXNotImplemented\n        If `G` is a directed graph.\n\n    Examples\n    --------\n    The barbell graph with parameter zero has a single bridge:\n\n    >>> G = nx.barbell_graph(10, 0)\n    >>> list(nx.bridges(G))\n    [(9, 10)]\n\n    Notes\n    -----\n    This is an implementation of the algorithm described in [1]_.  An edge is a\n    bridge if and only if it is not contained in any chain. Chains are found\n    using the :func:`networkx.chain_decomposition` function.\n\n    The algorithm described in [1]_ requires a simple graph. If the provided\n    graph is a multigraph, we convert it to a simple graph and verify that any\n    bridges discovered by the chain decomposition algorithm are not multi-edges.\n\n    Ignoring polylogarithmic factors, the worst-case time complexity is the\n    same as the :func:`networkx.chain_decomposition` function,\n    $O(m + n)$, where $n$ is the number of nodes in the graph and $m$ is\n    the number of edges.\n\n    References\n    ----------\n    .. [1] https://en.wikipedia.org/wiki/Bridge_%28graph_theory%29#Bridge-Finding_with_Chain_Decompositions\n    '
    multigraph = G.is_multigraph()
    H = nx.Graph(G) if multigraph else G
    chains = nx.chain_decomposition(H, root=root)
    chain_edges = set(chain.from_iterable(chains))
    H_copy = H.copy()
    if root is not None:
        H = H.subgraph(nx.node_connected_component(H, root)).copy()
    for (u, v) in H.edges():
        if (u, v) not in chain_edges and (v, u) not in chain_edges:
            if multigraph and len(G[u][v]) > 1:
                continue
            yield (u, v)

@not_implemented_for('directed')
@nx._dispatch
def has_bridges(G, root=None):
    if False:
        return 10
    'Decide whether a graph has any bridges.\n\n    A *bridge* in a graph is an edge whose removal causes the number of\n    connected components of the graph to increase.\n\n    Parameters\n    ----------\n    G : undirected graph\n\n    root : node (optional)\n       A node in the graph `G`. If specified, only the bridges in the\n       connected component containing this node will be considered.\n\n    Returns\n    -------\n    bool\n       Whether the graph (or the connected component containing `root`)\n       has any bridges.\n\n    Raises\n    ------\n    NodeNotFound\n       If `root` is not in the graph `G`.\n\n    NetworkXNotImplemented\n        If `G` is a directed graph.\n\n    Examples\n    --------\n    The barbell graph with parameter zero has a single bridge::\n\n        >>> G = nx.barbell_graph(10, 0)\n        >>> nx.has_bridges(G)\n        True\n\n    On the other hand, the cycle graph has no bridges::\n\n        >>> G = nx.cycle_graph(5)\n        >>> nx.has_bridges(G)\n        False\n\n    Notes\n    -----\n    This implementation uses the :func:`networkx.bridges` function, so\n    it shares its worst-case time complexity, $O(m + n)$, ignoring\n    polylogarithmic factors, where $n$ is the number of nodes in the\n    graph and $m$ is the number of edges.\n\n    '
    try:
        next(bridges(G, root=root))
    except StopIteration:
        return False
    else:
        return True

@not_implemented_for('multigraph')
@not_implemented_for('directed')
@nx._dispatch(edge_attrs='weight')
def local_bridges(G, with_span=True, weight=None):
    if False:
        while True:
            i = 10
    'Iterate over local bridges of `G` optionally computing the span\n\n    A *local bridge* is an edge whose endpoints have no common neighbors.\n    That is, the edge is not part of a triangle in the graph.\n\n    The *span* of a *local bridge* is the shortest path length between\n    the endpoints if the local bridge is removed.\n\n    Parameters\n    ----------\n    G : undirected graph\n\n    with_span : bool\n        If True, yield a 3-tuple `(u, v, span)`\n\n    weight : function, string or None (default: None)\n        If function, used to compute edge weights for the span.\n        If string, the edge data attribute used in calculating span.\n        If None, all edges have weight 1.\n\n    Yields\n    ------\n    e : edge\n        The local bridges as an edge 2-tuple of nodes `(u, v)` or\n        as a 3-tuple `(u, v, span)` when `with_span is True`.\n\n    Raises\n    ------\n    NetworkXNotImplemented\n        If `G` is a directed graph or multigraph.\n\n    Examples\n    --------\n    A cycle graph has every edge a local bridge with span N-1.\n\n       >>> G = nx.cycle_graph(9)\n       >>> (0, 8, 8) in set(nx.local_bridges(G))\n       True\n    '
    if with_span is not True:
        for (u, v) in G.edges:
            if not set(G[u]) & set(G[v]):
                yield (u, v)
    else:
        wt = nx.weighted._weight_function(G, weight)
        for (u, v) in G.edges:
            if not set(G[u]) & set(G[v]):
                enodes = {u, v}

                def hide_edge(n, nbr, d):
                    if False:
                        i = 10
                        return i + 15
                    if n not in enodes or nbr not in enodes:
                        return wt(n, nbr, d)
                    return None
                try:
                    span = nx.shortest_path_length(G, u, v, weight=hide_edge)
                    yield (u, v, span)
                except nx.NetworkXNoPath:
                    yield (u, v, float('inf'))