"""Betweenness centrality measures for subsets of nodes."""
import networkx as nx
from networkx.algorithms.centrality.betweenness import _add_edge_keys
from networkx.algorithms.centrality.betweenness import _single_source_dijkstra_path_basic as dijkstra
from networkx.algorithms.centrality.betweenness import _single_source_shortest_path_basic as shortest_path
__all__ = ['betweenness_centrality_subset', 'edge_betweenness_centrality_subset']

@nx._dispatch(edge_attrs='weight')
def betweenness_centrality_subset(G, sources, targets, normalized=False, weight=None):
    if False:
        print('Hello World!')
    'Compute betweenness centrality for a subset of nodes.\n\n    .. math::\n\n       c_B(v) =\\sum_{s\\in S, t \\in T} \\frac{\\sigma(s, t|v)}{\\sigma(s, t)}\n\n    where $S$ is the set of sources, $T$ is the set of targets,\n    $\\sigma(s, t)$ is the number of shortest $(s, t)$-paths,\n    and $\\sigma(s, t|v)$ is the number of those paths\n    passing through some  node $v$ other than $s, t$.\n    If $s = t$, $\\sigma(s, t) = 1$,\n    and if $v \\in {s, t}$, $\\sigma(s, t|v) = 0$ [2]_.\n\n\n    Parameters\n    ----------\n    G : graph\n      A NetworkX graph.\n\n    sources: list of nodes\n      Nodes to use as sources for shortest paths in betweenness\n\n    targets: list of nodes\n      Nodes to use as targets for shortest paths in betweenness\n\n    normalized : bool, optional\n      If True the betweenness values are normalized by $2/((n-1)(n-2))$\n      for graphs, and $1/((n-1)(n-2))$ for directed graphs where $n$\n      is the number of nodes in G.\n\n    weight : None or string, optional (default=None)\n      If None, all edge weights are considered equal.\n      Otherwise holds the name of the edge attribute used as weight.\n      Weights are used to calculate weighted shortest paths, so they are\n      interpreted as distances.\n\n    Returns\n    -------\n    nodes : dictionary\n       Dictionary of nodes with betweenness centrality as the value.\n\n    See Also\n    --------\n    edge_betweenness_centrality\n    load_centrality\n\n    Notes\n    -----\n    The basic algorithm is from [1]_.\n\n    For weighted graphs the edge weights must be greater than zero.\n    Zero edge weights can produce an infinite number of equal length\n    paths between pairs of nodes.\n\n    The normalization might seem a little strange but it is\n    designed to make betweenness_centrality(G) be the same as\n    betweenness_centrality_subset(G,sources=G.nodes(),targets=G.nodes()).\n\n    The total number of paths between source and target is counted\n    differently for directed and undirected graphs. Directed paths\n    are easy to count. Undirected paths are tricky: should a path\n    from "u" to "v" count as 1 undirected path or as 2 directed paths?\n\n    For betweenness_centrality we report the number of undirected\n    paths when G is undirected.\n\n    For betweenness_centrality_subset the reporting is different.\n    If the source and target subsets are the same, then we want\n    to count undirected paths. But if the source and target subsets\n    differ -- for example, if sources is {0} and targets is {1},\n    then we are only counting the paths in one direction. They are\n    undirected paths but we are counting them in a directed way.\n    To count them as undirected paths, each should count as half a path.\n\n    References\n    ----------\n    .. [1] Ulrik Brandes, A Faster Algorithm for Betweenness Centrality.\n       Journal of Mathematical Sociology 25(2):163-177, 2001.\n       https://doi.org/10.1080/0022250X.2001.9990249\n    .. [2] Ulrik Brandes: On Variants of Shortest-Path Betweenness\n       Centrality and their Generic Computation.\n       Social Networks 30(2):136-145, 2008.\n       https://doi.org/10.1016/j.socnet.2007.11.001\n    '
    b = dict.fromkeys(G, 0.0)
    for s in sources:
        if weight is None:
            (S, P, sigma, _) = shortest_path(G, s)
        else:
            (S, P, sigma, _) = dijkstra(G, s, weight)
        b = _accumulate_subset(b, S, P, sigma, s, targets)
    b = _rescale(b, len(G), normalized=normalized, directed=G.is_directed())
    return b

@nx._dispatch(edge_attrs='weight')
def edge_betweenness_centrality_subset(G, sources, targets, normalized=False, weight=None):
    if False:
        print('Hello World!')
    'Compute betweenness centrality for edges for a subset of nodes.\n\n    .. math::\n\n       c_B(v) =\\sum_{s\\in S,t \\in T} \\frac{\\sigma(s, t|e)}{\\sigma(s, t)}\n\n    where $S$ is the set of sources, $T$ is the set of targets,\n    $\\sigma(s, t)$ is the number of shortest $(s, t)$-paths,\n    and $\\sigma(s, t|e)$ is the number of those paths\n    passing through edge $e$ [2]_.\n\n    Parameters\n    ----------\n    G : graph\n      A networkx graph.\n\n    sources: list of nodes\n      Nodes to use as sources for shortest paths in betweenness\n\n    targets: list of nodes\n      Nodes to use as targets for shortest paths in betweenness\n\n    normalized : bool, optional\n      If True the betweenness values are normalized by `2/(n(n-1))`\n      for graphs, and `1/(n(n-1))` for directed graphs where `n`\n      is the number of nodes in G.\n\n    weight : None or string, optional (default=None)\n      If None, all edge weights are considered equal.\n      Otherwise holds the name of the edge attribute used as weight.\n      Weights are used to calculate weighted shortest paths, so they are\n      interpreted as distances.\n\n    Returns\n    -------\n    edges : dictionary\n       Dictionary of edges with Betweenness centrality as the value.\n\n    See Also\n    --------\n    betweenness_centrality\n    edge_load\n\n    Notes\n    -----\n    The basic algorithm is from [1]_.\n\n    For weighted graphs the edge weights must be greater than zero.\n    Zero edge weights can produce an infinite number of equal length\n    paths between pairs of nodes.\n\n    The normalization might seem a little strange but it is the same\n    as in edge_betweenness_centrality() and is designed to make\n    edge_betweenness_centrality(G) be the same as\n    edge_betweenness_centrality_subset(G,sources=G.nodes(),targets=G.nodes()).\n\n    References\n    ----------\n    .. [1] Ulrik Brandes, A Faster Algorithm for Betweenness Centrality.\n       Journal of Mathematical Sociology 25(2):163-177, 2001.\n       https://doi.org/10.1080/0022250X.2001.9990249\n    .. [2] Ulrik Brandes: On Variants of Shortest-Path Betweenness\n       Centrality and their Generic Computation.\n       Social Networks 30(2):136-145, 2008.\n       https://doi.org/10.1016/j.socnet.2007.11.001\n    '
    b = dict.fromkeys(G, 0.0)
    b.update(dict.fromkeys(G.edges(), 0.0))
    for s in sources:
        if weight is None:
            (S, P, sigma, _) = shortest_path(G, s)
        else:
            (S, P, sigma, _) = dijkstra(G, s, weight)
        b = _accumulate_edges_subset(b, S, P, sigma, s, targets)
    for n in G:
        del b[n]
    b = _rescale_e(b, len(G), normalized=normalized, directed=G.is_directed())
    if G.is_multigraph():
        b = _add_edge_keys(G, b, weight=weight)
    return b

def _accumulate_subset(betweenness, S, P, sigma, s, targets):
    if False:
        for i in range(10):
            print('nop')
    delta = dict.fromkeys(S, 0.0)
    target_set = set(targets) - {s}
    while S:
        w = S.pop()
        if w in target_set:
            coeff = (delta[w] + 1.0) / sigma[w]
        else:
            coeff = delta[w] / sigma[w]
        for v in P[w]:
            delta[v] += sigma[v] * coeff
        if w != s:
            betweenness[w] += delta[w]
    return betweenness

def _accumulate_edges_subset(betweenness, S, P, sigma, s, targets):
    if False:
        print('Hello World!')
    'edge_betweenness_centrality_subset helper.'
    delta = dict.fromkeys(S, 0)
    target_set = set(targets)
    while S:
        w = S.pop()
        for v in P[w]:
            if w in target_set:
                c = sigma[v] / sigma[w] * (1.0 + delta[w])
            else:
                c = delta[w] / len(P[w])
            if (v, w) not in betweenness:
                betweenness[w, v] += c
            else:
                betweenness[v, w] += c
            delta[v] += c
        if w != s:
            betweenness[w] += delta[w]
    return betweenness

def _rescale(betweenness, n, normalized, directed=False):
    if False:
        i = 10
        return i + 15
    'betweenness_centrality_subset helper.'
    if normalized:
        if n <= 2:
            scale = None
        else:
            scale = 1.0 / ((n - 1) * (n - 2))
    elif not directed:
        scale = 0.5
    else:
        scale = None
    if scale is not None:
        for v in betweenness:
            betweenness[v] *= scale
    return betweenness

def _rescale_e(betweenness, n, normalized, directed=False):
    if False:
        return 10
    'edge_betweenness_centrality_subset helper.'
    if normalized:
        if n <= 1:
            scale = None
        else:
            scale = 1.0 / (n * (n - 1))
    elif not directed:
        scale = 0.5
    else:
        scale = None
    if scale is not None:
        for v in betweenness:
            betweenness[v] *= scale
    return betweenness