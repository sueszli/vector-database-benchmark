"""Betweenness centrality measures."""
from collections import deque
from heapq import heappop, heappush
from itertools import count
import networkx as nx
from networkx.algorithms.shortest_paths.weighted import _weight_function
from networkx.utils import py_random_state
from networkx.utils.decorators import not_implemented_for
__all__ = ['betweenness_centrality', 'edge_betweenness_centrality']

@py_random_state(5)
@nx._dispatch(edge_attrs='weight')
def betweenness_centrality(G, k=None, normalized=True, weight=None, endpoints=False, seed=None):
    if False:
        while True:
            i = 10
    'Compute the shortest-path betweenness centrality for nodes.\n\n    Betweenness centrality of a node $v$ is the sum of the\n    fraction of all-pairs shortest paths that pass through $v$\n\n    .. math::\n\n       c_B(v) =\\sum_{s,t \\in V} \\frac{\\sigma(s, t|v)}{\\sigma(s, t)}\n\n    where $V$ is the set of nodes, $\\sigma(s, t)$ is the number of\n    shortest $(s, t)$-paths,  and $\\sigma(s, t|v)$ is the number of\n    those paths  passing through some  node $v$ other than $s, t$.\n    If $s = t$, $\\sigma(s, t) = 1$, and if $v \\in {s, t}$,\n    $\\sigma(s, t|v) = 0$ [2]_.\n\n    Parameters\n    ----------\n    G : graph\n      A NetworkX graph.\n\n    k : int, optional (default=None)\n      If k is not None use k node samples to estimate betweenness.\n      The value of k <= n where n is the number of nodes in the graph.\n      Higher values give better approximation.\n\n    normalized : bool, optional\n      If True the betweenness values are normalized by `2/((n-1)(n-2))`\n      for graphs, and `1/((n-1)(n-2))` for directed graphs where `n`\n      is the number of nodes in G.\n\n    weight : None or string, optional (default=None)\n      If None, all edge weights are considered equal.\n      Otherwise holds the name of the edge attribute used as weight.\n      Weights are used to calculate weighted shortest paths, so they are\n      interpreted as distances.\n\n    endpoints : bool, optional\n      If True include the endpoints in the shortest path counts.\n\n    seed : integer, random_state, or None (default)\n        Indicator of random number generation state.\n        See :ref:`Randomness<randomness>`.\n        Note that this is only used if k is not None.\n\n    Returns\n    -------\n    nodes : dictionary\n       Dictionary of nodes with betweenness centrality as the value.\n\n    See Also\n    --------\n    edge_betweenness_centrality\n    load_centrality\n\n    Notes\n    -----\n    The algorithm is from Ulrik Brandes [1]_.\n    See [4]_ for the original first published version and [2]_ for details on\n    algorithms for variations and related metrics.\n\n    For approximate betweenness calculations set k=#samples to use\n    k nodes ("pivots") to estimate the betweenness values. For an estimate\n    of the number of pivots needed see [3]_.\n\n    For weighted graphs the edge weights must be greater than zero.\n    Zero edge weights can produce an infinite number of equal length\n    paths between pairs of nodes.\n\n    The total number of paths between source and target is counted\n    differently for directed and undirected graphs. Directed paths\n    are easy to count. Undirected paths are tricky: should a path\n    from "u" to "v" count as 1 undirected path or as 2 directed paths?\n\n    For betweenness_centrality we report the number of undirected\n    paths when G is undirected.\n\n    For betweenness_centrality_subset the reporting is different.\n    If the source and target subsets are the same, then we want\n    to count undirected paths. But if the source and target subsets\n    differ -- for example, if sources is {0} and targets is {1},\n    then we are only counting the paths in one direction. They are\n    undirected paths but we are counting them in a directed way.\n    To count them as undirected paths, each should count as half a path.\n\n    This algorithm is not guaranteed to be correct if edge weights\n    are floating point numbers. As a workaround you can use integer\n    numbers by multiplying the relevant edge attributes by a convenient\n    constant factor (eg 100) and converting to integers.\n\n    References\n    ----------\n    .. [1] Ulrik Brandes:\n       A Faster Algorithm for Betweenness Centrality.\n       Journal of Mathematical Sociology 25(2):163-177, 2001.\n       https://doi.org/10.1080/0022250X.2001.9990249\n    .. [2] Ulrik Brandes:\n       On Variants of Shortest-Path Betweenness\n       Centrality and their Generic Computation.\n       Social Networks 30(2):136-145, 2008.\n       https://doi.org/10.1016/j.socnet.2007.11.001\n    .. [3] Ulrik Brandes and Christian Pich:\n       Centrality Estimation in Large Networks.\n       International Journal of Bifurcation and Chaos 17(7):2303-2318, 2007.\n       https://dx.doi.org/10.1142/S0218127407018403\n    .. [4] Linton C. Freeman:\n       A set of measures of centrality based on betweenness.\n       Sociometry 40: 35–41, 1977\n       https://doi.org/10.2307/3033543\n    '
    betweenness = dict.fromkeys(G, 0.0)
    if k is None:
        nodes = G
    else:
        nodes = seed.sample(list(G.nodes()), k)
    for s in nodes:
        if weight is None:
            (S, P, sigma, _) = _single_source_shortest_path_basic(G, s)
        else:
            (S, P, sigma, _) = _single_source_dijkstra_path_basic(G, s, weight)
        if endpoints:
            (betweenness, _) = _accumulate_endpoints(betweenness, S, P, sigma, s)
        else:
            (betweenness, _) = _accumulate_basic(betweenness, S, P, sigma, s)
    betweenness = _rescale(betweenness, len(G), normalized=normalized, directed=G.is_directed(), k=k, endpoints=endpoints)
    return betweenness

@py_random_state(4)
@nx._dispatch(edge_attrs='weight')
def edge_betweenness_centrality(G, k=None, normalized=True, weight=None, seed=None):
    if False:
        for i in range(10):
            print('nop')
    'Compute betweenness centrality for edges.\n\n    Betweenness centrality of an edge $e$ is the sum of the\n    fraction of all-pairs shortest paths that pass through $e$\n\n    .. math::\n\n       c_B(e) =\\sum_{s,t \\in V} \\frac{\\sigma(s, t|e)}{\\sigma(s, t)}\n\n    where $V$ is the set of nodes, $\\sigma(s, t)$ is the number of\n    shortest $(s, t)$-paths, and $\\sigma(s, t|e)$ is the number of\n    those paths passing through edge $e$ [2]_.\n\n    Parameters\n    ----------\n    G : graph\n      A NetworkX graph.\n\n    k : int, optional (default=None)\n      If k is not None use k node samples to estimate betweenness.\n      The value of k <= n where n is the number of nodes in the graph.\n      Higher values give better approximation.\n\n    normalized : bool, optional\n      If True the betweenness values are normalized by $2/(n(n-1))$\n      for graphs, and $1/(n(n-1))$ for directed graphs where $n$\n      is the number of nodes in G.\n\n    weight : None or string, optional (default=None)\n      If None, all edge weights are considered equal.\n      Otherwise holds the name of the edge attribute used as weight.\n      Weights are used to calculate weighted shortest paths, so they are\n      interpreted as distances.\n\n    seed : integer, random_state, or None (default)\n        Indicator of random number generation state.\n        See :ref:`Randomness<randomness>`.\n        Note that this is only used if k is not None.\n\n    Returns\n    -------\n    edges : dictionary\n       Dictionary of edges with betweenness centrality as the value.\n\n    See Also\n    --------\n    betweenness_centrality\n    edge_load\n\n    Notes\n    -----\n    The algorithm is from Ulrik Brandes [1]_.\n\n    For weighted graphs the edge weights must be greater than zero.\n    Zero edge weights can produce an infinite number of equal length\n    paths between pairs of nodes.\n\n    References\n    ----------\n    .. [1]  A Faster Algorithm for Betweenness Centrality. Ulrik Brandes,\n       Journal of Mathematical Sociology 25(2):163-177, 2001.\n       https://doi.org/10.1080/0022250X.2001.9990249\n    .. [2] Ulrik Brandes: On Variants of Shortest-Path Betweenness\n       Centrality and their Generic Computation.\n       Social Networks 30(2):136-145, 2008.\n       https://doi.org/10.1016/j.socnet.2007.11.001\n    '
    betweenness = dict.fromkeys(G, 0.0)
    betweenness.update(dict.fromkeys(G.edges(), 0.0))
    if k is None:
        nodes = G
    else:
        nodes = seed.sample(list(G.nodes()), k)
    for s in nodes:
        if weight is None:
            (S, P, sigma, _) = _single_source_shortest_path_basic(G, s)
        else:
            (S, P, sigma, _) = _single_source_dijkstra_path_basic(G, s, weight)
        betweenness = _accumulate_edges(betweenness, S, P, sigma, s)
    for n in G:
        del betweenness[n]
    betweenness = _rescale_e(betweenness, len(G), normalized=normalized, directed=G.is_directed())
    if G.is_multigraph():
        betweenness = _add_edge_keys(G, betweenness, weight=weight)
    return betweenness

def _single_source_shortest_path_basic(G, s):
    if False:
        i = 10
        return i + 15
    S = []
    P = {}
    for v in G:
        P[v] = []
    sigma = dict.fromkeys(G, 0.0)
    D = {}
    sigma[s] = 1.0
    D[s] = 0
    Q = deque([s])
    while Q:
        v = Q.popleft()
        S.append(v)
        Dv = D[v]
        sigmav = sigma[v]
        for w in G[v]:
            if w not in D:
                Q.append(w)
                D[w] = Dv + 1
            if D[w] == Dv + 1:
                sigma[w] += sigmav
                P[w].append(v)
    return (S, P, sigma, D)

def _single_source_dijkstra_path_basic(G, s, weight):
    if False:
        return 10
    weight = _weight_function(G, weight)
    S = []
    P = {}
    for v in G:
        P[v] = []
    sigma = dict.fromkeys(G, 0.0)
    D = {}
    sigma[s] = 1.0
    push = heappush
    pop = heappop
    seen = {s: 0}
    c = count()
    Q = []
    push(Q, (0, next(c), s, s))
    while Q:
        (dist, _, pred, v) = pop(Q)
        if v in D:
            continue
        sigma[v] += sigma[pred]
        S.append(v)
        D[v] = dist
        for (w, edgedata) in G[v].items():
            vw_dist = dist + weight(v, w, edgedata)
            if w not in D and (w not in seen or vw_dist < seen[w]):
                seen[w] = vw_dist
                push(Q, (vw_dist, next(c), v, w))
                sigma[w] = 0.0
                P[w] = [v]
            elif vw_dist == seen[w]:
                sigma[w] += sigma[v]
                P[w].append(v)
    return (S, P, sigma, D)

def _accumulate_basic(betweenness, S, P, sigma, s):
    if False:
        print('Hello World!')
    delta = dict.fromkeys(S, 0)
    while S:
        w = S.pop()
        coeff = (1 + delta[w]) / sigma[w]
        for v in P[w]:
            delta[v] += sigma[v] * coeff
        if w != s:
            betweenness[w] += delta[w]
    return (betweenness, delta)

def _accumulate_endpoints(betweenness, S, P, sigma, s):
    if False:
        return 10
    betweenness[s] += len(S) - 1
    delta = dict.fromkeys(S, 0)
    while S:
        w = S.pop()
        coeff = (1 + delta[w]) / sigma[w]
        for v in P[w]:
            delta[v] += sigma[v] * coeff
        if w != s:
            betweenness[w] += delta[w] + 1
    return (betweenness, delta)

def _accumulate_edges(betweenness, S, P, sigma, s):
    if False:
        for i in range(10):
            print('nop')
    delta = dict.fromkeys(S, 0)
    while S:
        w = S.pop()
        coeff = (1 + delta[w]) / sigma[w]
        for v in P[w]:
            c = sigma[v] * coeff
            if (v, w) not in betweenness:
                betweenness[w, v] += c
            else:
                betweenness[v, w] += c
            delta[v] += c
        if w != s:
            betweenness[w] += delta[w]
    return betweenness

def _rescale(betweenness, n, normalized, directed=False, k=None, endpoints=False):
    if False:
        for i in range(10):
            print('nop')
    if normalized:
        if endpoints:
            if n < 2:
                scale = None
            else:
                scale = 1 / (n * (n - 1))
        elif n <= 2:
            scale = None
        else:
            scale = 1 / ((n - 1) * (n - 2))
    elif not directed:
        scale = 0.5
    else:
        scale = None
    if scale is not None:
        if k is not None:
            scale = scale * n / k
        for v in betweenness:
            betweenness[v] *= scale
    return betweenness

def _rescale_e(betweenness, n, normalized, directed=False, k=None):
    if False:
        return 10
    if normalized:
        if n <= 1:
            scale = None
        else:
            scale = 1 / (n * (n - 1))
    elif not directed:
        scale = 0.5
    else:
        scale = None
    if scale is not None:
        if k is not None:
            scale = scale * n / k
        for v in betweenness:
            betweenness[v] *= scale
    return betweenness

@not_implemented_for('graph')
def _add_edge_keys(G, betweenness, weight=None):
    if False:
        while True:
            i = 10
    'Adds the corrected betweenness centrality (BC) values for multigraphs.\n\n    Parameters\n    ----------\n    G : NetworkX graph.\n\n    betweenness : dictionary\n        Dictionary mapping adjacent node tuples to betweenness centrality values.\n\n    weight : string or function\n        See `_weight_function` for details. Defaults to `None`.\n\n    Returns\n    -------\n    edges : dictionary\n        The parameter `betweenness` including edges with keys and their\n        betweenness centrality values.\n\n    The BC value is divided among edges of equal weight.\n    '
    _weight = _weight_function(G, weight)
    edge_bc = dict.fromkeys(G.edges, 0.0)
    for (u, v) in betweenness:
        d = G[u][v]
        wt = _weight(u, v, d)
        keys = [k for k in d if _weight(u, v, {k: d[k]}) == wt]
        bc = betweenness[u, v] / len(keys)
        for k in keys:
            edge_bc[u, v, k] = bc
    return edge_bc