"""Current-flow betweenness centrality measures."""
import networkx as nx
from networkx.algorithms.centrality.flow_matrix import CGInverseLaplacian, FullInverseLaplacian, SuperLUInverseLaplacian, flow_matrix_row
from networkx.utils import not_implemented_for, py_random_state, reverse_cuthill_mckee_ordering
__all__ = ['current_flow_betweenness_centrality', 'approximate_current_flow_betweenness_centrality', 'edge_current_flow_betweenness_centrality']

@not_implemented_for('directed')
@py_random_state(7)
@nx._dispatch(edge_attrs='weight')
def approximate_current_flow_betweenness_centrality(G, normalized=True, weight=None, dtype=float, solver='full', epsilon=0.5, kmax=10000, seed=None):
    if False:
        i = 10
        return i + 15
    'Compute the approximate current-flow betweenness centrality for nodes.\n\n    Approximates the current-flow betweenness centrality within absolute\n    error of epsilon with high probability [1]_.\n\n\n    Parameters\n    ----------\n    G : graph\n      A NetworkX graph\n\n    normalized : bool, optional (default=True)\n      If True the betweenness values are normalized by 2/[(n-1)(n-2)] where\n      n is the number of nodes in G.\n\n    weight : string or None, optional (default=None)\n      Key for edge data used as the edge weight.\n      If None, then use 1 as each edge weight.\n      The weight reflects the capacity or the strength of the\n      edge.\n\n    dtype : data type (float)\n      Default data type for internal matrices.\n      Set to np.float32 for lower memory consumption.\n\n    solver : string (default=\'full\')\n       Type of linear solver to use for computing the flow matrix.\n       Options are "full" (uses most memory), "lu" (recommended), and\n       "cg" (uses least memory).\n\n    epsilon: float\n        Absolute error tolerance.\n\n    kmax: int\n       Maximum number of sample node pairs to use for approximation.\n\n    seed : integer, random_state, or None (default)\n        Indicator of random number generation state.\n        See :ref:`Randomness<randomness>`.\n\n    Returns\n    -------\n    nodes : dictionary\n       Dictionary of nodes with betweenness centrality as the value.\n\n    See Also\n    --------\n    current_flow_betweenness_centrality\n\n    Notes\n    -----\n    The running time is $O((1/\\epsilon^2)m{\\sqrt k} \\log n)$\n    and the space required is $O(m)$ for $n$ nodes and $m$ edges.\n\n    If the edges have a \'weight\' attribute they will be used as\n    weights in this algorithm.  Unspecified weights are set to 1.\n\n    References\n    ----------\n    .. [1] Ulrik Brandes and Daniel Fleischer:\n       Centrality Measures Based on Current Flow.\n       Proc. 22nd Symp. Theoretical Aspects of Computer Science (STACS \'05).\n       LNCS 3404, pp. 533-544. Springer-Verlag, 2005.\n       https://doi.org/10.1007/978-3-540-31856-9_44\n    '
    import numpy as np
    if not nx.is_connected(G):
        raise nx.NetworkXError('Graph not connected.')
    solvername = {'full': FullInverseLaplacian, 'lu': SuperLUInverseLaplacian, 'cg': CGInverseLaplacian}
    n = G.number_of_nodes()
    ordering = list(reverse_cuthill_mckee_ordering(G))
    H = nx.relabel_nodes(G, dict(zip(ordering, range(n))))
    L = nx.laplacian_matrix(H, nodelist=range(n), weight=weight).asformat('csc')
    L = L.astype(dtype)
    C = solvername[solver](L, dtype=dtype)
    betweenness = dict.fromkeys(H, 0.0)
    nb = (n - 1.0) * (n - 2.0)
    cstar = n * (n - 1) / nb
    l = 1
    k = l * int(np.ceil((cstar / epsilon) ** 2 * np.log(n)))
    if k > kmax:
        msg = f'Number random pairs k>kmax ({k}>{kmax}) '
        raise nx.NetworkXError(msg, 'Increase kmax or epsilon')
    cstar2k = cstar / (2 * k)
    for _ in range(k):
        (s, t) = pair = seed.sample(range(n), 2)
        b = np.zeros(n, dtype=dtype)
        b[s] = 1
        b[t] = -1
        p = C.solve(b)
        for v in H:
            if v in pair:
                continue
            for nbr in H[v]:
                w = H[v][nbr].get(weight, 1.0)
                betweenness[v] += w * np.abs(p[v] - p[nbr]) * cstar2k
    if normalized:
        factor = 1.0
    else:
        factor = nb / 2.0
    return {ordering[k]: v * factor for (k, v) in betweenness.items()}

@not_implemented_for('directed')
@nx._dispatch(edge_attrs='weight')
def current_flow_betweenness_centrality(G, normalized=True, weight=None, dtype=float, solver='full'):
    if False:
        while True:
            i = 10
    'Compute current-flow betweenness centrality for nodes.\n\n    Current-flow betweenness centrality uses an electrical current\n    model for information spreading in contrast to betweenness\n    centrality which uses shortest paths.\n\n    Current-flow betweenness centrality is also known as\n    random-walk betweenness centrality [2]_.\n\n    Parameters\n    ----------\n    G : graph\n      A NetworkX graph\n\n    normalized : bool, optional (default=True)\n      If True the betweenness values are normalized by 2/[(n-1)(n-2)] where\n      n is the number of nodes in G.\n\n    weight : string or None, optional (default=None)\n      Key for edge data used as the edge weight.\n      If None, then use 1 as each edge weight.\n      The weight reflects the capacity or the strength of the\n      edge.\n\n    dtype : data type (float)\n      Default data type for internal matrices.\n      Set to np.float32 for lower memory consumption.\n\n    solver : string (default=\'full\')\n       Type of linear solver to use for computing the flow matrix.\n       Options are "full" (uses most memory), "lu" (recommended), and\n       "cg" (uses least memory).\n\n    Returns\n    -------\n    nodes : dictionary\n       Dictionary of nodes with betweenness centrality as the value.\n\n    See Also\n    --------\n    approximate_current_flow_betweenness_centrality\n    betweenness_centrality\n    edge_betweenness_centrality\n    edge_current_flow_betweenness_centrality\n\n    Notes\n    -----\n    Current-flow betweenness can be computed in  $O(I(n-1)+mn \\log n)$\n    time [1]_, where $I(n-1)$ is the time needed to compute the\n    inverse Laplacian.  For a full matrix this is $O(n^3)$ but using\n    sparse methods you can achieve $O(nm{\\sqrt k})$ where $k$ is the\n    Laplacian matrix condition number.\n\n    The space required is $O(nw)$ where $w$ is the width of the sparse\n    Laplacian matrix.  Worse case is $w=n$ for $O(n^2)$.\n\n    If the edges have a \'weight\' attribute they will be used as\n    weights in this algorithm.  Unspecified weights are set to 1.\n\n    References\n    ----------\n    .. [1] Centrality Measures Based on Current Flow.\n       Ulrik Brandes and Daniel Fleischer,\n       Proc. 22nd Symp. Theoretical Aspects of Computer Science (STACS \'05).\n       LNCS 3404, pp. 533-544. Springer-Verlag, 2005.\n       https://doi.org/10.1007/978-3-540-31856-9_44\n\n    .. [2] A measure of betweenness centrality based on random walks,\n       M. E. J. Newman, Social Networks 27, 39-54 (2005).\n    '
    if not nx.is_connected(G):
        raise nx.NetworkXError('Graph not connected.')
    n = G.number_of_nodes()
    ordering = list(reverse_cuthill_mckee_ordering(G))
    H = nx.relabel_nodes(G, dict(zip(ordering, range(n))))
    betweenness = dict.fromkeys(H, 0.0)
    for (row, (s, t)) in flow_matrix_row(H, weight=weight, dtype=dtype, solver=solver):
        pos = dict(zip(row.argsort()[::-1], range(n)))
        for i in range(n):
            betweenness[s] += (i - pos[i]) * row[i]
            betweenness[t] += (n - i - 1 - pos[i]) * row[i]
    if normalized:
        nb = (n - 1.0) * (n - 2.0)
    else:
        nb = 2.0
    for v in H:
        betweenness[v] = float((betweenness[v] - v) * 2.0 / nb)
    return {ordering[k]: v for (k, v) in betweenness.items()}

@not_implemented_for('directed')
@nx._dispatch(edge_attrs='weight')
def edge_current_flow_betweenness_centrality(G, normalized=True, weight=None, dtype=float, solver='full'):
    if False:
        for i in range(10):
            print('nop')
    'Compute current-flow betweenness centrality for edges.\n\n    Current-flow betweenness centrality uses an electrical current\n    model for information spreading in contrast to betweenness\n    centrality which uses shortest paths.\n\n    Current-flow betweenness centrality is also known as\n    random-walk betweenness centrality [2]_.\n\n    Parameters\n    ----------\n    G : graph\n      A NetworkX graph\n\n    normalized : bool, optional (default=True)\n      If True the betweenness values are normalized by 2/[(n-1)(n-2)] where\n      n is the number of nodes in G.\n\n    weight : string or None, optional (default=None)\n      Key for edge data used as the edge weight.\n      If None, then use 1 as each edge weight.\n      The weight reflects the capacity or the strength of the\n      edge.\n\n    dtype : data type (default=float)\n      Default data type for internal matrices.\n      Set to np.float32 for lower memory consumption.\n\n    solver : string (default=\'full\')\n       Type of linear solver to use for computing the flow matrix.\n       Options are "full" (uses most memory), "lu" (recommended), and\n       "cg" (uses least memory).\n\n    Returns\n    -------\n    nodes : dictionary\n       Dictionary of edge tuples with betweenness centrality as the value.\n\n    Raises\n    ------\n    NetworkXError\n        The algorithm does not support DiGraphs.\n        If the input graph is an instance of DiGraph class, NetworkXError\n        is raised.\n\n    See Also\n    --------\n    betweenness_centrality\n    edge_betweenness_centrality\n    current_flow_betweenness_centrality\n\n    Notes\n    -----\n    Current-flow betweenness can be computed in $O(I(n-1)+mn \\log n)$\n    time [1]_, where $I(n-1)$ is the time needed to compute the\n    inverse Laplacian.  For a full matrix this is $O(n^3)$ but using\n    sparse methods you can achieve $O(nm{\\sqrt k})$ where $k$ is the\n    Laplacian matrix condition number.\n\n    The space required is $O(nw)$ where $w$ is the width of the sparse\n    Laplacian matrix.  Worse case is $w=n$ for $O(n^2)$.\n\n    If the edges have a \'weight\' attribute they will be used as\n    weights in this algorithm.  Unspecified weights are set to 1.\n\n    References\n    ----------\n    .. [1] Centrality Measures Based on Current Flow.\n       Ulrik Brandes and Daniel Fleischer,\n       Proc. 22nd Symp. Theoretical Aspects of Computer Science (STACS \'05).\n       LNCS 3404, pp. 533-544. Springer-Verlag, 2005.\n       https://doi.org/10.1007/978-3-540-31856-9_44\n\n    .. [2] A measure of betweenness centrality based on random walks,\n       M. E. J. Newman, Social Networks 27, 39-54 (2005).\n    '
    if not nx.is_connected(G):
        raise nx.NetworkXError('Graph not connected.')
    n = G.number_of_nodes()
    ordering = list(reverse_cuthill_mckee_ordering(G))
    H = nx.relabel_nodes(G, dict(zip(ordering, range(n))))
    edges = (tuple(sorted((u, v))) for (u, v) in H.edges())
    betweenness = dict.fromkeys(edges, 0.0)
    if normalized:
        nb = (n - 1.0) * (n - 2.0)
    else:
        nb = 2.0
    for (row, e) in flow_matrix_row(H, weight=weight, dtype=dtype, solver=solver):
        pos = dict(zip(row.argsort()[::-1], range(1, n + 1)))
        for i in range(n):
            betweenness[e] += (i + 1 - pos[i]) * row[i]
            betweenness[e] += (n - i - pos[i]) * row[i]
        betweenness[e] /= nb
    return {(ordering[s], ordering[t]): v for ((s, t), v) in betweenness.items()}