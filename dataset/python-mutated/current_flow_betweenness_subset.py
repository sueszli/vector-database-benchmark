"""Current-flow betweenness centrality measures for subsets of nodes."""
import networkx as nx
from networkx.algorithms.centrality.flow_matrix import flow_matrix_row
from networkx.utils import not_implemented_for, reverse_cuthill_mckee_ordering
__all__ = ['current_flow_betweenness_centrality_subset', 'edge_current_flow_betweenness_centrality_subset']

@not_implemented_for('directed')
@nx._dispatch(edge_attrs='weight')
def current_flow_betweenness_centrality_subset(G, sources, targets, normalized=True, weight=None, dtype=float, solver='lu'):
    if False:
        i = 10
        return i + 15
    'Compute current-flow betweenness centrality for subsets of nodes.\n\n    Current-flow betweenness centrality uses an electrical current\n    model for information spreading in contrast to betweenness\n    centrality which uses shortest paths.\n\n    Current-flow betweenness centrality is also known as\n    random-walk betweenness centrality [2]_.\n\n    Parameters\n    ----------\n    G : graph\n      A NetworkX graph\n\n    sources: list of nodes\n      Nodes to use as sources for current\n\n    targets: list of nodes\n      Nodes to use as sinks for current\n\n    normalized : bool, optional (default=True)\n      If True the betweenness values are normalized by b=b/(n-1)(n-2) where\n      n is the number of nodes in G.\n\n    weight : string or None, optional (default=None)\n      Key for edge data used as the edge weight.\n      If None, then use 1 as each edge weight.\n      The weight reflects the capacity or the strength of the\n      edge.\n\n    dtype: data type (float)\n      Default data type for internal matrices.\n      Set to np.float32 for lower memory consumption.\n\n    solver: string (default=\'lu\')\n       Type of linear solver to use for computing the flow matrix.\n       Options are "full" (uses most memory), "lu" (recommended), and\n       "cg" (uses least memory).\n\n    Returns\n    -------\n    nodes : dictionary\n       Dictionary of nodes with betweenness centrality as the value.\n\n    See Also\n    --------\n    approximate_current_flow_betweenness_centrality\n    betweenness_centrality\n    edge_betweenness_centrality\n    edge_current_flow_betweenness_centrality\n\n    Notes\n    -----\n    Current-flow betweenness can be computed in $O(I(n-1)+mn \\log n)$\n    time [1]_, where $I(n-1)$ is the time needed to compute the\n    inverse Laplacian.  For a full matrix this is $O(n^3)$ but using\n    sparse methods you can achieve $O(nm{\\sqrt k})$ where $k$ is the\n    Laplacian matrix condition number.\n\n    The space required is $O(nw)$ where $w$ is the width of the sparse\n    Laplacian matrix.  Worse case is $w=n$ for $O(n^2)$.\n\n    If the edges have a \'weight\' attribute they will be used as\n    weights in this algorithm.  Unspecified weights are set to 1.\n\n    References\n    ----------\n    .. [1] Centrality Measures Based on Current Flow.\n       Ulrik Brandes and Daniel Fleischer,\n       Proc. 22nd Symp. Theoretical Aspects of Computer Science (STACS \'05).\n       LNCS 3404, pp. 533-544. Springer-Verlag, 2005.\n       https://doi.org/10.1007/978-3-540-31856-9_44\n\n    .. [2] A measure of betweenness centrality based on random walks,\n       M. E. J. Newman, Social Networks 27, 39-54 (2005).\n    '
    import numpy as np
    from networkx.utils import reverse_cuthill_mckee_ordering
    if not nx.is_connected(G):
        raise nx.NetworkXError('Graph not connected.')
    n = G.number_of_nodes()
    ordering = list(reverse_cuthill_mckee_ordering(G))
    mapping = dict(zip(ordering, range(n)))
    H = nx.relabel_nodes(G, mapping)
    betweenness = dict.fromkeys(H, 0.0)
    for (row, (s, t)) in flow_matrix_row(H, weight=weight, dtype=dtype, solver=solver):
        for ss in sources:
            i = mapping[ss]
            for tt in targets:
                j = mapping[tt]
                betweenness[s] += 0.5 * np.abs(row[i] - row[j])
                betweenness[t] += 0.5 * np.abs(row[i] - row[j])
    if normalized:
        nb = (n - 1.0) * (n - 2.0)
    else:
        nb = 2.0
    for v in H:
        betweenness[v] = betweenness[v] / nb + 1.0 / (2 - n)
    return {ordering[k]: v for (k, v) in betweenness.items()}

@not_implemented_for('directed')
@nx._dispatch(edge_attrs='weight')
def edge_current_flow_betweenness_centrality_subset(G, sources, targets, normalized=True, weight=None, dtype=float, solver='lu'):
    if False:
        i = 10
        return i + 15
    'Compute current-flow betweenness centrality for edges using subsets\n    of nodes.\n\n    Current-flow betweenness centrality uses an electrical current\n    model for information spreading in contrast to betweenness\n    centrality which uses shortest paths.\n\n    Current-flow betweenness centrality is also known as\n    random-walk betweenness centrality [2]_.\n\n    Parameters\n    ----------\n    G : graph\n      A NetworkX graph\n\n    sources: list of nodes\n      Nodes to use as sources for current\n\n    targets: list of nodes\n      Nodes to use as sinks for current\n\n    normalized : bool, optional (default=True)\n      If True the betweenness values are normalized by b=b/(n-1)(n-2) where\n      n is the number of nodes in G.\n\n    weight : string or None, optional (default=None)\n      Key for edge data used as the edge weight.\n      If None, then use 1 as each edge weight.\n      The weight reflects the capacity or the strength of the\n      edge.\n\n    dtype: data type (float)\n      Default data type for internal matrices.\n      Set to np.float32 for lower memory consumption.\n\n    solver: string (default=\'lu\')\n       Type of linear solver to use for computing the flow matrix.\n       Options are "full" (uses most memory), "lu" (recommended), and\n       "cg" (uses least memory).\n\n    Returns\n    -------\n    nodes : dict\n       Dictionary of edge tuples with betweenness centrality as the value.\n\n    See Also\n    --------\n    betweenness_centrality\n    edge_betweenness_centrality\n    current_flow_betweenness_centrality\n\n    Notes\n    -----\n    Current-flow betweenness can be computed in $O(I(n-1)+mn \\log n)$\n    time [1]_, where $I(n-1)$ is the time needed to compute the\n    inverse Laplacian.  For a full matrix this is $O(n^3)$ but using\n    sparse methods you can achieve $O(nm{\\sqrt k})$ where $k$ is the\n    Laplacian matrix condition number.\n\n    The space required is $O(nw)$ where $w$ is the width of the sparse\n    Laplacian matrix.  Worse case is $w=n$ for $O(n^2)$.\n\n    If the edges have a \'weight\' attribute they will be used as\n    weights in this algorithm.  Unspecified weights are set to 1.\n\n    References\n    ----------\n    .. [1] Centrality Measures Based on Current Flow.\n       Ulrik Brandes and Daniel Fleischer,\n       Proc. 22nd Symp. Theoretical Aspects of Computer Science (STACS \'05).\n       LNCS 3404, pp. 533-544. Springer-Verlag, 2005.\n       https://doi.org/10.1007/978-3-540-31856-9_44\n\n    .. [2] A measure of betweenness centrality based on random walks,\n       M. E. J. Newman, Social Networks 27, 39-54 (2005).\n    '
    import numpy as np
    if not nx.is_connected(G):
        raise nx.NetworkXError('Graph not connected.')
    n = G.number_of_nodes()
    ordering = list(reverse_cuthill_mckee_ordering(G))
    mapping = dict(zip(ordering, range(n)))
    H = nx.relabel_nodes(G, mapping)
    edges = (tuple(sorted((u, v))) for (u, v) in H.edges())
    betweenness = dict.fromkeys(edges, 0.0)
    if normalized:
        nb = (n - 1.0) * (n - 2.0)
    else:
        nb = 2.0
    for (row, e) in flow_matrix_row(H, weight=weight, dtype=dtype, solver=solver):
        for ss in sources:
            i = mapping[ss]
            for tt in targets:
                j = mapping[tt]
                betweenness[e] += 0.5 * np.abs(row[i] - row[j])
        betweenness[e] /= nb
    return {(ordering[s], ordering[t]): v for ((s, t), v) in betweenness.items()}