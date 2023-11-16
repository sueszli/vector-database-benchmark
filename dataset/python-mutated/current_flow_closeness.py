"""Current-flow closeness centrality measures."""
import networkx as nx
from networkx.algorithms.centrality.flow_matrix import CGInverseLaplacian, FullInverseLaplacian, SuperLUInverseLaplacian
from networkx.utils import not_implemented_for, reverse_cuthill_mckee_ordering
__all__ = ['current_flow_closeness_centrality', 'information_centrality']

@not_implemented_for('directed')
@nx._dispatch(edge_attrs='weight')
def current_flow_closeness_centrality(G, weight=None, dtype=float, solver='lu'):
    if False:
        i = 10
        return i + 15
    'Compute current-flow closeness centrality for nodes.\n\n    Current-flow closeness centrality is variant of closeness\n    centrality based on effective resistance between nodes in\n    a network. This metric is also known as information centrality.\n\n    Parameters\n    ----------\n    G : graph\n      A NetworkX graph.\n\n    weight : None or string, optional (default=None)\n      If None, all edge weights are considered equal.\n      Otherwise holds the name of the edge attribute used as weight.\n      The weight reflects the capacity or the strength of the\n      edge.\n\n    dtype: data type (default=float)\n      Default data type for internal matrices.\n      Set to np.float32 for lower memory consumption.\n\n    solver: string (default=\'lu\')\n       Type of linear solver to use for computing the flow matrix.\n       Options are "full" (uses most memory), "lu" (recommended), and\n       "cg" (uses least memory).\n\n    Returns\n    -------\n    nodes : dictionary\n       Dictionary of nodes with current flow closeness centrality as the value.\n\n    See Also\n    --------\n    closeness_centrality\n\n    Notes\n    -----\n    The algorithm is from Brandes [1]_.\n\n    See also [2]_ for the original definition of information centrality.\n\n    References\n    ----------\n    .. [1] Ulrik Brandes and Daniel Fleischer,\n       Centrality Measures Based on Current Flow.\n       Proc. 22nd Symp. Theoretical Aspects of Computer Science (STACS \'05).\n       LNCS 3404, pp. 533-544. Springer-Verlag, 2005.\n       https://doi.org/10.1007/978-3-540-31856-9_44\n\n    .. [2] Karen Stephenson and Marvin Zelen:\n       Rethinking centrality: Methods and examples.\n       Social Networks 11(1):1-37, 1989.\n       https://doi.org/10.1016/0378-8733(89)90016-6\n    '
    if not nx.is_connected(G):
        raise nx.NetworkXError('Graph not connected.')
    solvername = {'full': FullInverseLaplacian, 'lu': SuperLUInverseLaplacian, 'cg': CGInverseLaplacian}
    n = G.number_of_nodes()
    ordering = list(reverse_cuthill_mckee_ordering(G))
    H = nx.relabel_nodes(G, dict(zip(ordering, range(n))))
    betweenness = dict.fromkeys(H, 0.0)
    n = H.number_of_nodes()
    L = nx.laplacian_matrix(H, nodelist=range(n), weight=weight).asformat('csc')
    L = L.astype(dtype)
    C2 = solvername[solver](L, width=1, dtype=dtype)
    for v in H:
        col = C2.get_row(v)
        for w in H:
            betweenness[v] += col[v] - 2 * col[w]
            betweenness[w] += col[v]
    for v in H:
        betweenness[v] = 1 / betweenness[v]
    return {ordering[k]: v for (k, v) in betweenness.items()}
information_centrality = current_flow_closeness_centrality