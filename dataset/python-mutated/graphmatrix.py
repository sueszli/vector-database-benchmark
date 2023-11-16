"""
Adjacency matrix and incidence matrix of graphs.
"""
import networkx as nx
__all__ = ['incidence_matrix', 'adjacency_matrix']

@nx._dispatch(edge_attrs='weight')
def incidence_matrix(G, nodelist=None, edgelist=None, oriented=False, weight=None, *, dtype=None):
    if False:
        return 10
    'Returns incidence matrix of G.\n\n    The incidence matrix assigns each row to a node and each column to an edge.\n    For a standard incidence matrix a 1 appears wherever a row\'s node is\n    incident on the column\'s edge.  For an oriented incidence matrix each\n    edge is assigned an orientation (arbitrarily for undirected and aligning to\n    direction for directed).  A -1 appears for the source (tail) of an edge and\n    1 for the destination (head) of the edge.  The elements are zero otherwise.\n\n    Parameters\n    ----------\n    G : graph\n       A NetworkX graph\n\n    nodelist : list, optional   (default= all nodes in G)\n       The rows are ordered according to the nodes in nodelist.\n       If nodelist is None, then the ordering is produced by G.nodes().\n\n    edgelist : list, optional (default= all edges in G)\n       The columns are ordered according to the edges in edgelist.\n       If edgelist is None, then the ordering is produced by G.edges().\n\n    oriented: bool, optional (default=False)\n       If True, matrix elements are +1 or -1 for the head or tail node\n       respectively of each edge.  If False, +1 occurs at both nodes.\n\n    weight : string or None, optional (default=None)\n       The edge data key used to provide each value in the matrix.\n       If None, then each edge has weight 1.  Edge weights, if used,\n       should be positive so that the orientation can provide the sign.\n\n    dtype : a NumPy dtype or None (default=None)\n        The dtype of the output sparse array. This type should be a compatible\n        type of the weight argument, eg. if weight would return a float this\n        argument should also be a float.\n        If None, then the default for SciPy is used.\n\n    Returns\n    -------\n    A : SciPy sparse array\n      The incidence matrix of G.\n\n    Notes\n    -----\n    For MultiGraph/MultiDiGraph, the edges in edgelist should be\n    (u,v,key) 3-tuples.\n\n    "Networks are the best discrete model for so many problems in\n    applied mathematics" [1]_.\n\n    References\n    ----------\n    .. [1] Gil Strang, Network applications: A = incidence matrix,\n       http://videolectures.net/mit18085f07_strang_lec03/\n    '
    import scipy as sp
    if nodelist is None:
        nodelist = list(G)
    if edgelist is None:
        if G.is_multigraph():
            edgelist = list(G.edges(keys=True))
        else:
            edgelist = list(G.edges())
    A = sp.sparse.lil_array((len(nodelist), len(edgelist)), dtype=dtype)
    node_index = {node: i for (i, node) in enumerate(nodelist)}
    for (ei, e) in enumerate(edgelist):
        (u, v) = e[:2]
        if u == v:
            continue
        try:
            ui = node_index[u]
            vi = node_index[v]
        except KeyError as err:
            raise nx.NetworkXError(f'node {u} or {v} in edgelist but not in nodelist') from err
        if weight is None:
            wt = 1
        elif G.is_multigraph():
            ekey = e[2]
            wt = G[u][v][ekey].get(weight, 1)
        else:
            wt = G[u][v].get(weight, 1)
        if oriented:
            A[ui, ei] = -wt
            A[vi, ei] = wt
        else:
            A[ui, ei] = wt
            A[vi, ei] = wt
    return A.asformat('csc')

@nx._dispatch(edge_attrs='weight')
def adjacency_matrix(G, nodelist=None, dtype=None, weight='weight'):
    if False:
        i = 10
        return i + 15
    "Returns adjacency matrix of G.\n\n    Parameters\n    ----------\n    G : graph\n       A NetworkX graph\n\n    nodelist : list, optional\n       The rows and columns are ordered according to the nodes in nodelist.\n       If nodelist is None, then the ordering is produced by G.nodes().\n\n    dtype : NumPy data-type, optional\n        The desired data-type for the array.\n        If None, then the NumPy default is used.\n\n    weight : string or None, optional (default='weight')\n       The edge data key used to provide each value in the matrix.\n       If None, then each edge has weight 1.\n\n    Returns\n    -------\n    A : SciPy sparse array\n      Adjacency matrix representation of G.\n\n    Notes\n    -----\n    For directed graphs, entry i,j corresponds to an edge from i to j.\n\n    If you want a pure Python adjacency matrix representation try\n    networkx.convert.to_dict_of_dicts which will return a\n    dictionary-of-dictionaries format that can be addressed as a\n    sparse matrix.\n\n    For MultiGraph/MultiDiGraph with parallel edges the weights are summed.\n    See `to_numpy_array` for other options.\n\n    The convention used for self-loop edges in graphs is to assign the\n    diagonal matrix entry value to the edge weight attribute\n    (or the number 1 if the edge has no weight attribute).  If the\n    alternate convention of doubling the edge weight is desired the\n    resulting SciPy sparse array can be modified as follows:\n\n    >>> G = nx.Graph([(1, 1)])\n    >>> A = nx.adjacency_matrix(G)\n    >>> print(A.todense())\n    [[1]]\n    >>> A.setdiag(A.diagonal() * 2)\n    >>> print(A.todense())\n    [[2]]\n\n    See Also\n    --------\n    to_numpy_array\n    to_scipy_sparse_array\n    to_dict_of_dicts\n    adjacency_spectrum\n    "
    return nx.to_scipy_sparse_array(G, nodelist=nodelist, dtype=dtype, weight=weight)