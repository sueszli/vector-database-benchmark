"""Modularity matrix of graphs.
"""
import networkx as nx
from networkx.utils import not_implemented_for
__all__ = ['modularity_matrix', 'directed_modularity_matrix']

@not_implemented_for('directed')
@not_implemented_for('multigraph')
@nx._dispatch(edge_attrs='weight')
def modularity_matrix(G, nodelist=None, weight=None):
    if False:
        for i in range(10):
            print('nop')
    'Returns the modularity matrix of G.\n\n    The modularity matrix is the matrix B = A - <A>, where A is the adjacency\n    matrix and <A> is the average adjacency matrix, assuming that the graph\n    is described by the configuration model.\n\n    More specifically, the element B_ij of B is defined as\n\n    .. math::\n        A_{ij} - {k_i k_j \\over 2 m}\n\n    where k_i is the degree of node i, and where m is the number of edges\n    in the graph. When weight is set to a name of an attribute edge, Aij, k_i,\n    k_j and m are computed using its value.\n\n    Parameters\n    ----------\n    G : Graph\n       A NetworkX graph\n\n    nodelist : list, optional\n       The rows and columns are ordered according to the nodes in nodelist.\n       If nodelist is None, then the ordering is produced by G.nodes().\n\n    weight : string or None, optional (default=None)\n       The edge attribute that holds the numerical value used for\n       the edge weight.  If None then all edge weights are 1.\n\n    Returns\n    -------\n    B : Numpy array\n      The modularity matrix of G.\n\n    Examples\n    --------\n    >>> k = [3, 2, 2, 1, 0]\n    >>> G = nx.havel_hakimi_graph(k)\n    >>> B = nx.modularity_matrix(G)\n\n\n    See Also\n    --------\n    to_numpy_array\n    modularity_spectrum\n    adjacency_matrix\n    directed_modularity_matrix\n\n    References\n    ----------\n    .. [1] M. E. J. Newman, "Modularity and community structure in networks",\n           Proc. Natl. Acad. Sci. USA, vol. 103, pp. 8577-8582, 2006.\n    '
    import numpy as np
    if nodelist is None:
        nodelist = list(G)
    A = nx.to_scipy_sparse_array(G, nodelist=nodelist, weight=weight, format='csr')
    k = A.sum(axis=1)
    m = k.sum() * 0.5
    X = np.outer(k, k) / (2 * m)
    return A - X

@not_implemented_for('undirected')
@not_implemented_for('multigraph')
@nx._dispatch(edge_attrs='weight')
def directed_modularity_matrix(G, nodelist=None, weight=None):
    if False:
        return 10
    'Returns the directed modularity matrix of G.\n\n    The modularity matrix is the matrix B = A - <A>, where A is the adjacency\n    matrix and <A> is the expected adjacency matrix, assuming that the graph\n    is described by the configuration model.\n\n    More specifically, the element B_ij of B is defined as\n\n    .. math::\n        B_{ij} = A_{ij} - k_i^{out} k_j^{in} / m\n\n    where :math:`k_i^{in}` is the in degree of node i, and :math:`k_j^{out}` is the out degree\n    of node j, with m the number of edges in the graph. When weight is set\n    to a name of an attribute edge, Aij, k_i, k_j and m are computed using\n    its value.\n\n    Parameters\n    ----------\n    G : DiGraph\n       A NetworkX DiGraph\n\n    nodelist : list, optional\n       The rows and columns are ordered according to the nodes in nodelist.\n       If nodelist is None, then the ordering is produced by G.nodes().\n\n    weight : string or None, optional (default=None)\n       The edge attribute that holds the numerical value used for\n       the edge weight.  If None then all edge weights are 1.\n\n    Returns\n    -------\n    B : Numpy array\n      The modularity matrix of G.\n\n    Examples\n    --------\n    >>> G = nx.DiGraph()\n    >>> G.add_edges_from(\n    ...     (\n    ...         (1, 2),\n    ...         (1, 3),\n    ...         (3, 1),\n    ...         (3, 2),\n    ...         (3, 5),\n    ...         (4, 5),\n    ...         (4, 6),\n    ...         (5, 4),\n    ...         (5, 6),\n    ...         (6, 4),\n    ...     )\n    ... )\n    >>> B = nx.directed_modularity_matrix(G)\n\n\n    Notes\n    -----\n    NetworkX defines the element A_ij of the adjacency matrix as 1 if there\n    is a link going from node i to node j. Leicht and Newman use the opposite\n    definition. This explains the different expression for B_ij.\n\n    See Also\n    --------\n    to_numpy_array\n    modularity_spectrum\n    adjacency_matrix\n    modularity_matrix\n\n    References\n    ----------\n    .. [1] E. A. Leicht, M. E. J. Newman,\n        "Community structure in directed networks",\n        Phys. Rev Lett., vol. 100, no. 11, p. 118703, 2008.\n    '
    import numpy as np
    if nodelist is None:
        nodelist = list(G)
    A = nx.to_scipy_sparse_array(G, nodelist=nodelist, weight=weight, format='csr')
    k_in = A.sum(axis=0)
    k_out = A.sum(axis=1)
    m = k_in.sum()
    X = np.outer(k_out, k_in) / m
    return A - X