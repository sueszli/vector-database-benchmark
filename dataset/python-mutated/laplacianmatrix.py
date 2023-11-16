"""Laplacian matrix of graphs.
"""
import networkx as nx
from networkx.utils import not_implemented_for
__all__ = ['laplacian_matrix', 'normalized_laplacian_matrix', 'total_spanning_tree_weight', 'directed_laplacian_matrix', 'directed_combinatorial_laplacian_matrix']

@not_implemented_for('directed')
@nx._dispatch(edge_attrs='weight')
def laplacian_matrix(G, nodelist=None, weight='weight'):
    if False:
        print('Hello World!')
    "Returns the Laplacian matrix of G.\n\n    The graph Laplacian is the matrix L = D - A, where\n    A is the adjacency matrix and D is the diagonal matrix of node degrees.\n\n    Parameters\n    ----------\n    G : graph\n       A NetworkX graph\n\n    nodelist : list, optional\n       The rows and columns are ordered according to the nodes in nodelist.\n       If nodelist is None, then the ordering is produced by G.nodes().\n\n    weight : string or None, optional (default='weight')\n       The edge data key used to compute each value in the matrix.\n       If None, then each edge has weight 1.\n\n    Returns\n    -------\n    L : SciPy sparse array\n      The Laplacian matrix of G.\n\n    Notes\n    -----\n    For MultiGraph, the edges weights are summed.\n\n    See Also\n    --------\n    :func:`~networkx.convert_matrix.to_numpy_array`\n    normalized_laplacian_matrix\n    :func:`~networkx.linalg.spectrum.laplacian_spectrum`\n\n    Examples\n    --------\n    For graphs with multiple connected components, L is permutation-similar\n    to a block diagonal matrix where each block is the respective Laplacian\n    matrix for each component.\n\n    >>> G = nx.Graph([(1, 2), (2, 3), (4, 5)])\n    >>> print(nx.laplacian_matrix(G).toarray())\n    [[ 1 -1  0  0  0]\n     [-1  2 -1  0  0]\n     [ 0 -1  1  0  0]\n     [ 0  0  0  1 -1]\n     [ 0  0  0 -1  1]]\n\n    "
    import scipy as sp
    if nodelist is None:
        nodelist = list(G)
    A = nx.to_scipy_sparse_array(G, nodelist=nodelist, weight=weight, format='csr')
    (n, m) = A.shape
    D = sp.sparse.csr_array(sp.sparse.spdiags(A.sum(axis=1), 0, m, n, format='csr'))
    return D - A

@not_implemented_for('directed')
@nx._dispatch(edge_attrs='weight')
def normalized_laplacian_matrix(G, nodelist=None, weight='weight'):
    if False:
        i = 10
        return i + 15
    "Returns the normalized Laplacian matrix of G.\n\n    The normalized graph Laplacian is the matrix\n\n    .. math::\n\n        N = D^{-1/2} L D^{-1/2}\n\n    where `L` is the graph Laplacian and `D` is the diagonal matrix of\n    node degrees [1]_.\n\n    Parameters\n    ----------\n    G : graph\n       A NetworkX graph\n\n    nodelist : list, optional\n       The rows and columns are ordered according to the nodes in nodelist.\n       If nodelist is None, then the ordering is produced by G.nodes().\n\n    weight : string or None, optional (default='weight')\n       The edge data key used to compute each value in the matrix.\n       If None, then each edge has weight 1.\n\n    Returns\n    -------\n    N : SciPy sparse array\n      The normalized Laplacian matrix of G.\n\n    Notes\n    -----\n    For MultiGraph, the edges weights are summed.\n    See :func:`to_numpy_array` for other options.\n\n    If the Graph contains selfloops, D is defined as ``diag(sum(A, 1))``, where A is\n    the adjacency matrix [2]_.\n\n    See Also\n    --------\n    laplacian_matrix\n    normalized_laplacian_spectrum\n\n    References\n    ----------\n    .. [1] Fan Chung-Graham, Spectral Graph Theory,\n       CBMS Regional Conference Series in Mathematics, Number 92, 1997.\n    .. [2] Steve Butler, Interlacing For Weighted Graphs Using The Normalized\n       Laplacian, Electronic Journal of Linear Algebra, Volume 16, pp. 90-98,\n       March 2007.\n    "
    import numpy as np
    import scipy as sp
    if nodelist is None:
        nodelist = list(G)
    A = nx.to_scipy_sparse_array(G, nodelist=nodelist, weight=weight, format='csr')
    (n, m) = A.shape
    diags = A.sum(axis=1)
    D = sp.sparse.csr_array(sp.sparse.spdiags(diags, 0, m, n, format='csr'))
    L = D - A
    with np.errstate(divide='ignore'):
        diags_sqrt = 1.0 / np.sqrt(diags)
    diags_sqrt[np.isinf(diags_sqrt)] = 0
    DH = sp.sparse.csr_array(sp.sparse.spdiags(diags_sqrt, 0, m, n, format='csr'))
    return DH @ (L @ DH)

@nx._dispatch(edge_attrs='weight')
def total_spanning_tree_weight(G, weight=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Returns the total weight of all spanning trees of `G`.\n\n    Kirchoff's Tree Matrix Theorem states that the determinant of any cofactor of the\n    Laplacian matrix of a graph is the number of spanning trees in the graph. For a\n    weighted Laplacian matrix, it is the sum across all spanning trees of the\n    multiplicative weight of each tree. That is, the weight of each tree is the\n    product of its edge weights.\n\n    Parameters\n    ----------\n    G : NetworkX Graph\n        The graph to use Kirchhoff's theorem on.\n\n    weight : string or None\n        The key for the edge attribute holding the edge weight. If `None`, then\n        each edge is assumed to have a weight of 1 and this function returns the\n        total number of spanning trees in `G`.\n\n    Returns\n    -------\n    float\n        The sum of the total multiplicative weights for all spanning trees in `G`\n    "
    import numpy as np
    G_laplacian = nx.laplacian_matrix(G, weight=weight).toarray()
    return abs(np.linalg.det(G_laplacian[1:, 1:]))

@not_implemented_for('undirected')
@not_implemented_for('multigraph')
@nx._dispatch(edge_attrs='weight')
def directed_laplacian_matrix(G, nodelist=None, weight='weight', walk_type=None, alpha=0.95):
    if False:
        for i in range(10):
            print('nop')
    "Returns the directed Laplacian matrix of G.\n\n    The graph directed Laplacian is the matrix\n\n    .. math::\n\n        L = I - (\\Phi^{1/2} P \\Phi^{-1/2} + \\Phi^{-1/2} P^T \\Phi^{1/2} ) / 2\n\n    where `I` is the identity matrix, `P` is the transition matrix of the\n    graph, and `\\Phi` a matrix with the Perron vector of `P` in the diagonal and\n    zeros elsewhere [1]_.\n\n    Depending on the value of walk_type, `P` can be the transition matrix\n    induced by a random walk, a lazy random walk, or a random walk with\n    teleportation (PageRank).\n\n    Parameters\n    ----------\n    G : DiGraph\n       A NetworkX graph\n\n    nodelist : list, optional\n       The rows and columns are ordered according to the nodes in nodelist.\n       If nodelist is None, then the ordering is produced by G.nodes().\n\n    weight : string or None, optional (default='weight')\n       The edge data key used to compute each value in the matrix.\n       If None, then each edge has weight 1.\n\n    walk_type : string or None, optional (default=None)\n       If None, `P` is selected depending on the properties of the\n       graph. Otherwise is one of 'random', 'lazy', or 'pagerank'\n\n    alpha : real\n       (1 - alpha) is the teleportation probability used with pagerank\n\n    Returns\n    -------\n    L : NumPy matrix\n      Normalized Laplacian of G.\n\n    Notes\n    -----\n    Only implemented for DiGraphs\n\n    See Also\n    --------\n    laplacian_matrix\n\n    References\n    ----------\n    .. [1] Fan Chung (2005).\n       Laplacians and the Cheeger inequality for directed graphs.\n       Annals of Combinatorics, 9(1), 2005\n    "
    import numpy as np
    import scipy as sp
    P = _transition_matrix(G, nodelist=nodelist, weight=weight, walk_type=walk_type, alpha=alpha)
    (n, m) = P.shape
    (evals, evecs) = sp.sparse.linalg.eigs(P.T, k=1)
    v = evecs.flatten().real
    p = v / v.sum()
    sqrtp = np.sqrt(np.abs(p))
    Q = sp.sparse.csr_array(sp.sparse.spdiags(sqrtp, 0, n, n)) @ P @ sp.sparse.csr_array(sp.sparse.spdiags(1.0 / sqrtp, 0, n, n))
    I = np.identity(len(G))
    return I - (Q + Q.T) / 2.0

@not_implemented_for('undirected')
@not_implemented_for('multigraph')
@nx._dispatch(edge_attrs='weight')
def directed_combinatorial_laplacian_matrix(G, nodelist=None, weight='weight', walk_type=None, alpha=0.95):
    if False:
        return 10
    "Return the directed combinatorial Laplacian matrix of G.\n\n    The graph directed combinatorial Laplacian is the matrix\n\n    .. math::\n\n        L = \\Phi - (\\Phi P + P^T \\Phi) / 2\n\n    where `P` is the transition matrix of the graph and `\\Phi` a matrix\n    with the Perron vector of `P` in the diagonal and zeros elsewhere [1]_.\n\n    Depending on the value of walk_type, `P` can be the transition matrix\n    induced by a random walk, a lazy random walk, or a random walk with\n    teleportation (PageRank).\n\n    Parameters\n    ----------\n    G : DiGraph\n       A NetworkX graph\n\n    nodelist : list, optional\n       The rows and columns are ordered according to the nodes in nodelist.\n       If nodelist is None, then the ordering is produced by G.nodes().\n\n    weight : string or None, optional (default='weight')\n       The edge data key used to compute each value in the matrix.\n       If None, then each edge has weight 1.\n\n    walk_type : string or None, optional (default=None)\n       If None, `P` is selected depending on the properties of the\n       graph. Otherwise is one of 'random', 'lazy', or 'pagerank'\n\n    alpha : real\n       (1 - alpha) is the teleportation probability used with pagerank\n\n    Returns\n    -------\n    L : NumPy matrix\n      Combinatorial Laplacian of G.\n\n    Notes\n    -----\n    Only implemented for DiGraphs\n\n    See Also\n    --------\n    laplacian_matrix\n\n    References\n    ----------\n    .. [1] Fan Chung (2005).\n       Laplacians and the Cheeger inequality for directed graphs.\n       Annals of Combinatorics, 9(1), 2005\n    "
    import scipy as sp
    P = _transition_matrix(G, nodelist=nodelist, weight=weight, walk_type=walk_type, alpha=alpha)
    (n, m) = P.shape
    (evals, evecs) = sp.sparse.linalg.eigs(P.T, k=1)
    v = evecs.flatten().real
    p = v / v.sum()
    Phi = sp.sparse.csr_array(sp.sparse.spdiags(p, 0, n, n)).toarray()
    return Phi - (Phi @ P + P.T @ Phi) / 2.0

def _transition_matrix(G, nodelist=None, weight='weight', walk_type=None, alpha=0.95):
    if False:
        i = 10
        return i + 15
    "Returns the transition matrix of G.\n\n    This is a row stochastic giving the transition probabilities while\n    performing a random walk on the graph. Depending on the value of walk_type,\n    P can be the transition matrix induced by a random walk, a lazy random walk,\n    or a random walk with teleportation (PageRank).\n\n    Parameters\n    ----------\n    G : DiGraph\n       A NetworkX graph\n\n    nodelist : list, optional\n       The rows and columns are ordered according to the nodes in nodelist.\n       If nodelist is None, then the ordering is produced by G.nodes().\n\n    weight : string or None, optional (default='weight')\n       The edge data key used to compute each value in the matrix.\n       If None, then each edge has weight 1.\n\n    walk_type : string or None, optional (default=None)\n       If None, `P` is selected depending on the properties of the\n       graph. Otherwise is one of 'random', 'lazy', or 'pagerank'\n\n    alpha : real\n       (1 - alpha) is the teleportation probability used with pagerank\n\n    Returns\n    -------\n    P : numpy.ndarray\n      transition matrix of G.\n\n    Raises\n    ------\n    NetworkXError\n        If walk_type not specified or alpha not in valid range\n    "
    import numpy as np
    import scipy as sp
    if walk_type is None:
        if nx.is_strongly_connected(G):
            if nx.is_aperiodic(G):
                walk_type = 'random'
            else:
                walk_type = 'lazy'
        else:
            walk_type = 'pagerank'
    A = nx.to_scipy_sparse_array(G, nodelist=nodelist, weight=weight, dtype=float)
    (n, m) = A.shape
    if walk_type in ['random', 'lazy']:
        DI = sp.sparse.csr_array(sp.sparse.spdiags(1.0 / A.sum(axis=1), 0, n, n))
        if walk_type == 'random':
            P = DI @ A
        else:
            I = sp.sparse.csr_array(sp.sparse.identity(n))
            P = (I + DI @ A) / 2.0
    elif walk_type == 'pagerank':
        if not 0 < alpha < 1:
            raise nx.NetworkXError('alpha must be between 0 and 1')
        A = A.toarray()
        A[A.sum(axis=1) == 0, :] = 1 / n
        A = A / A.sum(axis=1)[np.newaxis, :].T
        P = alpha * A + (1 - alpha) / n
    else:
        raise nx.NetworkXError('walk_type must be random, lazy, or pagerank')
    return P