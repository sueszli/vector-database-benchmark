"""Bethe Hessian or deformed Laplacian matrix of graphs."""
import networkx as nx
from networkx.utils import not_implemented_for
__all__ = ['bethe_hessian_matrix']

@not_implemented_for('directed')
@not_implemented_for('multigraph')
@nx._dispatch
def bethe_hessian_matrix(G, r=None, nodelist=None):
    if False:
        print('Hello World!')
    'Returns the Bethe Hessian matrix of G.\n\n    The Bethe Hessian is a family of matrices parametrized by r, defined as\n    H(r) = (r^2 - 1) I - r A + D where A is the adjacency matrix, D is the\n    diagonal matrix of node degrees, and I is the identify matrix. It is equal\n    to the graph laplacian when the regularizer r = 1.\n\n    The default choice of regularizer should be the ratio [2]_\n\n    .. math::\n      r_m = \\left(\\sum k_i \\right)^{-1}\\left(\\sum k_i^2 \\right) - 1\n\n    Parameters\n    ----------\n    G : Graph\n       A NetworkX graph\n    r : float\n       Regularizer parameter\n    nodelist : list, optional\n       The rows and columns are ordered according to the nodes in nodelist.\n       If nodelist is None, then the ordering is produced by ``G.nodes()``.\n\n    Returns\n    -------\n    H : scipy.sparse.csr_array\n      The Bethe Hessian matrix of `G`, with parameter `r`.\n\n    Examples\n    --------\n    >>> k = [3, 2, 2, 1, 0]\n    >>> G = nx.havel_hakimi_graph(k)\n    >>> H = nx.bethe_hessian_matrix(G)\n    >>> H.toarray()\n    array([[ 3.5625, -1.25  , -1.25  , -1.25  ,  0.    ],\n           [-1.25  ,  2.5625, -1.25  ,  0.    ,  0.    ],\n           [-1.25  , -1.25  ,  2.5625,  0.    ,  0.    ],\n           [-1.25  ,  0.    ,  0.    ,  1.5625,  0.    ],\n           [ 0.    ,  0.    ,  0.    ,  0.    ,  0.5625]])\n\n    See Also\n    --------\n    bethe_hessian_spectrum\n    adjacency_matrix\n    laplacian_matrix\n\n    References\n    ----------\n    .. [1] A. Saade, F. Krzakala and L. Zdeborová\n       "Spectral Clustering of Graphs with the Bethe Hessian",\n       Advances in Neural Information Processing Systems, 2014.\n    .. [2] C. M. Le, E. Levina\n       "Estimating the number of communities in networks by spectral methods"\n       arXiv:1507.00827, 2015.\n    '
    import scipy as sp
    if nodelist is None:
        nodelist = list(G)
    if r is None:
        r = sum((d ** 2 for (v, d) in nx.degree(G))) / sum((d for (v, d) in nx.degree(G))) - 1
    A = nx.to_scipy_sparse_array(G, nodelist=nodelist, format='csr')
    (n, m) = A.shape
    D = sp.sparse.csr_array(sp.sparse.spdiags(A.sum(axis=1), 0, m, n, format='csr'))
    I = sp.sparse.csr_array(sp.sparse.eye(m, n, format='csr'))
    return (r ** 2 - 1) * I - r * A + D