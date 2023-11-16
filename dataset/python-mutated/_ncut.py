import networkx as nx
import numpy as np
from scipy import sparse
from . import _ncut_cy

def DW_matrices(graph):
    if False:
        while True:
            i = 10
    'Returns the diagonal and weight matrices of a graph.\n\n    Parameters\n    ----------\n    graph : RAG\n        A Region Adjacency Graph.\n\n    Returns\n    -------\n    D : csc_matrix\n        The diagonal matrix of the graph. ``D[i, i]`` is the sum of weights of\n        all edges incident on `i`. All other entries are `0`.\n    W : csc_matrix\n        The weight matrix of the graph. ``W[i, j]`` is the weight of the edge\n        joining `i` to `j`.\n    '
    W = nx.to_scipy_sparse_array(graph, format='csc')
    entries = W.sum(axis=0)
    D = sparse.dia_matrix((entries, 0), shape=W.shape).tocsc()
    return (D, W)

def ncut_cost(cut, D, W):
    if False:
        return 10
    'Returns the N-cut cost of a bi-partition of a graph.\n\n    Parameters\n    ----------\n    cut : ndarray\n        The mask for the nodes in the graph. Nodes corresponding to a `True`\n        value are in one set.\n    D : csc_matrix\n        The diagonal matrix of the graph.\n    W : csc_matrix\n        The weight matrix of the graph.\n\n    Returns\n    -------\n    cost : float\n        The cost of performing the N-cut.\n\n    References\n    ----------\n    .. [1] Normalized Cuts and Image Segmentation, Jianbo Shi and\n           Jitendra Malik, IEEE Transactions on Pattern Analysis and Machine\n           Intelligence, Page 889, Equation 2.\n    '
    cut = np.array(cut)
    cut_cost = _ncut_cy.cut_cost(cut, W.data, W.indices, W.indptr, num_cols=W.shape[0])
    assoc_a = D.data[cut].sum()
    assoc_b = D.data[~cut].sum()
    return cut_cost / assoc_a + cut_cost / assoc_b