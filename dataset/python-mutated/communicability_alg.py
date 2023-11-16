"""
Communicability.
"""
import networkx as nx
from networkx.utils import not_implemented_for
__all__ = ['communicability', 'communicability_exp']

@not_implemented_for('directed')
@not_implemented_for('multigraph')
@nx._dispatch
def communicability(G):
    if False:
        i = 10
        return i + 15
    'Returns communicability between all pairs of nodes in G.\n\n    The communicability between pairs of nodes in G is the sum of\n    walks of different lengths starting at node u and ending at node v.\n\n    Parameters\n    ----------\n    G: graph\n\n    Returns\n    -------\n    comm: dictionary of dictionaries\n        Dictionary of dictionaries keyed by nodes with communicability\n        as the value.\n\n    Raises\n    ------\n    NetworkXError\n       If the graph is not undirected and simple.\n\n    See Also\n    --------\n    communicability_exp:\n       Communicability between all pairs of nodes in G  using spectral\n       decomposition.\n    communicability_betweenness_centrality:\n       Communicability betweenness centrality for each node in G.\n\n    Notes\n    -----\n    This algorithm uses a spectral decomposition of the adjacency matrix.\n    Let G=(V,E) be a simple undirected graph.  Using the connection between\n    the powers  of the adjacency matrix and the number of walks in the graph,\n    the communicability  between nodes `u` and `v` based on the graph spectrum\n    is [1]_\n\n    .. math::\n        C(u,v)=\\sum_{j=1}^{n}\\phi_{j}(u)\\phi_{j}(v)e^{\\lambda_{j}},\n\n    where `\\phi_{j}(u)` is the `u\\rm{th}` element of the `j\\rm{th}` orthonormal\n    eigenvector of the adjacency matrix associated with the eigenvalue\n    `\\lambda_{j}`.\n\n    References\n    ----------\n    .. [1] Ernesto Estrada, Naomichi Hatano,\n       "Communicability in complex networks",\n       Phys. Rev. E 77, 036111 (2008).\n       https://arxiv.org/abs/0707.0756\n\n    Examples\n    --------\n    >>> G = nx.Graph([(0, 1), (1, 2), (1, 5), (5, 4), (2, 4), (2, 3), (4, 3), (3, 6)])\n    >>> c = nx.communicability(G)\n    '
    import numpy as np
    nodelist = list(G)
    A = nx.to_numpy_array(G, nodelist)
    A[A != 0.0] = 1
    (w, vec) = np.linalg.eigh(A)
    expw = np.exp(w)
    mapping = dict(zip(nodelist, range(len(nodelist))))
    c = {}
    for u in G:
        c[u] = {}
        for v in G:
            s = 0
            p = mapping[u]
            q = mapping[v]
            for j in range(len(nodelist)):
                s += vec[:, j][p] * vec[:, j][q] * expw[j]
            c[u][v] = float(s)
    return c

@not_implemented_for('directed')
@not_implemented_for('multigraph')
@nx._dispatch
def communicability_exp(G):
    if False:
        while True:
            i = 10
    'Returns communicability between all pairs of nodes in G.\n\n    Communicability between pair of node (u,v) of node in G is the sum of\n    walks of different lengths starting at node u and ending at node v.\n\n    Parameters\n    ----------\n    G: graph\n\n    Returns\n    -------\n    comm: dictionary of dictionaries\n        Dictionary of dictionaries keyed by nodes with communicability\n        as the value.\n\n    Raises\n    ------\n    NetworkXError\n        If the graph is not undirected and simple.\n\n    See Also\n    --------\n    communicability:\n       Communicability between pairs of nodes in G.\n    communicability_betweenness_centrality:\n       Communicability betweenness centrality for each node in G.\n\n    Notes\n    -----\n    This algorithm uses matrix exponentiation of the adjacency matrix.\n\n    Let G=(V,E) be a simple undirected graph.  Using the connection between\n    the powers  of the adjacency matrix and the number of walks in the graph,\n    the communicability between nodes u and v is [1]_,\n\n    .. math::\n        C(u,v) = (e^A)_{uv},\n\n    where `A` is the adjacency matrix of G.\n\n    References\n    ----------\n    .. [1] Ernesto Estrada, Naomichi Hatano,\n       "Communicability in complex networks",\n       Phys. Rev. E 77, 036111 (2008).\n       https://arxiv.org/abs/0707.0756\n\n    Examples\n    --------\n    >>> G = nx.Graph([(0, 1), (1, 2), (1, 5), (5, 4), (2, 4), (2, 3), (4, 3), (3, 6)])\n    >>> c = nx.communicability_exp(G)\n    '
    import scipy as sp
    nodelist = list(G)
    A = nx.to_numpy_array(G, nodelist)
    A[A != 0.0] = 1
    expA = sp.linalg.expm(A)
    mapping = dict(zip(nodelist, range(len(nodelist))))
    c = {}
    for u in G:
        c[u] = {}
        for v in G:
            c[u][v] = float(expA[mapping[u], mapping[v]])
    return c