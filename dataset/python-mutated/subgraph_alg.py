"""
Subraph centrality and communicability betweenness.
"""
import networkx as nx
from networkx.utils import not_implemented_for
__all__ = ['subgraph_centrality_exp', 'subgraph_centrality', 'communicability_betweenness_centrality', 'estrada_index']

@not_implemented_for('directed')
@not_implemented_for('multigraph')
@nx._dispatch
def subgraph_centrality_exp(G):
    if False:
        for i in range(10):
            print('nop')
    'Returns the subgraph centrality for each node of G.\n\n    Subgraph centrality  of a node `n` is the sum of weighted closed\n    walks of all lengths starting and ending at node `n`. The weights\n    decrease with path length. Each closed walk is associated with a\n    connected subgraph ([1]_).\n\n    Parameters\n    ----------\n    G: graph\n\n    Returns\n    -------\n    nodes:dictionary\n        Dictionary of nodes with subgraph centrality as the value.\n\n    Raises\n    ------\n    NetworkXError\n        If the graph is not undirected and simple.\n\n    See Also\n    --------\n    subgraph_centrality:\n        Alternative algorithm of the subgraph centrality for each node of G.\n\n    Notes\n    -----\n    This version of the algorithm exponentiates the adjacency matrix.\n\n    The subgraph centrality of a node `u` in G can be found using\n    the matrix exponential of the adjacency matrix of G [1]_,\n\n    .. math::\n\n        SC(u)=(e^A)_{uu} .\n\n    References\n    ----------\n    .. [1] Ernesto Estrada, Juan A. Rodriguez-Velazquez,\n       "Subgraph centrality in complex networks",\n       Physical Review E 71, 056103 (2005).\n       https://arxiv.org/abs/cond-mat/0504730\n\n    Examples\n    --------\n    (Example from [1]_)\n    >>> G = nx.Graph(\n    ...     [\n    ...         (1, 2),\n    ...         (1, 5),\n    ...         (1, 8),\n    ...         (2, 3),\n    ...         (2, 8),\n    ...         (3, 4),\n    ...         (3, 6),\n    ...         (4, 5),\n    ...         (4, 7),\n    ...         (5, 6),\n    ...         (6, 7),\n    ...         (7, 8),\n    ...     ]\n    ... )\n    >>> sc = nx.subgraph_centrality_exp(G)\n    >>> print([f"{node} {sc[node]:0.2f}" for node in sorted(sc)])\n    [\'1 3.90\', \'2 3.90\', \'3 3.64\', \'4 3.71\', \'5 3.64\', \'6 3.71\', \'7 3.64\', \'8 3.90\']\n    '
    import scipy as sp
    nodelist = list(G)
    A = nx.to_numpy_array(G, nodelist)
    A[A != 0.0] = 1
    expA = sp.linalg.expm(A)
    sc = dict(zip(nodelist, map(float, expA.diagonal())))
    return sc

@not_implemented_for('directed')
@not_implemented_for('multigraph')
@nx._dispatch
def subgraph_centrality(G):
    if False:
        for i in range(10):
            print('nop')
    'Returns subgraph centrality for each node in G.\n\n    Subgraph centrality  of a node `n` is the sum of weighted closed\n    walks of all lengths starting and ending at node `n`. The weights\n    decrease with path length. Each closed walk is associated with a\n    connected subgraph ([1]_).\n\n    Parameters\n    ----------\n    G: graph\n\n    Returns\n    -------\n    nodes : dictionary\n       Dictionary of nodes with subgraph centrality as the value.\n\n    Raises\n    ------\n    NetworkXError\n       If the graph is not undirected and simple.\n\n    See Also\n    --------\n    subgraph_centrality_exp:\n        Alternative algorithm of the subgraph centrality for each node of G.\n\n    Notes\n    -----\n    This version of the algorithm computes eigenvalues and eigenvectors\n    of the adjacency matrix.\n\n    Subgraph centrality of a node `u` in G can be found using\n    a spectral decomposition of the adjacency matrix [1]_,\n\n    .. math::\n\n       SC(u)=\\sum_{j=1}^{N}(v_{j}^{u})^2 e^{\\lambda_{j}},\n\n    where `v_j` is an eigenvector of the adjacency matrix `A` of G\n    corresponding to the eigenvalue `\\lambda_j`.\n\n    Examples\n    --------\n    (Example from [1]_)\n    >>> G = nx.Graph(\n    ...     [\n    ...         (1, 2),\n    ...         (1, 5),\n    ...         (1, 8),\n    ...         (2, 3),\n    ...         (2, 8),\n    ...         (3, 4),\n    ...         (3, 6),\n    ...         (4, 5),\n    ...         (4, 7),\n    ...         (5, 6),\n    ...         (6, 7),\n    ...         (7, 8),\n    ...     ]\n    ... )\n    >>> sc = nx.subgraph_centrality(G)\n    >>> print([f"{node} {sc[node]:0.2f}" for node in sorted(sc)])\n    [\'1 3.90\', \'2 3.90\', \'3 3.64\', \'4 3.71\', \'5 3.64\', \'6 3.71\', \'7 3.64\', \'8 3.90\']\n\n    References\n    ----------\n    .. [1] Ernesto Estrada, Juan A. Rodriguez-Velazquez,\n       "Subgraph centrality in complex networks",\n       Physical Review E 71, 056103 (2005).\n       https://arxiv.org/abs/cond-mat/0504730\n\n    '
    import numpy as np
    nodelist = list(G)
    A = nx.to_numpy_array(G, nodelist)
    A[np.nonzero(A)] = 1
    (w, v) = np.linalg.eigh(A)
    vsquare = np.array(v) ** 2
    expw = np.exp(w)
    xg = vsquare @ expw
    sc = dict(zip(nodelist, map(float, xg)))
    return sc

@not_implemented_for('directed')
@not_implemented_for('multigraph')
@nx._dispatch
def communicability_betweenness_centrality(G):
    if False:
        i = 10
        return i + 15
    'Returns subgraph communicability for all pairs of nodes in G.\n\n    Communicability betweenness measure makes use of the number of walks\n    connecting every pair of nodes as the basis of a betweenness centrality\n    measure.\n\n    Parameters\n    ----------\n    G: graph\n\n    Returns\n    -------\n    nodes : dictionary\n        Dictionary of nodes with communicability betweenness as the value.\n\n    Raises\n    ------\n    NetworkXError\n        If the graph is not undirected and simple.\n\n    Notes\n    -----\n    Let `G=(V,E)` be a simple undirected graph with `n` nodes and `m` edges,\n    and `A` denote the adjacency matrix of `G`.\n\n    Let `G(r)=(V,E(r))` be the graph resulting from\n    removing all edges connected to node `r` but not the node itself.\n\n    The adjacency matrix for `G(r)` is `A+E(r)`,  where `E(r)` has nonzeros\n    only in row and column `r`.\n\n    The subraph betweenness of a node `r`  is [1]_\n\n    .. math::\n\n         \\omega_{r} = \\frac{1}{C}\\sum_{p}\\sum_{q}\\frac{G_{prq}}{G_{pq}},\n         p\\neq q, q\\neq r,\n\n    where\n    `G_{prq}=(e^{A}_{pq} - (e^{A+E(r)})_{pq}`  is the number of walks\n    involving node r,\n    `G_{pq}=(e^{A})_{pq}` is the number of closed walks starting\n    at node `p` and ending at node `q`,\n    and `C=(n-1)^{2}-(n-1)` is a normalization factor equal to the\n    number of terms in the sum.\n\n    The resulting `\\omega_{r}` takes values between zero and one.\n    The lower bound cannot be attained for a connected\n    graph, and the upper bound is attained in the star graph.\n\n    References\n    ----------\n    .. [1] Ernesto Estrada, Desmond J. Higham, Naomichi Hatano,\n       "Communicability Betweenness in Complex Networks"\n       Physica A 388 (2009) 764-774.\n       https://arxiv.org/abs/0905.4102\n\n    Examples\n    --------\n    >>> G = nx.Graph([(0, 1), (1, 2), (1, 5), (5, 4), (2, 4), (2, 3), (4, 3), (3, 6)])\n    >>> cbc = nx.communicability_betweenness_centrality(G)\n    >>> print([f"{node} {cbc[node]:0.2f}" for node in sorted(cbc)])\n    [\'0 0.03\', \'1 0.45\', \'2 0.51\', \'3 0.45\', \'4 0.40\', \'5 0.19\', \'6 0.03\']\n    '
    import numpy as np
    import scipy as sp
    nodelist = list(G)
    n = len(nodelist)
    A = nx.to_numpy_array(G, nodelist)
    A[np.nonzero(A)] = 1
    expA = sp.linalg.expm(A)
    mapping = dict(zip(nodelist, range(n)))
    cbc = {}
    for v in G:
        i = mapping[v]
        row = A[i, :].copy()
        col = A[:, i].copy()
        A[i, :] = 0
        A[:, i] = 0
        B = (expA - sp.linalg.expm(A)) / expA
        B[i, :] = 0
        B[:, i] = 0
        B -= np.diag(np.diag(B))
        cbc[v] = B.sum()
        A[i, :] = row
        A[:, i] = col
    order = len(cbc)
    if order > 2:
        scale = 1.0 / ((order - 1.0) ** 2 - (order - 1.0))
        for v in cbc:
            cbc[v] *= scale
    return cbc

@nx._dispatch
def estrada_index(G):
    if False:
        i = 10
        return i + 15
    'Returns the Estrada index of a the graph G.\n\n    The Estrada Index is a topological index of folding or 3D "compactness" ([1]_).\n\n    Parameters\n    ----------\n    G: graph\n\n    Returns\n    -------\n    estrada index: float\n\n    Raises\n    ------\n    NetworkXError\n        If the graph is not undirected and simple.\n\n    Notes\n    -----\n    Let `G=(V,E)` be a simple undirected graph with `n` nodes  and let\n    `\\lambda_{1}\\leq\\lambda_{2}\\leq\\cdots\\lambda_{n}`\n    be a non-increasing ordering of the eigenvalues of its adjacency\n    matrix `A`. The Estrada index is ([1]_, [2]_)\n\n    .. math::\n        EE(G)=\\sum_{j=1}^n e^{\\lambda _j}.\n\n    References\n    ----------\n    .. [1] E. Estrada, "Characterization of 3D molecular structure",\n       Chem. Phys. Lett. 319, 713 (2000).\n       https://doi.org/10.1016/S0009-2614(00)00158-5\n    .. [2] José Antonio de la Peñaa, Ivan Gutman, Juan Rada,\n       "Estimating the Estrada index",\n       Linear Algebra and its Applications. 427, 1 (2007).\n       https://doi.org/10.1016/j.laa.2007.06.020\n\n    Examples\n    --------\n    >>> G = nx.Graph([(0, 1), (1, 2), (1, 5), (5, 4), (2, 4), (2, 3), (4, 3), (3, 6)])\n    >>> ei = nx.estrada_index(G)\n    >>> print(f"{ei:0.5}")\n    20.55\n    '
    return sum(subgraph_centrality(G).values())