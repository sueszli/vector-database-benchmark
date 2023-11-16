"""Hubs and authorities analysis of graph structure.
"""
import networkx as nx
__all__ = ['hits']

@nx._dispatch(preserve_edge_attrs={'G': {'weight': 1}})
def hits(G, max_iter=100, tol=1e-08, nstart=None, normalized=True):
    if False:
        print('Hello World!')
    'Returns HITS hubs and authorities values for nodes.\n\n    The HITS algorithm computes two numbers for a node.\n    Authorities estimates the node value based on the incoming links.\n    Hubs estimates the node value based on outgoing links.\n\n    Parameters\n    ----------\n    G : graph\n      A NetworkX graph\n\n    max_iter : integer, optional\n      Maximum number of iterations in power method.\n\n    tol : float, optional\n      Error tolerance used to check convergence in power method iteration.\n\n    nstart : dictionary, optional\n      Starting value of each node for power method iteration.\n\n    normalized : bool (default=True)\n       Normalize results by the sum of all of the values.\n\n    Returns\n    -------\n    (hubs,authorities) : two-tuple of dictionaries\n       Two dictionaries keyed by node containing the hub and authority\n       values.\n\n    Raises\n    ------\n    PowerIterationFailedConvergence\n        If the algorithm fails to converge to the specified tolerance\n        within the specified number of iterations of the power iteration\n        method.\n\n    Examples\n    --------\n    >>> G = nx.path_graph(4)\n    >>> h, a = nx.hits(G)\n\n    Notes\n    -----\n    The eigenvector calculation is done by the power iteration method\n    and has no guarantee of convergence.  The iteration will stop\n    after max_iter iterations or an error tolerance of\n    number_of_nodes(G)*tol has been reached.\n\n    The HITS algorithm was designed for directed graphs but this\n    algorithm does not check if the input graph is directed and will\n    execute on undirected graphs.\n\n    References\n    ----------\n    .. [1] A. Langville and C. Meyer,\n       "A survey of eigenvector methods of web information retrieval."\n       http://citeseer.ist.psu.edu/713792.html\n    .. [2] Jon Kleinberg,\n       Authoritative sources in a hyperlinked environment\n       Journal of the ACM 46 (5): 604-32, 1999.\n       doi:10.1145/324133.324140.\n       http://www.cs.cornell.edu/home/kleinber/auth.pdf.\n    '
    import numpy as np
    import scipy as sp
    if len(G) == 0:
        return ({}, {})
    A = nx.adjacency_matrix(G, nodelist=list(G), dtype=float)
    if nstart is not None:
        nstart = np.array(list(nstart.values()))
    if max_iter <= 0:
        raise nx.PowerIterationFailedConvergence(max_iter)
    try:
        (_, _, vt) = sp.sparse.linalg.svds(A, k=1, v0=nstart, maxiter=max_iter, tol=tol)
    except sp.sparse.linalg.ArpackNoConvergence as exc:
        raise nx.PowerIterationFailedConvergence(max_iter) from exc
    a = vt.flatten().real
    h = A @ a
    if normalized:
        h /= h.sum()
        a /= a.sum()
    hubs = dict(zip(G, map(float, h)))
    authorities = dict(zip(G, map(float, a)))
    return (hubs, authorities)

def _hits_python(G, max_iter=100, tol=1e-08, nstart=None, normalized=True):
    if False:
        i = 10
        return i + 15
    if isinstance(G, nx.MultiGraph | nx.MultiDiGraph):
        raise Exception('hits() not defined for graphs with multiedges.')
    if len(G) == 0:
        return ({}, {})
    if nstart is None:
        h = dict.fromkeys(G, 1.0 / G.number_of_nodes())
    else:
        h = nstart
        s = 1.0 / sum(h.values())
        for k in h:
            h[k] *= s
    for _ in range(max_iter):
        hlast = h
        h = dict.fromkeys(hlast.keys(), 0)
        a = dict.fromkeys(hlast.keys(), 0)
        for n in h:
            for nbr in G[n]:
                a[nbr] += hlast[n] * G[n][nbr].get('weight', 1)
        for n in h:
            for nbr in G[n]:
                h[n] += a[nbr] * G[n][nbr].get('weight', 1)
        s = 1.0 / max(h.values())
        for n in h:
            h[n] *= s
        s = 1.0 / max(a.values())
        for n in a:
            a[n] *= s
        err = sum((abs(h[n] - hlast[n]) for n in h))
        if err < tol:
            break
    else:
        raise nx.PowerIterationFailedConvergence(max_iter)
    if normalized:
        s = 1.0 / sum(a.values())
        for n in a:
            a[n] *= s
        s = 1.0 / sum(h.values())
        for n in h:
            h[n] *= s
    return (h, a)

def _hits_numpy(G, normalized=True):
    if False:
        for i in range(10):
            print('nop')
    'Returns HITS hubs and authorities values for nodes.\n\n    The HITS algorithm computes two numbers for a node.\n    Authorities estimates the node value based on the incoming links.\n    Hubs estimates the node value based on outgoing links.\n\n    Parameters\n    ----------\n    G : graph\n      A NetworkX graph\n\n    normalized : bool (default=True)\n       Normalize results by the sum of all of the values.\n\n    Returns\n    -------\n    (hubs,authorities) : two-tuple of dictionaries\n       Two dictionaries keyed by node containing the hub and authority\n       values.\n\n    Examples\n    --------\n    >>> G = nx.path_graph(4)\n\n    The `hubs` and `authorities` are given by the eigenvectors corresponding to the\n    maximum eigenvalues of the hubs_matrix and the authority_matrix, respectively.\n\n    The ``hubs`` and ``authority`` matrices are computed from the adjacency\n    matrix:\n\n    >>> adj_ary = nx.to_numpy_array(G)\n    >>> hubs_matrix = adj_ary @ adj_ary.T\n    >>> authority_matrix = adj_ary.T @ adj_ary\n\n    `_hits_numpy` maps the eigenvector corresponding to the maximum eigenvalue\n    of the respective matrices to the nodes in `G`:\n\n    >>> from networkx.algorithms.link_analysis.hits_alg import _hits_numpy\n    >>> hubs, authority = _hits_numpy(G)\n\n    Notes\n    -----\n    The eigenvector calculation uses NumPy\'s interface to LAPACK.\n\n    The HITS algorithm was designed for directed graphs but this\n    algorithm does not check if the input graph is directed and will\n    execute on undirected graphs.\n\n    References\n    ----------\n    .. [1] A. Langville and C. Meyer,\n       "A survey of eigenvector methods of web information retrieval."\n       http://citeseer.ist.psu.edu/713792.html\n    .. [2] Jon Kleinberg,\n       Authoritative sources in a hyperlinked environment\n       Journal of the ACM 46 (5): 604-32, 1999.\n       doi:10.1145/324133.324140.\n       http://www.cs.cornell.edu/home/kleinber/auth.pdf.\n    '
    import numpy as np
    if len(G) == 0:
        return ({}, {})
    adj_ary = nx.to_numpy_array(G)
    H = adj_ary @ adj_ary.T
    (e, ev) = np.linalg.eig(H)
    h = ev[:, np.argmax(e)]
    A = adj_ary.T @ adj_ary
    (e, ev) = np.linalg.eig(A)
    a = ev[:, np.argmax(e)]
    if normalized:
        h /= h.sum()
        a /= a.sum()
    else:
        h /= h.max()
        a /= a.max()
    hubs = dict(zip(G, map(float, h)))
    authorities = dict(zip(G, map(float, a)))
    return (hubs, authorities)

def _hits_scipy(G, max_iter=100, tol=1e-06, nstart=None, normalized=True):
    if False:
        while True:
            i = 10
    'Returns HITS hubs and authorities values for nodes.\n\n\n    The HITS algorithm computes two numbers for a node.\n    Authorities estimates the node value based on the incoming links.\n    Hubs estimates the node value based on outgoing links.\n\n    Parameters\n    ----------\n    G : graph\n      A NetworkX graph\n\n    max_iter : integer, optional\n      Maximum number of iterations in power method.\n\n    tol : float, optional\n      Error tolerance used to check convergence in power method iteration.\n\n    nstart : dictionary, optional\n      Starting value of each node for power method iteration.\n\n    normalized : bool (default=True)\n       Normalize results by the sum of all of the values.\n\n    Returns\n    -------\n    (hubs,authorities) : two-tuple of dictionaries\n       Two dictionaries keyed by node containing the hub and authority\n       values.\n\n    Examples\n    --------\n    >>> from networkx.algorithms.link_analysis.hits_alg import _hits_scipy\n    >>> G = nx.path_graph(4)\n    >>> h, a = _hits_scipy(G)\n\n    Notes\n    -----\n    This implementation uses SciPy sparse matrices.\n\n    The eigenvector calculation is done by the power iteration method\n    and has no guarantee of convergence.  The iteration will stop\n    after max_iter iterations or an error tolerance of\n    number_of_nodes(G)*tol has been reached.\n\n    The HITS algorithm was designed for directed graphs but this\n    algorithm does not check if the input graph is directed and will\n    execute on undirected graphs.\n\n    Raises\n    ------\n    PowerIterationFailedConvergence\n        If the algorithm fails to converge to the specified tolerance\n        within the specified number of iterations of the power iteration\n        method.\n\n    References\n    ----------\n    .. [1] A. Langville and C. Meyer,\n       "A survey of eigenvector methods of web information retrieval."\n       http://citeseer.ist.psu.edu/713792.html\n    .. [2] Jon Kleinberg,\n       Authoritative sources in a hyperlinked environment\n       Journal of the ACM 46 (5): 604-632, 1999.\n       doi:10.1145/324133.324140.\n       http://www.cs.cornell.edu/home/kleinber/auth.pdf.\n    '
    import numpy as np
    if len(G) == 0:
        return ({}, {})
    A = nx.to_scipy_sparse_array(G, nodelist=list(G))
    (n, _) = A.shape
    ATA = A.T @ A
    if nstart is None:
        x = np.ones((n, 1)) / n
    else:
        x = np.array([nstart.get(n, 0) for n in list(G)], dtype=float)
        x /= x.sum()
    i = 0
    while True:
        xlast = x
        x = ATA @ x
        x /= x.max()
        err = np.absolute(x - xlast).sum()
        if err < tol:
            break
        if i > max_iter:
            raise nx.PowerIterationFailedConvergence(max_iter)
        i += 1
    a = x.flatten()
    h = A @ a
    if normalized:
        h /= h.sum()
        a /= a.sum()
    hubs = dict(zip(G, map(float, h)))
    authorities = dict(zip(G, map(float, a)))
    return (hubs, authorities)