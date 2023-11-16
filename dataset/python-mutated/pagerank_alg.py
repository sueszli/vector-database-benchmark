"""PageRank analysis of graph structure. """
from warnings import warn
import networkx as nx
__all__ = ['pagerank', 'google_matrix']

@nx._dispatch(edge_attrs='weight')
def pagerank(G, alpha=0.85, personalization=None, max_iter=100, tol=1e-06, nstart=None, weight='weight', dangling=None):
    if False:
        print('Hello World!')
    'Returns the PageRank of the nodes in the graph.\n\n    PageRank computes a ranking of the nodes in the graph G based on\n    the structure of the incoming links. It was originally designed as\n    an algorithm to rank web pages.\n\n    Parameters\n    ----------\n    G : graph\n      A NetworkX graph.  Undirected graphs will be converted to a directed\n      graph with two directed edges for each undirected edge.\n\n    alpha : float, optional\n      Damping parameter for PageRank, default=0.85.\n\n    personalization: dict, optional\n      The "personalization vector" consisting of a dictionary with a\n      key some subset of graph nodes and personalization value each of those.\n      At least one personalization value must be non-zero.\n      If not specified, a nodes personalization value will be zero.\n      By default, a uniform distribution is used.\n\n    max_iter : integer, optional\n      Maximum number of iterations in power method eigenvalue solver.\n\n    tol : float, optional\n      Error tolerance used to check convergence in power method solver.\n      The iteration will stop after a tolerance of ``len(G) * tol`` is reached.\n\n    nstart : dictionary, optional\n      Starting value of PageRank iteration for each node.\n\n    weight : key, optional\n      Edge data key to use as weight.  If None weights are set to 1.\n\n    dangling: dict, optional\n      The outedges to be assigned to any "dangling" nodes, i.e., nodes without\n      any outedges. The dict key is the node the outedge points to and the dict\n      value is the weight of that outedge. By default, dangling nodes are given\n      outedges according to the personalization vector (uniform if not\n      specified). This must be selected to result in an irreducible transition\n      matrix (see notes under google_matrix). It may be common to have the\n      dangling dict to be the same as the personalization dict.\n\n\n    Returns\n    -------\n    pagerank : dictionary\n       Dictionary of nodes with PageRank as value\n\n    Examples\n    --------\n    >>> G = nx.DiGraph(nx.path_graph(4))\n    >>> pr = nx.pagerank(G, alpha=0.9)\n\n    Notes\n    -----\n    The eigenvector calculation is done by the power iteration method\n    and has no guarantee of convergence.  The iteration will stop after\n    an error tolerance of ``len(G) * tol`` has been reached. If the\n    number of iterations exceed `max_iter`, a\n    :exc:`networkx.exception.PowerIterationFailedConvergence` exception\n    is raised.\n\n    The PageRank algorithm was designed for directed graphs but this\n    algorithm does not check if the input graph is directed and will\n    execute on undirected graphs by converting each edge in the\n    directed graph to two edges.\n\n    See Also\n    --------\n    google_matrix\n\n    Raises\n    ------\n    PowerIterationFailedConvergence\n        If the algorithm fails to converge to the specified tolerance\n        within the specified number of iterations of the power iteration\n        method.\n\n    References\n    ----------\n    .. [1] A. Langville and C. Meyer,\n       "A survey of eigenvector methods of web information retrieval."\n       http://citeseer.ist.psu.edu/713792.html\n    .. [2] Page, Lawrence; Brin, Sergey; Motwani, Rajeev and Winograd, Terry,\n       The PageRank citation ranking: Bringing order to the Web. 1999\n       http://dbpubs.stanford.edu:8090/pub/showDoc.Fulltext?lang=en&doc=1999-66&format=pdf\n\n    '
    return _pagerank_scipy(G, alpha, personalization, max_iter, tol, nstart, weight, dangling)

def _pagerank_python(G, alpha=0.85, personalization=None, max_iter=100, tol=1e-06, nstart=None, weight='weight', dangling=None):
    if False:
        print('Hello World!')
    if len(G) == 0:
        return {}
    D = G.to_directed()
    W = nx.stochastic_graph(D, weight=weight)
    N = W.number_of_nodes()
    if nstart is None:
        x = dict.fromkeys(W, 1.0 / N)
    else:
        s = sum(nstart.values())
        x = {k: v / s for (k, v) in nstart.items()}
    if personalization is None:
        p = dict.fromkeys(W, 1.0 / N)
    else:
        s = sum(personalization.values())
        p = {k: v / s for (k, v) in personalization.items()}
    if dangling is None:
        dangling_weights = p
    else:
        s = sum(dangling.values())
        dangling_weights = {k: v / s for (k, v) in dangling.items()}
    dangling_nodes = [n for n in W if W.out_degree(n, weight=weight) == 0.0]
    for _ in range(max_iter):
        xlast = x
        x = dict.fromkeys(xlast.keys(), 0)
        danglesum = alpha * sum((xlast[n] for n in dangling_nodes))
        for n in x:
            for (_, nbr, wt) in W.edges(n, data=weight):
                x[nbr] += alpha * xlast[n] * wt
            x[n] += danglesum * dangling_weights.get(n, 0) + (1.0 - alpha) * p.get(n, 0)
        err = sum((abs(x[n] - xlast[n]) for n in x))
        if err < N * tol:
            return x
    raise nx.PowerIterationFailedConvergence(max_iter)

@nx._dispatch(edge_attrs='weight')
def google_matrix(G, alpha=0.85, personalization=None, nodelist=None, weight='weight', dangling=None):
    if False:
        for i in range(10):
            print('nop')
    'Returns the Google matrix of the graph.\n\n    Parameters\n    ----------\n    G : graph\n      A NetworkX graph.  Undirected graphs will be converted to a directed\n      graph with two directed edges for each undirected edge.\n\n    alpha : float\n      The damping factor.\n\n    personalization: dict, optional\n      The "personalization vector" consisting of a dictionary with a\n      key some subset of graph nodes and personalization value each of those.\n      At least one personalization value must be non-zero.\n      If not specified, a nodes personalization value will be zero.\n      By default, a uniform distribution is used.\n\n    nodelist : list, optional\n      The rows and columns are ordered according to the nodes in nodelist.\n      If nodelist is None, then the ordering is produced by G.nodes().\n\n    weight : key, optional\n      Edge data key to use as weight.  If None weights are set to 1.\n\n    dangling: dict, optional\n      The outedges to be assigned to any "dangling" nodes, i.e., nodes without\n      any outedges. The dict key is the node the outedge points to and the dict\n      value is the weight of that outedge. By default, dangling nodes are given\n      outedges according to the personalization vector (uniform if not\n      specified) This must be selected to result in an irreducible transition\n      matrix (see notes below). It may be common to have the dangling dict to\n      be the same as the personalization dict.\n\n    Returns\n    -------\n    A : 2D NumPy ndarray\n       Google matrix of the graph\n\n    Notes\n    -----\n    The array returned represents the transition matrix that describes the\n    Markov chain used in PageRank. For PageRank to converge to a unique\n    solution (i.e., a unique stationary distribution in a Markov chain), the\n    transition matrix must be irreducible. In other words, it must be that\n    there exists a path between every pair of nodes in the graph, or else there\n    is the potential of "rank sinks."\n\n    This implementation works with Multi(Di)Graphs. For multigraphs the\n    weight between two nodes is set to be the sum of all edge weights\n    between those nodes.\n\n    See Also\n    --------\n    pagerank\n    '
    import numpy as np
    if nodelist is None:
        nodelist = list(G)
    A = nx.to_numpy_array(G, nodelist=nodelist, weight=weight)
    N = len(G)
    if N == 0:
        return A
    if personalization is None:
        p = np.repeat(1.0 / N, N)
    else:
        p = np.array([personalization.get(n, 0) for n in nodelist], dtype=float)
        if p.sum() == 0:
            raise ZeroDivisionError
        p /= p.sum()
    if dangling is None:
        dangling_weights = p
    else:
        dangling_weights = np.array([dangling.get(n, 0) for n in nodelist], dtype=float)
        dangling_weights /= dangling_weights.sum()
    dangling_nodes = np.where(A.sum(axis=1) == 0)[0]
    A[dangling_nodes] = dangling_weights
    A /= A.sum(axis=1)[:, np.newaxis]
    return alpha * A + (1 - alpha) * p

def _pagerank_numpy(G, alpha=0.85, personalization=None, weight='weight', dangling=None):
    if False:
        for i in range(10):
            print('nop')
    'Returns the PageRank of the nodes in the graph.\n\n    PageRank computes a ranking of the nodes in the graph G based on\n    the structure of the incoming links. It was originally designed as\n    an algorithm to rank web pages.\n\n    Parameters\n    ----------\n    G : graph\n      A NetworkX graph.  Undirected graphs will be converted to a directed\n      graph with two directed edges for each undirected edge.\n\n    alpha : float, optional\n      Damping parameter for PageRank, default=0.85.\n\n    personalization: dict, optional\n      The "personalization vector" consisting of a dictionary with a\n      key some subset of graph nodes and personalization value each of those.\n      At least one personalization value must be non-zero.\n      If not specified, a nodes personalization value will be zero.\n      By default, a uniform distribution is used.\n\n    weight : key, optional\n      Edge data key to use as weight.  If None weights are set to 1.\n\n    dangling: dict, optional\n      The outedges to be assigned to any "dangling" nodes, i.e., nodes without\n      any outedges. The dict key is the node the outedge points to and the dict\n      value is the weight of that outedge. By default, dangling nodes are given\n      outedges according to the personalization vector (uniform if not\n      specified) This must be selected to result in an irreducible transition\n      matrix (see notes under google_matrix). It may be common to have the\n      dangling dict to be the same as the personalization dict.\n\n    Returns\n    -------\n    pagerank : dictionary\n       Dictionary of nodes with PageRank as value.\n\n    Examples\n    --------\n    >>> from networkx.algorithms.link_analysis.pagerank_alg import _pagerank_numpy\n    >>> G = nx.DiGraph(nx.path_graph(4))\n    >>> pr = _pagerank_numpy(G, alpha=0.9)\n\n    Notes\n    -----\n    The eigenvector calculation uses NumPy\'s interface to the LAPACK\n    eigenvalue solvers.  This will be the fastest and most accurate\n    for small graphs.\n\n    This implementation works with Multi(Di)Graphs. For multigraphs the\n    weight between two nodes is set to be the sum of all edge weights\n    between those nodes.\n\n    See Also\n    --------\n    pagerank, google_matrix\n\n    References\n    ----------\n    .. [1] A. Langville and C. Meyer,\n       "A survey of eigenvector methods of web information retrieval."\n       http://citeseer.ist.psu.edu/713792.html\n    .. [2] Page, Lawrence; Brin, Sergey; Motwani, Rajeev and Winograd, Terry,\n       The PageRank citation ranking: Bringing order to the Web. 1999\n       http://dbpubs.stanford.edu:8090/pub/showDoc.Fulltext?lang=en&doc=1999-66&format=pdf\n    '
    import numpy as np
    if len(G) == 0:
        return {}
    M = google_matrix(G, alpha, personalization=personalization, weight=weight, dangling=dangling)
    (eigenvalues, eigenvectors) = np.linalg.eig(M.T)
    ind = np.argmax(eigenvalues)
    largest = np.array(eigenvectors[:, ind]).flatten().real
    norm = largest.sum()
    return dict(zip(G, map(float, largest / norm)))

def _pagerank_scipy(G, alpha=0.85, personalization=None, max_iter=100, tol=1e-06, nstart=None, weight='weight', dangling=None):
    if False:
        print('Hello World!')
    'Returns the PageRank of the nodes in the graph.\n\n    PageRank computes a ranking of the nodes in the graph G based on\n    the structure of the incoming links. It was originally designed as\n    an algorithm to rank web pages.\n\n    Parameters\n    ----------\n    G : graph\n      A NetworkX graph.  Undirected graphs will be converted to a directed\n      graph with two directed edges for each undirected edge.\n\n    alpha : float, optional\n      Damping parameter for PageRank, default=0.85.\n\n    personalization: dict, optional\n      The "personalization vector" consisting of a dictionary with a\n      key some subset of graph nodes and personalization value each of those.\n      At least one personalization value must be non-zero.\n      If not specified, a nodes personalization value will be zero.\n      By default, a uniform distribution is used.\n\n    max_iter : integer, optional\n      Maximum number of iterations in power method eigenvalue solver.\n\n    tol : float, optional\n      Error tolerance used to check convergence in power method solver.\n      The iteration will stop after a tolerance of ``len(G) * tol`` is reached.\n\n    nstart : dictionary, optional\n      Starting value of PageRank iteration for each node.\n\n    weight : key, optional\n      Edge data key to use as weight.  If None weights are set to 1.\n\n    dangling: dict, optional\n      The outedges to be assigned to any "dangling" nodes, i.e., nodes without\n      any outedges. The dict key is the node the outedge points to and the dict\n      value is the weight of that outedge. By default, dangling nodes are given\n      outedges according to the personalization vector (uniform if not\n      specified) This must be selected to result in an irreducible transition\n      matrix (see notes under google_matrix). It may be common to have the\n      dangling dict to be the same as the personalization dict.\n\n    Returns\n    -------\n    pagerank : dictionary\n       Dictionary of nodes with PageRank as value\n\n    Examples\n    --------\n    >>> from networkx.algorithms.link_analysis.pagerank_alg import _pagerank_scipy\n    >>> G = nx.DiGraph(nx.path_graph(4))\n    >>> pr = _pagerank_scipy(G, alpha=0.9)\n\n    Notes\n    -----\n    The eigenvector calculation uses power iteration with a SciPy\n    sparse matrix representation.\n\n    This implementation works with Multi(Di)Graphs. For multigraphs the\n    weight between two nodes is set to be the sum of all edge weights\n    between those nodes.\n\n    See Also\n    --------\n    pagerank\n\n    Raises\n    ------\n    PowerIterationFailedConvergence\n        If the algorithm fails to converge to the specified tolerance\n        within the specified number of iterations of the power iteration\n        method.\n\n    References\n    ----------\n    .. [1] A. Langville and C. Meyer,\n       "A survey of eigenvector methods of web information retrieval."\n       http://citeseer.ist.psu.edu/713792.html\n    .. [2] Page, Lawrence; Brin, Sergey; Motwani, Rajeev and Winograd, Terry,\n       The PageRank citation ranking: Bringing order to the Web. 1999\n       http://dbpubs.stanford.edu:8090/pub/showDoc.Fulltext?lang=en&doc=1999-66&format=pdf\n    '
    import numpy as np
    import scipy as sp
    N = len(G)
    if N == 0:
        return {}
    nodelist = list(G)
    A = nx.to_scipy_sparse_array(G, nodelist=nodelist, weight=weight, dtype=float)
    S = A.sum(axis=1)
    S[S != 0] = 1.0 / S[S != 0]
    Q = sp.sparse.csr_array(sp.sparse.spdiags(S.T, 0, *A.shape))
    A = Q @ A
    if nstart is None:
        x = np.repeat(1.0 / N, N)
    else:
        x = np.array([nstart.get(n, 0) for n in nodelist], dtype=float)
        x /= x.sum()
    if personalization is None:
        p = np.repeat(1.0 / N, N)
    else:
        p = np.array([personalization.get(n, 0) for n in nodelist], dtype=float)
        if p.sum() == 0:
            raise ZeroDivisionError
        p /= p.sum()
    if dangling is None:
        dangling_weights = p
    else:
        dangling_weights = np.array([dangling.get(n, 0) for n in nodelist], dtype=float)
        dangling_weights /= dangling_weights.sum()
    is_dangling = np.where(S == 0)[0]
    for _ in range(max_iter):
        xlast = x
        x = alpha * (x @ A + sum(x[is_dangling]) * dangling_weights) + (1 - alpha) * p
        err = np.absolute(x - xlast).sum()
        if err < N * tol:
            return dict(zip(nodelist, map(float, x)))
    raise nx.PowerIterationFailedConvergence(max_iter)