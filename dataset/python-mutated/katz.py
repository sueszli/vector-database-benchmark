"""Katz centrality."""
import math
import networkx as nx
from networkx.utils import not_implemented_for
__all__ = ['katz_centrality', 'katz_centrality_numpy']

@not_implemented_for('multigraph')
@nx._dispatch(edge_attrs='weight')
def katz_centrality(G, alpha=0.1, beta=1.0, max_iter=1000, tol=1e-06, nstart=None, normalized=True, weight=None):
    if False:
        print('Hello World!')
    'Compute the Katz centrality for the nodes of the graph G.\n\n    Katz centrality computes the centrality for a node based on the centrality\n    of its neighbors. It is a generalization of the eigenvector centrality. The\n    Katz centrality for node $i$ is\n\n    .. math::\n\n        x_i = \\alpha \\sum_{j} A_{ij} x_j + \\beta,\n\n    where $A$ is the adjacency matrix of graph G with eigenvalues $\\lambda$.\n\n    The parameter $\\beta$ controls the initial centrality and\n\n    .. math::\n\n        \\alpha < \\frac{1}{\\lambda_{\\max}}.\n\n    Katz centrality computes the relative influence of a node within a\n    network by measuring the number of the immediate neighbors (first\n    degree nodes) and also all other nodes in the network that connect\n    to the node under consideration through these immediate neighbors.\n\n    Extra weight can be provided to immediate neighbors through the\n    parameter $\\beta$.  Connections made with distant neighbors\n    are, however, penalized by an attenuation factor $\\alpha$ which\n    should be strictly less than the inverse largest eigenvalue of the\n    adjacency matrix in order for the Katz centrality to be computed\n    correctly. More information is provided in [1]_.\n\n    Parameters\n    ----------\n    G : graph\n      A NetworkX graph.\n\n    alpha : float, optional (default=0.1)\n      Attenuation factor\n\n    beta : scalar or dictionary, optional (default=1.0)\n      Weight attributed to the immediate neighborhood. If not a scalar, the\n      dictionary must have a value for every node.\n\n    max_iter : integer, optional (default=1000)\n      Maximum number of iterations in power method.\n\n    tol : float, optional (default=1.0e-6)\n      Error tolerance used to check convergence in power method iteration.\n\n    nstart : dictionary, optional\n      Starting value of Katz iteration for each node.\n\n    normalized : bool, optional (default=True)\n      If True normalize the resulting values.\n\n    weight : None or string, optional (default=None)\n      If None, all edge weights are considered equal.\n      Otherwise holds the name of the edge attribute used as weight.\n      In this measure the weight is interpreted as the connection strength.\n\n    Returns\n    -------\n    nodes : dictionary\n       Dictionary of nodes with Katz centrality as the value.\n\n    Raises\n    ------\n    NetworkXError\n       If the parameter `beta` is not a scalar but lacks a value for at least\n       one node\n\n    PowerIterationFailedConvergence\n        If the algorithm fails to converge to the specified tolerance\n        within the specified number of iterations of the power iteration\n        method.\n\n    Examples\n    --------\n    >>> import math\n    >>> G = nx.path_graph(4)\n    >>> phi = (1 + math.sqrt(5)) / 2.0  # largest eigenvalue of adj matrix\n    >>> centrality = nx.katz_centrality(G, 1 / phi - 0.01)\n    >>> for n, c in sorted(centrality.items()):\n    ...     print(f"{n} {c:.2f}")\n    0 0.37\n    1 0.60\n    2 0.60\n    3 0.37\n\n    See Also\n    --------\n    katz_centrality_numpy\n    eigenvector_centrality\n    eigenvector_centrality_numpy\n    :func:`~networkx.algorithms.link_analysis.pagerank_alg.pagerank`\n    :func:`~networkx.algorithms.link_analysis.hits_alg.hits`\n\n    Notes\n    -----\n    Katz centrality was introduced by [2]_.\n\n    This algorithm it uses the power method to find the eigenvector\n    corresponding to the largest eigenvalue of the adjacency matrix of ``G``.\n    The parameter ``alpha`` should be strictly less than the inverse of largest\n    eigenvalue of the adjacency matrix for the algorithm to converge.\n    You can use ``max(nx.adjacency_spectrum(G))`` to get $\\lambda_{\\max}$ the largest\n    eigenvalue of the adjacency matrix.\n    The iteration will stop after ``max_iter`` iterations or an error tolerance of\n    ``number_of_nodes(G) * tol`` has been reached.\n\n    For strongly connected graphs, as $\\alpha \\to 1/\\lambda_{\\max}$, and $\\beta > 0$,\n    Katz centrality approaches the results for eigenvector centrality.\n\n    For directed graphs this finds "left" eigenvectors which corresponds\n    to the in-edges in the graph. For out-edges Katz centrality,\n    first reverse the graph with ``G.reverse()``.\n\n    References\n    ----------\n    .. [1] Mark E. J. Newman:\n       Networks: An Introduction.\n       Oxford University Press, USA, 2010, p. 720.\n    .. [2] Leo Katz:\n       A New Status Index Derived from Sociometric Index.\n       Psychometrika 18(1):39–43, 1953\n       https://link.springer.com/content/pdf/10.1007/BF02289026.pdf\n    '
    if len(G) == 0:
        return {}
    nnodes = G.number_of_nodes()
    if nstart is None:
        x = {n: 0 for n in G}
    else:
        x = nstart
    try:
        b = dict.fromkeys(G, float(beta))
    except (TypeError, ValueError, AttributeError) as err:
        b = beta
        if set(beta) != set(G):
            raise nx.NetworkXError('beta dictionary must have a value for every node') from err
    for _ in range(max_iter):
        xlast = x
        x = dict.fromkeys(xlast, 0)
        for n in x:
            for nbr in G[n]:
                x[nbr] += xlast[n] * G[n][nbr].get(weight, 1)
        for n in x:
            x[n] = alpha * x[n] + b[n]
        error = sum((abs(x[n] - xlast[n]) for n in x))
        if error < nnodes * tol:
            if normalized:
                try:
                    s = 1.0 / math.hypot(*x.values())
                except ZeroDivisionError:
                    s = 1.0
            else:
                s = 1
            for n in x:
                x[n] *= s
            return x
    raise nx.PowerIterationFailedConvergence(max_iter)

@not_implemented_for('multigraph')
@nx._dispatch(edge_attrs='weight')
def katz_centrality_numpy(G, alpha=0.1, beta=1.0, normalized=True, weight=None):
    if False:
        i = 10
        return i + 15
    'Compute the Katz centrality for the graph G.\n\n    Katz centrality computes the centrality for a node based on the centrality\n    of its neighbors. It is a generalization of the eigenvector centrality. The\n    Katz centrality for node $i$ is\n\n    .. math::\n\n        x_i = \\alpha \\sum_{j} A_{ij} x_j + \\beta,\n\n    where $A$ is the adjacency matrix of graph G with eigenvalues $\\lambda$.\n\n    The parameter $\\beta$ controls the initial centrality and\n\n    .. math::\n\n        \\alpha < \\frac{1}{\\lambda_{\\max}}.\n\n    Katz centrality computes the relative influence of a node within a\n    network by measuring the number of the immediate neighbors (first\n    degree nodes) and also all other nodes in the network that connect\n    to the node under consideration through these immediate neighbors.\n\n    Extra weight can be provided to immediate neighbors through the\n    parameter $\\beta$.  Connections made with distant neighbors\n    are, however, penalized by an attenuation factor $\\alpha$ which\n    should be strictly less than the inverse largest eigenvalue of the\n    adjacency matrix in order for the Katz centrality to be computed\n    correctly. More information is provided in [1]_.\n\n    Parameters\n    ----------\n    G : graph\n      A NetworkX graph\n\n    alpha : float\n      Attenuation factor\n\n    beta : scalar or dictionary, optional (default=1.0)\n      Weight attributed to the immediate neighborhood. If not a scalar the\n      dictionary must have an value for every node.\n\n    normalized : bool\n      If True normalize the resulting values.\n\n    weight : None or string, optional\n      If None, all edge weights are considered equal.\n      Otherwise holds the name of the edge attribute used as weight.\n      In this measure the weight is interpreted as the connection strength.\n\n    Returns\n    -------\n    nodes : dictionary\n       Dictionary of nodes with Katz centrality as the value.\n\n    Raises\n    ------\n    NetworkXError\n       If the parameter `beta` is not a scalar but lacks a value for at least\n       one node\n\n    Examples\n    --------\n    >>> import math\n    >>> G = nx.path_graph(4)\n    >>> phi = (1 + math.sqrt(5)) / 2.0  # largest eigenvalue of adj matrix\n    >>> centrality = nx.katz_centrality_numpy(G, 1 / phi)\n    >>> for n, c in sorted(centrality.items()):\n    ...     print(f"{n} {c:.2f}")\n    0 0.37\n    1 0.60\n    2 0.60\n    3 0.37\n\n    See Also\n    --------\n    katz_centrality\n    eigenvector_centrality_numpy\n    eigenvector_centrality\n    :func:`~networkx.algorithms.link_analysis.pagerank_alg.pagerank`\n    :func:`~networkx.algorithms.link_analysis.hits_alg.hits`\n\n    Notes\n    -----\n    Katz centrality was introduced by [2]_.\n\n    This algorithm uses a direct linear solver to solve the above equation.\n    The parameter ``alpha`` should be strictly less than the inverse of largest\n    eigenvalue of the adjacency matrix for there to be a solution.\n    You can use ``max(nx.adjacency_spectrum(G))`` to get $\\lambda_{\\max}$ the largest\n    eigenvalue of the adjacency matrix.\n\n    For strongly connected graphs, as $\\alpha \\to 1/\\lambda_{\\max}$, and $\\beta > 0$,\n    Katz centrality approaches the results for eigenvector centrality.\n\n    For directed graphs this finds "left" eigenvectors which corresponds\n    to the in-edges in the graph. For out-edges Katz centrality,\n    first reverse the graph with ``G.reverse()``.\n\n    References\n    ----------\n    .. [1] Mark E. J. Newman:\n       Networks: An Introduction.\n       Oxford University Press, USA, 2010, p. 173.\n    .. [2] Leo Katz:\n       A New Status Index Derived from Sociometric Index.\n       Psychometrika 18(1):39–43, 1953\n       https://link.springer.com/content/pdf/10.1007/BF02289026.pdf\n    '
    import numpy as np
    if len(G) == 0:
        return {}
    try:
        nodelist = beta.keys()
        if set(nodelist) != set(G):
            raise nx.NetworkXError('beta dictionary must have a value for every node')
        b = np.array(list(beta.values()), dtype=float)
    except AttributeError:
        nodelist = list(G)
        try:
            b = np.ones((len(nodelist), 1)) * beta
        except (TypeError, ValueError, AttributeError) as err:
            raise nx.NetworkXError('beta must be a number') from err
    A = nx.adjacency_matrix(G, nodelist=nodelist, weight=weight).todense().T
    n = A.shape[0]
    centrality = np.linalg.solve(np.eye(n, n) - alpha * A, b).squeeze()
    norm = np.sign(sum(centrality)) * np.linalg.norm(centrality) if normalized else 1
    return dict(zip(nodelist, centrality / norm))