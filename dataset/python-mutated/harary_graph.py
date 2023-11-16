"""Generators for Harary graphs

This module gives two generators for the Harary graph, which was
introduced by the famous mathematician Frank Harary in his 1962 work [H]_.
The first generator gives the Harary graph that maximizes the node
connectivity with given number of nodes and given number of edges.
The second generator gives the Harary graph that minimizes
the number of edges in the graph with given node connectivity and
number of nodes.

References
----------
.. [H] Harary, F. "The Maximum Connectivity of a Graph."
       Proc. Nat. Acad. Sci. USA 48, 1142-1146, 1962.

"""
import networkx as nx
from networkx.exception import NetworkXError
__all__ = ['hnm_harary_graph', 'hkn_harary_graph']

@nx._dispatch(graphs=None)
def hnm_harary_graph(n, m, create_using=None):
    if False:
        i = 10
        return i + 15
    'Returns the Harary graph with given numbers of nodes and edges.\n\n    The Harary graph $H_{n,m}$ is the graph that maximizes node connectivity\n    with $n$ nodes and $m$ edges.\n\n    This maximum node connectivity is known to be floor($2m/n$). [1]_\n\n    Parameters\n    ----------\n    n: integer\n       The number of nodes the generated graph is to contain\n\n    m: integer\n       The number of edges the generated graph is to contain\n\n    create_using : NetworkX graph constructor, optional Graph type\n     to create (default=nx.Graph). If graph instance, then cleared\n     before populated.\n\n    Returns\n    -------\n    NetworkX graph\n        The Harary graph $H_{n,m}$.\n\n    See Also\n    --------\n    hkn_harary_graph\n\n    Notes\n    -----\n    This algorithm runs in $O(m)$ time.\n    It is implemented by following the Reference [2]_.\n\n    References\n    ----------\n    .. [1] F. T. Boesch, A. Satyanarayana, and C. L. Suffel,\n       "A Survey of Some Network Reliability Analysis and Synthesis Results,"\n       Networks, pp. 99-107, 2009.\n\n    .. [2] Harary, F. "The Maximum Connectivity of a Graph."\n       Proc. Nat. Acad. Sci. USA 48, 1142-1146, 1962.\n    '
    if n < 1:
        raise NetworkXError('The number of nodes must be >= 1!')
    if m < n - 1:
        raise NetworkXError('The number of edges must be >= n - 1 !')
    if m > n * (n - 1) // 2:
        raise NetworkXError('The number of edges must be <= n(n-1)/2')
    H = nx.empty_graph(n, create_using)
    d = 2 * m // n
    if n % 2 == 0 or d % 2 == 0:
        offset = d // 2
        for i in range(n):
            for j in range(1, offset + 1):
                H.add_edge(i, (i - j) % n)
                H.add_edge(i, (i + j) % n)
        if d & 1:
            half = n // 2
            for i in range(half):
                H.add_edge(i, i + half)
        r = 2 * m % n
        if r > 0:
            for i in range(r // 2):
                H.add_edge(i, i + offset + 1)
    else:
        offset = (d - 1) // 2
        for i in range(n):
            for j in range(1, offset + 1):
                H.add_edge(i, (i - j) % n)
                H.add_edge(i, (i + j) % n)
        half = n // 2
        for i in range(m - n * offset):
            H.add_edge(i, (i + half) % n)
    return H

@nx._dispatch(graphs=None)
def hkn_harary_graph(k, n, create_using=None):
    if False:
        i = 10
        return i + 15
    'Returns the Harary graph with given node connectivity and node number.\n\n    The Harary graph $H_{k,n}$ is the graph that minimizes the number of\n    edges needed with given node connectivity $k$ and node number $n$.\n\n    This smallest number of edges is known to be ceil($kn/2$) [1]_.\n\n    Parameters\n    ----------\n    k: integer\n       The node connectivity of the generated graph\n\n    n: integer\n       The number of nodes the generated graph is to contain\n\n    create_using : NetworkX graph constructor, optional Graph type\n     to create (default=nx.Graph). If graph instance, then cleared\n     before populated.\n\n    Returns\n    -------\n    NetworkX graph\n        The Harary graph $H_{k,n}$.\n\n    See Also\n    --------\n    hnm_harary_graph\n\n    Notes\n    -----\n    This algorithm runs in $O(kn)$ time.\n    It is implemented by following the Reference [2]_.\n\n    References\n    ----------\n    .. [1] Weisstein, Eric W. "Harary Graph." From MathWorld--A Wolfram Web\n     Resource. http://mathworld.wolfram.com/HararyGraph.html.\n\n    .. [2] Harary, F. "The Maximum Connectivity of a Graph."\n      Proc. Nat. Acad. Sci. USA 48, 1142-1146, 1962.\n    '
    if k < 1:
        raise NetworkXError('The node connectivity must be >= 1!')
    if n < k + 1:
        raise NetworkXError('The number of nodes must be >= k+1 !')
    if k == 1:
        H = nx.path_graph(n, create_using)
        return H
    H = nx.empty_graph(n, create_using)
    if k % 2 == 0 or n % 2 == 0:
        offset = k // 2
        for i in range(n):
            for j in range(1, offset + 1):
                H.add_edge(i, (i - j) % n)
                H.add_edge(i, (i + j) % n)
        if k & 1:
            half = n // 2
            for i in range(half):
                H.add_edge(i, i + half)
    else:
        offset = (k - 1) // 2
        for i in range(n):
            for j in range(1, offset + 1):
                H.add_edge(i, (i - j) % n)
                H.add_edge(i, (i + j) % n)
        half = n // 2
        for i in range(half + 1):
            H.add_edge(i, (i + half) % n)
    return H