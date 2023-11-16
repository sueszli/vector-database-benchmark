"""Functions for computing large cliques and maximum independent sets."""
import networkx as nx
from networkx.algorithms.approximation import ramsey
from networkx.utils import not_implemented_for
__all__ = ['clique_removal', 'max_clique', 'large_clique_size', 'maximum_independent_set']

@not_implemented_for('directed')
@not_implemented_for('multigraph')
@nx._dispatch
def maximum_independent_set(G):
    if False:
        for i in range(10):
            print('nop')
    'Returns an approximate maximum independent set.\n\n    Independent set or stable set is a set of vertices in a graph, no two of\n    which are adjacent. That is, it is a set I of vertices such that for every\n    two vertices in I, there is no edge connecting the two. Equivalently, each\n    edge in the graph has at most one endpoint in I. The size of an independent\n    set is the number of vertices it contains [1]_.\n\n    A maximum independent set is a largest independent set for a given graph G\n    and its size is denoted $\\alpha(G)$. The problem of finding such a set is called\n    the maximum independent set problem and is an NP-hard optimization problem.\n    As such, it is unlikely that there exists an efficient algorithm for finding\n    a maximum independent set of a graph.\n\n    The Independent Set algorithm is based on [2]_.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n        Undirected graph\n\n    Returns\n    -------\n    iset : Set\n        The apx-maximum independent set\n\n    Examples\n    --------\n    >>> G = nx.path_graph(10)\n    >>> nx.approximation.maximum_independent_set(G)\n    {0, 2, 4, 6, 9}\n\n    Raises\n    ------\n    NetworkXNotImplemented\n        If the graph is directed or is a multigraph.\n\n    Notes\n    -----\n    Finds the $O(|V|/(log|V|)^2)$ apx of independent set in the worst case.\n\n    References\n    ----------\n    .. [1] `Wikipedia: Independent set\n        <https://en.wikipedia.org/wiki/Independent_set_(graph_theory)>`_\n    .. [2] Boppana, R., & Halldórsson, M. M. (1992).\n       Approximating maximum independent sets by excluding subgraphs.\n       BIT Numerical Mathematics, 32(2), 180–196. Springer.\n    '
    (iset, _) = clique_removal(G)
    return iset

@not_implemented_for('directed')
@not_implemented_for('multigraph')
@nx._dispatch
def max_clique(G):
    if False:
        while True:
            i = 10
    'Find the Maximum Clique\n\n    Finds the $O(|V|/(log|V|)^2)$ apx of maximum clique/independent set\n    in the worst case.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n        Undirected graph\n\n    Returns\n    -------\n    clique : set\n        The apx-maximum clique of the graph\n\n    Examples\n    --------\n    >>> G = nx.path_graph(10)\n    >>> nx.approximation.max_clique(G)\n    {8, 9}\n\n    Raises\n    ------\n    NetworkXNotImplemented\n        If the graph is directed or is a multigraph.\n\n    Notes\n    -----\n    A clique in an undirected graph G = (V, E) is a subset of the vertex set\n    `C \\subseteq V` such that for every two vertices in C there exists an edge\n    connecting the two. This is equivalent to saying that the subgraph\n    induced by C is complete (in some cases, the term clique may also refer\n    to the subgraph).\n\n    A maximum clique is a clique of the largest possible size in a given graph.\n    The clique number `\\omega(G)` of a graph G is the number of\n    vertices in a maximum clique in G. The intersection number of\n    G is the smallest number of cliques that together cover all edges of G.\n\n    https://en.wikipedia.org/wiki/Maximum_clique\n\n    References\n    ----------\n    .. [1] Boppana, R., & Halldórsson, M. M. (1992).\n        Approximating maximum independent sets by excluding subgraphs.\n        BIT Numerical Mathematics, 32(2), 180–196. Springer.\n        doi:10.1007/BF01994876\n    '
    cgraph = nx.complement(G)
    (iset, _) = clique_removal(cgraph)
    return iset

@not_implemented_for('directed')
@not_implemented_for('multigraph')
@nx._dispatch
def clique_removal(G):
    if False:
        print('Hello World!')
    'Repeatedly remove cliques from the graph.\n\n    Results in a $O(|V|/(\\log |V|)^2)$ approximation of maximum clique\n    and independent set. Returns the largest independent set found, along\n    with found maximal cliques.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n        Undirected graph\n\n    Returns\n    -------\n    max_ind_cliques : (set, list) tuple\n        2-tuple of Maximal Independent Set and list of maximal cliques (sets).\n\n    Examples\n    --------\n    >>> G = nx.path_graph(10)\n    >>> nx.approximation.clique_removal(G)\n    ({0, 2, 4, 6, 9}, [{0, 1}, {2, 3}, {4, 5}, {6, 7}, {8, 9}])\n\n    Raises\n    ------\n    NetworkXNotImplemented\n        If the graph is directed or is a multigraph.\n\n    References\n    ----------\n    .. [1] Boppana, R., & Halldórsson, M. M. (1992).\n        Approximating maximum independent sets by excluding subgraphs.\n        BIT Numerical Mathematics, 32(2), 180–196. Springer.\n    '
    graph = G.copy()
    (c_i, i_i) = ramsey.ramsey_R2(graph)
    cliques = [c_i]
    isets = [i_i]
    while graph:
        graph.remove_nodes_from(c_i)
        (c_i, i_i) = ramsey.ramsey_R2(graph)
        if c_i:
            cliques.append(c_i)
        if i_i:
            isets.append(i_i)
    maxiset = max(isets, key=len)
    return (maxiset, cliques)

@not_implemented_for('directed')
@not_implemented_for('multigraph')
@nx._dispatch
def large_clique_size(G):
    if False:
        for i in range(10):
            print('nop')
    'Find the size of a large clique in a graph.\n\n    A *clique* is a subset of nodes in which each pair of nodes is\n    adjacent. This function is a heuristic for finding the size of a\n    large clique in the graph.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n\n    Returns\n    -------\n    k: integer\n       The size of a large clique in the graph.\n\n    Examples\n    --------\n    >>> G = nx.path_graph(10)\n    >>> nx.approximation.large_clique_size(G)\n    2\n\n    Raises\n    ------\n    NetworkXNotImplemented\n        If the graph is directed or is a multigraph.\n\n    Notes\n    -----\n    This implementation is from [1]_. Its worst case time complexity is\n    :math:`O(n d^2)`, where *n* is the number of nodes in the graph and\n    *d* is the maximum degree.\n\n    This function is a heuristic, which means it may work well in\n    practice, but there is no rigorous mathematical guarantee on the\n    ratio between the returned number and the actual largest clique size\n    in the graph.\n\n    References\n    ----------\n    .. [1] Pattabiraman, Bharath, et al.\n       "Fast Algorithms for the Maximum Clique Problem on Massive Graphs\n       with Applications to Overlapping Community Detection."\n       *Internet Mathematics* 11.4-5 (2015): 421--448.\n       <https://doi.org/10.1080/15427951.2014.986778>\n\n    See also\n    --------\n\n    :func:`networkx.algorithms.approximation.clique.max_clique`\n        A function that returns an approximate maximum clique with a\n        guarantee on the approximation ratio.\n\n    :mod:`networkx.algorithms.clique`\n        Functions for finding the exact maximum clique in a graph.\n\n    '
    degrees = G.degree

    def _clique_heuristic(G, U, size, best_size):
        if False:
            print('Hello World!')
        if not U:
            return max(best_size, size)
        u = max(U, key=degrees)
        U.remove(u)
        N_prime = {v for v in G[u] if degrees[v] >= best_size}
        return _clique_heuristic(G, U & N_prime, size + 1, best_size)
    best_size = 0
    nodes = (u for u in G if degrees[u] >= best_size)
    for u in nodes:
        neighbors = {v for v in G[u] if degrees[v] >= best_size}
        best_size = _clique_heuristic(G, neighbors, 1, best_size)
    return best_size