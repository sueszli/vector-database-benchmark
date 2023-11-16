"""
Cuthill-McKee ordering of graph nodes to produce sparse matrices
"""
from collections import deque
from operator import itemgetter
import networkx as nx
from ..utils import arbitrary_element
__all__ = ['cuthill_mckee_ordering', 'reverse_cuthill_mckee_ordering']

def cuthill_mckee_ordering(G, heuristic=None):
    if False:
        i = 10
        return i + 15
    'Generate an ordering (permutation) of the graph nodes to make\n    a sparse matrix.\n\n    Uses the Cuthill-McKee heuristic (based on breadth-first search) [1]_.\n\n    Parameters\n    ----------\n    G : graph\n      A NetworkX graph\n\n    heuristic : function, optional\n      Function to choose starting node for RCM algorithm.  If None\n      a node from a pseudo-peripheral pair is used.  A user-defined function\n      can be supplied that takes a graph object and returns a single node.\n\n    Returns\n    -------\n    nodes : generator\n       Generator of nodes in Cuthill-McKee ordering.\n\n    Examples\n    --------\n    >>> from networkx.utils import cuthill_mckee_ordering\n    >>> G = nx.path_graph(4)\n    >>> rcm = list(cuthill_mckee_ordering(G))\n    >>> A = nx.adjacency_matrix(G, nodelist=rcm)\n\n    Smallest degree node as heuristic function:\n\n    >>> def smallest_degree(G):\n    ...     return min(G, key=G.degree)\n    >>> rcm = list(cuthill_mckee_ordering(G, heuristic=smallest_degree))\n\n\n    See Also\n    --------\n    reverse_cuthill_mckee_ordering\n\n    Notes\n    -----\n    The optimal solution the bandwidth reduction is NP-complete [2]_.\n\n\n    References\n    ----------\n    .. [1] E. Cuthill and J. McKee.\n       Reducing the bandwidth of sparse symmetric matrices,\n       In Proc. 24th Nat. Conf. ACM, pages 157-172, 1969.\n       http://doi.acm.org/10.1145/800195.805928\n    .. [2]  Steven S. Skiena. 1997. The Algorithm Design Manual.\n       Springer-Verlag New York, Inc., New York, NY, USA.\n    '
    for c in nx.connected_components(G):
        yield from connected_cuthill_mckee_ordering(G.subgraph(c), heuristic)

def reverse_cuthill_mckee_ordering(G, heuristic=None):
    if False:
        for i in range(10):
            print('nop')
    'Generate an ordering (permutation) of the graph nodes to make\n    a sparse matrix.\n\n    Uses the reverse Cuthill-McKee heuristic (based on breadth-first search)\n    [1]_.\n\n    Parameters\n    ----------\n    G : graph\n      A NetworkX graph\n\n    heuristic : function, optional\n      Function to choose starting node for RCM algorithm.  If None\n      a node from a pseudo-peripheral pair is used.  A user-defined function\n      can be supplied that takes a graph object and returns a single node.\n\n    Returns\n    -------\n    nodes : generator\n       Generator of nodes in reverse Cuthill-McKee ordering.\n\n    Examples\n    --------\n    >>> from networkx.utils import reverse_cuthill_mckee_ordering\n    >>> G = nx.path_graph(4)\n    >>> rcm = list(reverse_cuthill_mckee_ordering(G))\n    >>> A = nx.adjacency_matrix(G, nodelist=rcm)\n\n    Smallest degree node as heuristic function:\n\n    >>> def smallest_degree(G):\n    ...     return min(G, key=G.degree)\n    >>> rcm = list(reverse_cuthill_mckee_ordering(G, heuristic=smallest_degree))\n\n\n    See Also\n    --------\n    cuthill_mckee_ordering\n\n    Notes\n    -----\n    The optimal solution the bandwidth reduction is NP-complete [2]_.\n\n    References\n    ----------\n    .. [1] E. Cuthill and J. McKee.\n       Reducing the bandwidth of sparse symmetric matrices,\n       In Proc. 24th Nat. Conf. ACM, pages 157-72, 1969.\n       http://doi.acm.org/10.1145/800195.805928\n    .. [2]  Steven S. Skiena. 1997. The Algorithm Design Manual.\n       Springer-Verlag New York, Inc., New York, NY, USA.\n    '
    return reversed(list(cuthill_mckee_ordering(G, heuristic=heuristic)))

def connected_cuthill_mckee_ordering(G, heuristic=None):
    if False:
        i = 10
        return i + 15
    if heuristic is None:
        start = pseudo_peripheral_node(G)
    else:
        start = heuristic(G)
    visited = {start}
    queue = deque([start])
    while queue:
        parent = queue.popleft()
        yield parent
        nd = sorted(G.degree(set(G[parent]) - visited), key=itemgetter(1))
        children = [n for (n, d) in nd]
        visited.update(children)
        queue.extend(children)

def pseudo_peripheral_node(G):
    if False:
        for i in range(10):
            print('nop')
    u = arbitrary_element(G)
    lp = 0
    v = u
    while True:
        spl = dict(nx.shortest_path_length(G, v))
        l = max(spl.values())
        if l <= lp:
            break
        lp = l
        farthest = (n for (n, dist) in spl.items() if dist == l)
        (v, deg) = min(G.degree(farthest), key=itemgetter(1))
    return v