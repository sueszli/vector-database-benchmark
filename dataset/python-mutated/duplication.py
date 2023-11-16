"""Functions for generating graphs based on the "duplication" method.

These graph generators start with a small initial graph then duplicate
nodes and (partially) duplicate their edges. These functions are
generally inspired by biological networks.

"""
import networkx as nx
from networkx.exception import NetworkXError
from networkx.utils import py_random_state
__all__ = ['partial_duplication_graph', 'duplication_divergence_graph']

@py_random_state(4)
@nx._dispatch(graphs=None)
def partial_duplication_graph(N, n, p, q, seed=None):
    if False:
        for i in range(10):
            print('nop')
    'Returns a random graph using the partial duplication model.\n\n    Parameters\n    ----------\n    N : int\n        The total number of nodes in the final graph.\n\n    n : int\n        The number of nodes in the initial clique.\n\n    p : float\n        The probability of joining each neighbor of a node to the\n        duplicate node. Must be a number in the between zero and one,\n        inclusive.\n\n    q : float\n        The probability of joining the source node to the duplicate\n        node. Must be a number in the between zero and one, inclusive.\n\n    seed : integer, random_state, or None (default)\n        Indicator of random number generation state.\n        See :ref:`Randomness<randomness>`.\n\n    Notes\n    -----\n    A graph of nodes is grown by creating a fully connected graph\n    of size `n`. The following procedure is then repeated until\n    a total of `N` nodes have been reached.\n\n    1. A random node, *u*, is picked and a new node, *v*, is created.\n    2. For each neighbor of *u* an edge from the neighbor to *v* is created\n       with probability `p`.\n    3. An edge from *u* to *v* is created with probability `q`.\n\n    This algorithm appears in [1].\n\n    This implementation allows the possibility of generating\n    disconnected graphs.\n\n    References\n    ----------\n    .. [1] Knudsen Michael, and Carsten Wiuf. "A Markov chain approach to\n           randomly grown graphs." Journal of Applied Mathematics 2008.\n           <https://doi.org/10.1155/2008/190836>\n\n    '
    if p < 0 or p > 1 or q < 0 or (q > 1):
        msg = 'partial duplication graph must have 0 <= p, q <= 1.'
        raise NetworkXError(msg)
    if n > N:
        raise NetworkXError('partial duplication graph must have n <= N.')
    G = nx.complete_graph(n)
    for new_node in range(n, N):
        src_node = seed.randint(0, new_node - 1)
        G.add_node(new_node)
        for neighbor_node in list(nx.all_neighbors(G, src_node)):
            if seed.random() < p:
                G.add_edge(new_node, neighbor_node)
        if seed.random() < q:
            G.add_edge(new_node, src_node)
    return G

@py_random_state(2)
@nx._dispatch(graphs=None)
def duplication_divergence_graph(n, p, seed=None):
    if False:
        i = 10
        return i + 15
    'Returns an undirected graph using the duplication-divergence model.\n\n    A graph of `n` nodes is created by duplicating the initial nodes\n    and retaining edges incident to the original nodes with a retention\n    probability `p`.\n\n    Parameters\n    ----------\n    n : int\n        The desired number of nodes in the graph.\n    p : float\n        The probability for retaining the edge of the replicated node.\n    seed : integer, random_state, or None (default)\n        Indicator of random number generation state.\n        See :ref:`Randomness<randomness>`.\n\n    Returns\n    -------\n    G : Graph\n\n    Raises\n    ------\n    NetworkXError\n        If `p` is not a valid probability.\n        If `n` is less than 2.\n\n    Notes\n    -----\n    This algorithm appears in [1].\n\n    This implementation disallows the possibility of generating\n    disconnected graphs.\n\n    References\n    ----------\n    .. [1] I. Ispolatov, P. L. Krapivsky, A. Yuryev,\n       "Duplication-divergence model of protein interaction network",\n       Phys. Rev. E, 71, 061911, 2005.\n\n    '
    if p > 1 or p < 0:
        msg = f'NetworkXError p={p} is not in [0,1].'
        raise nx.NetworkXError(msg)
    if n < 2:
        msg = 'n must be greater than or equal to 2'
        raise nx.NetworkXError(msg)
    G = nx.Graph()
    G.add_edge(0, 1)
    i = 2
    while i < n:
        random_node = seed.choice(list(G))
        G.add_node(i)
        flag = False
        for nbr in G.neighbors(random_node):
            if seed.random() < p:
                G.add_edge(i, nbr)
                flag = True
        if not flag:
            G.remove_node(i)
        else:
            i += 1
    return G