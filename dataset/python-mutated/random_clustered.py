"""Generate graphs with given degree and triangle sequence.
"""
import networkx as nx
from networkx.utils import py_random_state
__all__ = ['random_clustered_graph']

@py_random_state(2)
@nx._dispatch(graphs=None)
def random_clustered_graph(joint_degree_sequence, create_using=None, seed=None):
    if False:
        while True:
            i = 10
    'Generate a random graph with the given joint independent edge degree and\n    triangle degree sequence.\n\n    This uses a configuration model-like approach to generate a random graph\n    (with parallel edges and self-loops) by randomly assigning edges to match\n    the given joint degree sequence.\n\n    The joint degree sequence is a list of pairs of integers of the form\n    $[(d_{1,i}, d_{1,t}), \\dotsc, (d_{n,i}, d_{n,t})]$. According to this list,\n    vertex $u$ is a member of $d_{u,t}$ triangles and has $d_{u, i}$ other\n    edges. The number $d_{u,t}$ is the *triangle degree* of $u$ and the number\n    $d_{u,i}$ is the *independent edge degree*.\n\n    Parameters\n    ----------\n    joint_degree_sequence : list of integer pairs\n        Each list entry corresponds to the independent edge degree and\n        triangle degree of a node.\n    create_using : NetworkX graph constructor, optional (default MultiGraph)\n       Graph type to create. If graph instance, then cleared before populated.\n    seed : integer, random_state, or None (default)\n        Indicator of random number generation state.\n        See :ref:`Randomness<randomness>`.\n\n    Returns\n    -------\n    G : MultiGraph\n        A graph with the specified degree sequence. Nodes are labeled\n        starting at 0 with an index corresponding to the position in\n        deg_sequence.\n\n    Raises\n    ------\n    NetworkXError\n        If the independent edge degree sequence sum is not even\n        or the triangle degree sequence sum is not divisible by 3.\n\n    Notes\n    -----\n    As described by Miller [1]_ (see also Newman [2]_ for an equivalent\n    description).\n\n    A non-graphical degree sequence (not realizable by some simple\n    graph) is allowed since this function returns graphs with self\n    loops and parallel edges.  An exception is raised if the\n    independent degree sequence does not have an even sum or the\n    triangle degree sequence sum is not divisible by 3.\n\n    This configuration model-like construction process can lead to\n    duplicate edges and loops.  You can remove the self-loops and\n    parallel edges (see below) which will likely result in a graph\n    that doesn\'t have the exact degree sequence specified.  This\n    "finite-size effect" decreases as the size of the graph increases.\n\n    References\n    ----------\n    .. [1] Joel C. Miller. "Percolation and epidemics in random clustered\n           networks". In: Physical review. E, Statistical, nonlinear, and soft\n           matter physics 80 (2 Part 1 August 2009).\n    .. [2] M. E. J. Newman. "Random Graphs with Clustering".\n           In: Physical Review Letters 103 (5 July 2009)\n\n    Examples\n    --------\n    >>> deg = [(1, 0), (1, 0), (1, 0), (2, 0), (1, 0), (2, 1), (0, 1), (0, 1)]\n    >>> G = nx.random_clustered_graph(deg)\n\n    To remove parallel edges:\n\n    >>> G = nx.Graph(G)\n\n    To remove self loops:\n\n    >>> G.remove_edges_from(nx.selfloop_edges(G))\n\n    '
    joint_degree_sequence = list(joint_degree_sequence)
    N = len(joint_degree_sequence)
    G = nx.empty_graph(N, create_using, default=nx.MultiGraph)
    if G.is_directed():
        raise nx.NetworkXError('Directed Graph not supported')
    ilist = []
    tlist = []
    for n in G:
        degrees = joint_degree_sequence[n]
        for icount in range(degrees[0]):
            ilist.append(n)
        for tcount in range(degrees[1]):
            tlist.append(n)
    if len(ilist) % 2 != 0 or len(tlist) % 3 != 0:
        raise nx.NetworkXError('Invalid degree sequence')
    seed.shuffle(ilist)
    seed.shuffle(tlist)
    while ilist:
        G.add_edge(ilist.pop(), ilist.pop())
    while tlist:
        n1 = tlist.pop()
        n2 = tlist.pop()
        n3 = tlist.pop()
        G.add_edges_from([(n1, n2), (n1, n3), (n2, n3)])
    return G