"""Functions concerning tournament graphs.

A `tournament graph`_ is a complete oriented graph. In other words, it
is a directed graph in which there is exactly one directed edge joining
each pair of distinct nodes. For each function in this module that
accepts a graph as input, you must provide a tournament graph. The
responsibility is on the caller to ensure that the graph is a tournament
graph:

    >>> G = nx.DiGraph([(0, 1), (1, 2), (2, 0)])
    >>> nx.is_tournament(G)
    True

To access the functions in this module, you must access them through the
:mod:`networkx.tournament` module::

    >>> nx.tournament.is_reachable(G, 0, 1)
    True

.. _tournament graph: https://en.wikipedia.org/wiki/Tournament_%28graph_theory%29

"""
from itertools import combinations
import networkx as nx
from networkx.algorithms.simple_paths import is_simple_path as is_path
from networkx.utils import arbitrary_element, not_implemented_for, py_random_state
__all__ = ['hamiltonian_path', 'is_reachable', 'is_strongly_connected', 'is_tournament', 'random_tournament', 'score_sequence']

def index_satisfying(iterable, condition):
    if False:
        print('Hello World!')
    'Returns the index of the first element in `iterable` that\n    satisfies the given condition.\n\n    If no such element is found (that is, when the iterable is\n    exhausted), this returns the length of the iterable (that is, one\n    greater than the last index of the iterable).\n\n    `iterable` must not be empty. If `iterable` is empty, this\n    function raises :exc:`ValueError`.\n\n    '
    for (i, x) in enumerate(iterable):
        if condition(x):
            return i
    try:
        return i + 1
    except NameError as err:
        raise ValueError('iterable must be non-empty') from err

@not_implemented_for('undirected')
@not_implemented_for('multigraph')
@nx._dispatch
def is_tournament(G):
    if False:
        i = 10
        return i + 15
    'Returns True if and only if `G` is a tournament.\n\n    A tournament is a directed graph, with neither self-loops nor\n    multi-edges, in which there is exactly one directed edge joining\n    each pair of distinct nodes.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n        A directed graph representing a tournament.\n\n    Returns\n    -------\n    bool\n        Whether the given graph is a tournament graph.\n\n    Examples\n    --------\n    >>> G = nx.DiGraph([(0, 1), (1, 2), (2, 0)])\n    >>> nx.is_tournament(G)\n    True\n\n    Notes\n    -----\n    Some definitions require a self-loop on each node, but that is not\n    the convention used here.\n\n    '
    return all(((v in G[u]) ^ (u in G[v]) for (u, v) in combinations(G, 2))) and nx.number_of_selfloops(G) == 0

@not_implemented_for('undirected')
@not_implemented_for('multigraph')
@nx._dispatch
def hamiltonian_path(G):
    if False:
        while True:
            i = 10
    'Returns a Hamiltonian path in the given tournament graph.\n\n    Each tournament has a Hamiltonian path. If furthermore, the\n    tournament is strongly connected, then the returned Hamiltonian path\n    is a Hamiltonian cycle (by joining the endpoints of the path).\n\n    Parameters\n    ----------\n    G : NetworkX graph\n        A directed graph representing a tournament.\n\n    Returns\n    -------\n    path : list\n        A list of nodes which form a Hamiltonian path in `G`.\n\n    Examples\n    --------\n    >>> G = nx.DiGraph([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])\n    >>> nx.is_tournament(G)\n    True\n    >>> nx.tournament.hamiltonian_path(G)\n    [0, 1, 2, 3]\n\n    Notes\n    -----\n    This is a recursive implementation with an asymptotic running time\n    of $O(n^2)$, ignoring multiplicative polylogarithmic factors, where\n    $n$ is the number of nodes in the graph.\n\n    '
    if len(G) == 0:
        return []
    if len(G) == 1:
        return [arbitrary_element(G)]
    v = arbitrary_element(G)
    hampath = hamiltonian_path(G.subgraph(set(G) - {v}))
    index = index_satisfying(hampath, lambda u: v not in G[u])
    hampath.insert(index, v)
    return hampath

@py_random_state(1)
@nx._dispatch(graphs=None)
def random_tournament(n, seed=None):
    if False:
        return 10
    'Returns a random tournament graph on `n` nodes.\n\n    Parameters\n    ----------\n    n : int\n        The number of nodes in the returned graph.\n    seed : integer, random_state, or None (default)\n        Indicator of random number generation state.\n        See :ref:`Randomness<randomness>`.\n\n    Returns\n    -------\n    G : DiGraph\n        A tournament on `n` nodes, with exactly one directed edge joining\n        each pair of distinct nodes.\n\n    Notes\n    -----\n    This algorithm adds, for each pair of distinct nodes, an edge with\n    uniformly random orientation. In other words, `\\binom{n}{2}` flips\n    of an unbiased coin decide the orientations of the edges in the\n    graph.\n\n    '
    coins = (seed.random() for i in range(n * (n - 1) // 2))
    pairs = combinations(range(n), 2)
    edges = ((u, v) if r < 0.5 else (v, u) for ((u, v), r) in zip(pairs, coins))
    return nx.DiGraph(edges)

@not_implemented_for('undirected')
@not_implemented_for('multigraph')
@nx._dispatch
def score_sequence(G):
    if False:
        i = 10
        return i + 15
    'Returns the score sequence for the given tournament graph.\n\n    The score sequence is the sorted list of the out-degrees of the\n    nodes of the graph.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n        A directed graph representing a tournament.\n\n    Returns\n    -------\n    list\n        A sorted list of the out-degrees of the nodes of `G`.\n\n    Examples\n    --------\n    >>> G = nx.DiGraph([(1, 0), (1, 3), (0, 2), (0, 3), (2, 1), (3, 2)])\n    >>> nx.is_tournament(G)\n    True\n    >>> nx.tournament.score_sequence(G)\n    [1, 1, 2, 2]\n\n    '
    return sorted((d for (v, d) in G.out_degree()))

@not_implemented_for('undirected')
@not_implemented_for('multigraph')
@nx._dispatch
def tournament_matrix(G):
    if False:
        i = 10
        return i + 15
    'Returns the tournament matrix for the given tournament graph.\n\n    This function requires SciPy.\n\n    The *tournament matrix* of a tournament graph with edge set *E* is\n    the matrix *T* defined by\n\n    .. math::\n\n       T_{i j} =\n       \\begin{cases}\n       +1 & \\text{if } (i, j) \\in E \\\\\n       -1 & \\text{if } (j, i) \\in E \\\\\n       0 & \\text{if } i == j.\n       \\end{cases}\n\n    An equivalent definition is `T = A - A^T`, where *A* is the\n    adjacency matrix of the graph `G`.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n        A directed graph representing a tournament.\n\n    Returns\n    -------\n    SciPy sparse array\n        The tournament matrix of the tournament graph `G`.\n\n    Raises\n    ------\n    ImportError\n        If SciPy is not available.\n\n    '
    A = nx.adjacency_matrix(G)
    return A - A.T

@not_implemented_for('undirected')
@not_implemented_for('multigraph')
@nx._dispatch
def is_reachable(G, s, t):
    if False:
        for i in range(10):
            print('nop')
    'Decides whether there is a path from `s` to `t` in the\n    tournament.\n\n    This function is more theoretically efficient than the reachability\n    checks than the shortest path algorithms in\n    :mod:`networkx.algorithms.shortest_paths`.\n\n    The given graph **must** be a tournament, otherwise this function\'s\n    behavior is undefined.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n        A directed graph representing a tournament.\n\n    s : node\n        A node in the graph.\n\n    t : node\n        A node in the graph.\n\n    Returns\n    -------\n    bool\n        Whether there is a path from `s` to `t` in `G`.\n\n    Examples\n    --------\n    >>> G = nx.DiGraph([(1, 0), (1, 3), (1, 2), (2, 3), (2, 0), (3, 0)])\n    >>> nx.is_tournament(G)\n    True\n    >>> nx.tournament.is_reachable(G, 1, 3)\n    True\n    >>> nx.tournament.is_reachable(G, 3, 2)\n    False\n\n    Notes\n    -----\n    Although this function is more theoretically efficient than the\n    generic shortest path functions, a speedup requires the use of\n    parallelism. Though it may in the future, the current implementation\n    does not use parallelism, thus you may not see much of a speedup.\n\n    This algorithm comes from [1].\n\n    References\n    ----------\n    .. [1] Tantau, Till.\n           "A note on the complexity of the reachability problem for\n           tournaments."\n           *Electronic Colloquium on Computational Complexity*. 2001.\n           <http://eccc.hpi-web.de/report/2001/092/>\n    '

    def two_neighborhood(G, v):
        if False:
            i = 10
            return i + 15
        'Returns the set of nodes at distance at most two from `v`.\n\n        `G` must be a graph and `v` a node in that graph.\n\n        The returned set includes the nodes at distance zero (that is,\n        the node `v` itself), the nodes at distance one (that is, the\n        out-neighbors of `v`), and the nodes at distance two.\n\n        '
        return {x for x in G if x == v or x in G[v] or any((is_path(G, [v, z, x]) for z in G))}

    def is_closed(G, nodes):
        if False:
            for i in range(10):
                print('nop')
        'Decides whether the given set of nodes is closed.\n\n        A set *S* of nodes is *closed* if for each node *u* in the graph\n        not in *S* and for each node *v* in *S*, there is an edge from\n        *u* to *v*.\n\n        '
        return all((v in G[u] for u in set(G) - nodes for v in nodes))
    neighborhoods = [two_neighborhood(G, v) for v in G]
    return all((not (is_closed(G, S) and s in S and (t not in S)) for S in neighborhoods))

@not_implemented_for('undirected')
@not_implemented_for('multigraph')
@nx._dispatch(name='tournament_is_strongly_connected')
def is_strongly_connected(G):
    if False:
        while True:
            i = 10
    'Decides whether the given tournament is strongly connected.\n\n    This function is more theoretically efficient than the\n    :func:`~networkx.algorithms.components.is_strongly_connected`\n    function.\n\n    The given graph **must** be a tournament, otherwise this function\'s\n    behavior is undefined.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n        A directed graph representing a tournament.\n\n    Returns\n    -------\n    bool\n        Whether the tournament is strongly connected.\n\n    Examples\n    --------\n    >>> G = nx.DiGraph([(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 0)])\n    >>> nx.is_tournament(G)\n    True\n    >>> nx.tournament.is_strongly_connected(G)\n    True\n    >>> G.remove_edge(3, 0)\n    >>> G.add_edge(0, 3)\n    >>> nx.is_tournament(G)\n    True\n    >>> nx.tournament.is_strongly_connected(G)\n    False\n\n    Notes\n    -----\n    Although this function is more theoretically efficient than the\n    generic strong connectivity function, a speedup requires the use of\n    parallelism. Though it may in the future, the current implementation\n    does not use parallelism, thus you may not see much of a speedup.\n\n    This algorithm comes from [1].\n\n    References\n    ----------\n    .. [1] Tantau, Till.\n           "A note on the complexity of the reachability problem for\n           tournaments."\n           *Electronic Colloquium on Computational Complexity*. 2001.\n           <http://eccc.hpi-web.de/report/2001/092/>\n\n    '
    return all((is_reachable(G, u, v) for u in G for v in G))