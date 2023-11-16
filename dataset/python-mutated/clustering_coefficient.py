import networkx as nx
from networkx.utils import not_implemented_for, py_random_state
__all__ = ['average_clustering']

@not_implemented_for('directed')
@py_random_state(2)
@nx._dispatch(name='approximate_average_clustering')
def average_clustering(G, trials=1000, seed=None):
    if False:
        i = 10
        return i + 15
    'Estimates the average clustering coefficient of G.\n\n    The local clustering of each node in `G` is the fraction of triangles\n    that actually exist over all possible triangles in its neighborhood.\n    The average clustering coefficient of a graph `G` is the mean of\n    local clusterings.\n\n    This function finds an approximate average clustering coefficient\n    for G by repeating `n` times (defined in `trials`) the following\n    experiment: choose a node at random, choose two of its neighbors\n    at random, and check if they are connected. The approximate\n    coefficient is the fraction of triangles found over the number\n    of trials [1]_.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n\n    trials : integer\n        Number of trials to perform (default 1000).\n\n    seed : integer, random_state, or None (default)\n        Indicator of random number generation state.\n        See :ref:`Randomness<randomness>`.\n\n    Returns\n    -------\n    c : float\n        Approximated average clustering coefficient.\n\n    Examples\n    --------\n    >>> from networkx.algorithms import approximation\n    >>> G = nx.erdos_renyi_graph(10, 0.2, seed=10)\n    >>> approximation.average_clustering(G, trials=1000, seed=10)\n    0.214\n\n    References\n    ----------\n    .. [1] Schank, Thomas, and Dorothea Wagner. Approximating clustering\n       coefficient and transitivity. Universität Karlsruhe, Fakultät für\n       Informatik, 2004.\n       https://doi.org/10.5445/IR/1000001239\n\n    '
    n = len(G)
    triangles = 0
    nodes = list(G)
    for i in [int(seed.random() * n) for i in range(trials)]:
        nbrs = list(G[nodes[i]])
        if len(nbrs) < 2:
            continue
        (u, v) = seed.sample(nbrs, 2)
        if u in G[v]:
            triangles += 1
    return triangles / trials