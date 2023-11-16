"""
Algorithm to find a maximal (not maximum) independent set.

"""
import networkx as nx
from networkx.utils import not_implemented_for, py_random_state
__all__ = ['maximal_independent_set']

@not_implemented_for('directed')
@py_random_state(2)
@nx._dispatch
def maximal_independent_set(G, nodes=None, seed=None):
    if False:
        while True:
            i = 10
    'Returns a random maximal independent set guaranteed to contain\n    a given set of nodes.\n\n    An independent set is a set of nodes such that the subgraph\n    of G induced by these nodes contains no edges. A maximal\n    independent set is an independent set such that it is not possible\n    to add a new node and still get an independent set.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n\n    nodes : list or iterable\n       Nodes that must be part of the independent set. This set of nodes\n       must be independent.\n\n    seed : integer, random_state, or None (default)\n        Indicator of random number generation state.\n        See :ref:`Randomness<randomness>`.\n\n    Returns\n    -------\n    indep_nodes : list\n       List of nodes that are part of a maximal independent set.\n\n    Raises\n    ------\n    NetworkXUnfeasible\n       If the nodes in the provided list are not part of the graph or\n       do not form an independent set, an exception is raised.\n\n    NetworkXNotImplemented\n        If `G` is directed.\n\n    Examples\n    --------\n    >>> G = nx.path_graph(5)\n    >>> nx.maximal_independent_set(G)  # doctest: +SKIP\n    [4, 0, 2]\n    >>> nx.maximal_independent_set(G, [1])  # doctest: +SKIP\n    [1, 3]\n\n    Notes\n    -----\n    This algorithm does not solve the maximum independent set problem.\n\n    '
    if not nodes:
        nodes = {seed.choice(list(G))}
    else:
        nodes = set(nodes)
    if not nodes.issubset(G):
        raise nx.NetworkXUnfeasible(f'{nodes} is not a subset of the nodes of G')
    neighbors = set.union(*[set(G.adj[v]) for v in nodes])
    if set.intersection(neighbors, nodes):
        raise nx.NetworkXUnfeasible(f'{nodes} is not an independent set of G')
    indep_nodes = list(nodes)
    available_nodes = set(G.nodes()).difference(neighbors.union(nodes))
    while available_nodes:
        node = seed.choice(list(available_nodes))
        indep_nodes.append(node)
        available_nodes.difference_update(list(G.adj[node]) + [node])
    return indep_nodes