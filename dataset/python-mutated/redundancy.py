"""Node redundancy for bipartite graphs."""
from itertools import combinations
import networkx as nx
from networkx import NetworkXError
__all__ = ['node_redundancy']

@nx._dispatch
def node_redundancy(G, nodes=None):
    if False:
        i = 10
        return i + 15
    "Computes the node redundancy coefficients for the nodes in the bipartite\n    graph `G`.\n\n    The redundancy coefficient of a node `v` is the fraction of pairs of\n    neighbors of `v` that are both linked to other nodes. In a one-mode\n    projection these nodes would be linked together even if `v` were\n    not there.\n\n    More formally, for any vertex `v`, the *redundancy coefficient of `v`* is\n    defined by\n\n    .. math::\n\n        rc(v) = \\frac{|\\{\\{u, w\\} \\subseteq N(v),\n        \\: \\exists v' \\neq  v,\\: (v',u) \\in E\\:\n        \\mathrm{and}\\: (v',w) \\in E\\}|}{ \\frac{|N(v)|(|N(v)|-1)}{2}},\n\n    where `N(v)` is the set of neighbors of `v` in `G`.\n\n    Parameters\n    ----------\n    G : graph\n        A bipartite graph\n\n    nodes : list or iterable (optional)\n        Compute redundancy for these nodes. The default is all nodes in G.\n\n    Returns\n    -------\n    redundancy : dictionary\n        A dictionary keyed by node with the node redundancy value.\n\n    Examples\n    --------\n    Compute the redundancy coefficient of each node in a graph::\n\n        >>> from networkx.algorithms import bipartite\n        >>> G = nx.cycle_graph(4)\n        >>> rc = bipartite.node_redundancy(G)\n        >>> rc[0]\n        1.0\n\n    Compute the average redundancy for the graph::\n\n        >>> from networkx.algorithms import bipartite\n        >>> G = nx.cycle_graph(4)\n        >>> rc = bipartite.node_redundancy(G)\n        >>> sum(rc.values()) / len(G)\n        1.0\n\n    Compute the average redundancy for a set of nodes::\n\n        >>> from networkx.algorithms import bipartite\n        >>> G = nx.cycle_graph(4)\n        >>> rc = bipartite.node_redundancy(G)\n        >>> nodes = [0, 2]\n        >>> sum(rc[n] for n in nodes) / len(nodes)\n        1.0\n\n    Raises\n    ------\n    NetworkXError\n        If any of the nodes in the graph (or in `nodes`, if specified) has\n        (out-)degree less than two (which would result in division by zero,\n        according to the definition of the redundancy coefficient).\n\n    References\n    ----------\n    .. [1] Latapy, Matthieu, Clémence Magnien, and Nathalie Del Vecchio (2008).\n       Basic notions for the analysis of large two-mode networks.\n       Social Networks 30(1), 31--48.\n\n    "
    if nodes is None:
        nodes = G
    if any((len(G[v]) < 2 for v in nodes)):
        raise NetworkXError('Cannot compute redundancy coefficient for a node that has fewer than two neighbors.')
    return {v: _node_redundancy(G, v) for v in nodes}

def _node_redundancy(G, v):
    if False:
        return 10
    'Returns the redundancy of the node `v` in the bipartite graph `G`.\n\n    If `G` is a graph with `n` nodes, the redundancy of a node is the ratio\n    of the "overlap" of `v` to the maximum possible overlap of `v`\n    according to its degree. The overlap of `v` is the number of pairs of\n    neighbors that have mutual neighbors themselves, other than `v`.\n\n    `v` must have at least two neighbors in `G`.\n\n    '
    n = len(G[v])
    overlap = sum((1 for (u, w) in combinations(G[v], 2) if (set(G[u]) & set(G[w])) - {v}))
    return 2 * overlap / (n * (n - 1))