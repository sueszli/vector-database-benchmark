"""Function for computing walks in a graph.
"""
import networkx as nx
__all__ = ['number_of_walks']

@nx._dispatch
def number_of_walks(G, walk_length):
    if False:
        for i in range(10):
            print('nop')
    'Returns the number of walks connecting each pair of nodes in `G`\n\n    A *walk* is a sequence of nodes in which each adjacent pair of nodes\n    in the sequence is adjacent in the graph. A walk can repeat the same\n    edge and go in the opposite direction just as people can walk on a\n    set of paths, but standing still is not counted as part of the walk.\n\n    This function only counts the walks with `walk_length` edges. Note that\n    the number of nodes in the walk sequence is one more than `walk_length`.\n    The number of walks can grow very quickly on a larger graph\n    and with a larger walk length.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n\n    walk_length : int\n        A nonnegative integer representing the length of a walk.\n\n    Returns\n    -------\n    dict\n        A dictionary of dictionaries in which outer keys are source\n        nodes, inner keys are target nodes, and inner values are the\n        number of walks of length `walk_length` connecting those nodes.\n\n    Raises\n    ------\n    ValueError\n        If `walk_length` is negative\n\n    Examples\n    --------\n\n    >>> G = nx.Graph([(0, 1), (1, 2)])\n    >>> walks = nx.number_of_walks(G, 2)\n    >>> walks\n    {0: {0: 1, 1: 0, 2: 1}, 1: {0: 0, 1: 2, 2: 0}, 2: {0: 1, 1: 0, 2: 1}}\n    >>> total_walks = sum(sum(tgts.values()) for _, tgts in walks.items())\n\n    You can also get the number of walks from a specific source node using the\n    returned dictionary. For example, number of walks of length 1 from node 0\n    can be found as follows:\n\n    >>> walks = nx.number_of_walks(G, 1)\n    >>> walks[0]\n    {0: 0, 1: 1, 2: 0}\n    >>> sum(walks[0].values())  # walks from 0 of length 1\n    1\n\n    Similarly, a target node can also be specified:\n\n    >>> walks[0][1]\n    1\n\n    '
    import numpy as np
    if walk_length < 0:
        raise ValueError(f'`walk_length` cannot be negative: {walk_length}')
    A = nx.adjacency_matrix(G, weight=None)
    power = np.linalg.matrix_power(A.toarray(), walk_length)
    result = {u: {v: power[u_idx, v_idx] for (v_idx, v) in enumerate(G)} for (u_idx, u) in enumerate(G)}
    return result