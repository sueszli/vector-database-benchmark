"""Functions for computing the Kernighan–Lin bipartition algorithm."""
from itertools import count
import networkx as nx
from networkx.algorithms.community.community_utils import is_partition
from networkx.utils import BinaryHeap, not_implemented_for, py_random_state
__all__ = ['kernighan_lin_bisection']

def _kernighan_lin_sweep(edges, side):
    if False:
        print('Hello World!')
    '\n    This is a modified form of Kernighan-Lin, which moves single nodes at a\n    time, alternating between sides to keep the bisection balanced.  We keep\n    two min-heaps of swap costs to make optimal-next-move selection fast.\n    '
    (costs0, costs1) = costs = (BinaryHeap(), BinaryHeap())
    for (u, side_u, edges_u) in zip(count(), side, edges):
        cost_u = sum((w if side[v] else -w for (v, w) in edges_u))
        costs[side_u].insert(u, cost_u if side_u else -cost_u)

    def _update_costs(costs_x, x):
        if False:
            return 10
        for (y, w) in edges[x]:
            costs_y = costs[side[y]]
            cost_y = costs_y.get(y)
            if cost_y is not None:
                cost_y += 2 * (-w if costs_x is costs_y else w)
                costs_y.insert(y, cost_y, True)
    i = 0
    totcost = 0
    while costs0 and costs1:
        (u, cost_u) = costs0.pop()
        _update_costs(costs0, u)
        (v, cost_v) = costs1.pop()
        _update_costs(costs1, v)
        totcost += cost_u + cost_v
        i += 1
        yield (totcost, i, (u, v))

@not_implemented_for('directed')
@py_random_state(4)
@nx._dispatch(edge_attrs='weight')
def kernighan_lin_bisection(G, partition=None, max_iter=10, weight='weight', seed=None):
    if False:
        while True:
            i = 10
    'Partition a graph into two blocks using the Kernighan–Lin\n    algorithm.\n\n    This algorithm partitions a network into two sets by iteratively\n    swapping pairs of nodes to reduce the edge cut between the two sets.  The\n    pairs are chosen according to a modified form of Kernighan-Lin [1]_, which\n    moves node individually, alternating between sides to keep the bisection\n    balanced.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n        Graph must be undirected.\n\n    partition : tuple\n        Pair of iterables containing an initial partition. If not\n        specified, a random balanced partition is used.\n\n    max_iter : int\n        Maximum number of times to attempt swaps to find an\n        improvement before giving up.\n\n    weight : key\n        Edge data key to use as weight. If None, the weights are all\n        set to one.\n\n    seed : integer, random_state, or None (default)\n        Indicator of random number generation state.\n        See :ref:`Randomness<randomness>`.\n        Only used if partition is None\n\n    Returns\n    -------\n    partition : tuple\n        A pair of sets of nodes representing the bipartition.\n\n    Raises\n    ------\n    NetworkXError\n        If partition is not a valid partition of the nodes of the graph.\n\n    References\n    ----------\n    .. [1] Kernighan, B. W.; Lin, Shen (1970).\n       "An efficient heuristic procedure for partitioning graphs."\n       *Bell Systems Technical Journal* 49: 291--307.\n       Oxford University Press 2011.\n\n    '
    n = len(G)
    labels = list(G)
    seed.shuffle(labels)
    index = {v: i for (i, v) in enumerate(labels)}
    if partition is None:
        side = [0] * (n // 2) + [1] * ((n + 1) // 2)
    else:
        try:
            (A, B) = partition
        except (TypeError, ValueError) as err:
            raise nx.NetworkXError('partition must be two sets') from err
        if not is_partition(G, (A, B)):
            raise nx.NetworkXError('partition invalid')
        side = [0] * n
        for a in A:
            side[index[a]] = 1
    if G.is_multigraph():
        edges = [[(index[u], sum((e.get(weight, 1) for e in d.values()))) for (u, d) in G[v].items()] for v in labels]
    else:
        edges = [[(index[u], e.get(weight, 1)) for (u, e) in G[v].items()] for v in labels]
    for i in range(max_iter):
        costs = list(_kernighan_lin_sweep(edges, side))
        (min_cost, min_i, _) = min(costs)
        if min_cost >= 0:
            break
        for (_, _, (u, v)) in costs[:min_i]:
            side[u] = 1
            side[v] = 0
    A = {u for (u, s) in zip(labels, side) if s == 0}
    B = {u for (u, s) in zip(labels, side) if s == 1}
    return (A, B)