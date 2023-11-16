"""
Dominance algorithms.
"""
from functools import reduce
import networkx as nx
from networkx.utils import not_implemented_for
__all__ = ['immediate_dominators', 'dominance_frontiers']

@not_implemented_for('undirected')
@nx._dispatch
def immediate_dominators(G, start):
    if False:
        i = 10
        return i + 15
    'Returns the immediate dominators of all nodes of a directed graph.\n\n    Parameters\n    ----------\n    G : a DiGraph or MultiDiGraph\n        The graph where dominance is to be computed.\n\n    start : node\n        The start node of dominance computation.\n\n    Returns\n    -------\n    idom : dict keyed by nodes\n        A dict containing the immediate dominators of each node reachable from\n        `start`.\n\n    Raises\n    ------\n    NetworkXNotImplemented\n        If `G` is undirected.\n\n    NetworkXError\n        If `start` is not in `G`.\n\n    Notes\n    -----\n    Except for `start`, the immediate dominators are the parents of their\n    corresponding nodes in the dominator tree.\n\n    Examples\n    --------\n    >>> G = nx.DiGraph([(1, 2), (1, 3), (2, 5), (3, 4), (4, 5)])\n    >>> sorted(nx.immediate_dominators(G, 1).items())\n    [(1, 1), (2, 1), (3, 1), (4, 3), (5, 1)]\n\n    References\n    ----------\n    .. [1] K. D. Cooper, T. J. Harvey, and K. Kennedy.\n           A simple, fast dominance algorithm.\n           Software Practice & Experience, 4:110, 2001.\n    '
    if start not in G:
        raise nx.NetworkXError('start is not in G')
    idom = {start: start}
    order = list(nx.dfs_postorder_nodes(G, start))
    dfn = {u: i for (i, u) in enumerate(order)}
    order.pop()
    order.reverse()

    def intersect(u, v):
        if False:
            for i in range(10):
                print('nop')
        while u != v:
            while dfn[u] < dfn[v]:
                u = idom[u]
            while dfn[u] > dfn[v]:
                v = idom[v]
        return u
    changed = True
    while changed:
        changed = False
        for u in order:
            new_idom = reduce(intersect, (v for v in G.pred[u] if v in idom))
            if u not in idom or idom[u] != new_idom:
                idom[u] = new_idom
                changed = True
    return idom

@nx._dispatch
def dominance_frontiers(G, start):
    if False:
        return 10
    'Returns the dominance frontiers of all nodes of a directed graph.\n\n    Parameters\n    ----------\n    G : a DiGraph or MultiDiGraph\n        The graph where dominance is to be computed.\n\n    start : node\n        The start node of dominance computation.\n\n    Returns\n    -------\n    df : dict keyed by nodes\n        A dict containing the dominance frontiers of each node reachable from\n        `start` as lists.\n\n    Raises\n    ------\n    NetworkXNotImplemented\n        If `G` is undirected.\n\n    NetworkXError\n        If `start` is not in `G`.\n\n    Examples\n    --------\n    >>> G = nx.DiGraph([(1, 2), (1, 3), (2, 5), (3, 4), (4, 5)])\n    >>> sorted((u, sorted(df)) for u, df in nx.dominance_frontiers(G, 1).items())\n    [(1, []), (2, [5]), (3, [5]), (4, [5]), (5, [])]\n\n    References\n    ----------\n    .. [1] K. D. Cooper, T. J. Harvey, and K. Kennedy.\n           A simple, fast dominance algorithm.\n           Software Practice & Experience, 4:110, 2001.\n    '
    idom = nx.immediate_dominators(G, start)
    df = {u: set() for u in idom}
    for u in idom:
        if len(G.pred[u]) >= 2:
            for v in G.pred[u]:
                if v in idom:
                    while v != idom[u]:
                        df[v].add(u)
                        v = idom[v]
    return df