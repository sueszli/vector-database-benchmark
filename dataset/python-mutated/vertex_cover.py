"""Functions for computing an approximate minimum weight vertex cover.

A |vertex cover|_ is a subset of nodes such that each edge in the graph
is incident to at least one node in the subset.

.. _vertex cover: https://en.wikipedia.org/wiki/Vertex_cover
.. |vertex cover| replace:: *vertex cover*

"""
import networkx as nx
__all__ = ['min_weighted_vertex_cover']

@nx._dispatch(node_attrs='weight')
def min_weighted_vertex_cover(G, weight=None):
    if False:
        while True:
            i = 10
    'Returns an approximate minimum weighted vertex cover.\n\n    The set of nodes returned by this function is guaranteed to be a\n    vertex cover, and the total weight of the set is guaranteed to be at\n    most twice the total weight of the minimum weight vertex cover. In\n    other words,\n\n    .. math::\n\n       w(S) \\leq 2 * w(S^*),\n\n    where $S$ is the vertex cover returned by this function,\n    $S^*$ is the vertex cover of minimum weight out of all vertex\n    covers of the graph, and $w$ is the function that computes the\n    sum of the weights of each node in that given set.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n\n    weight : string, optional (default = None)\n        If None, every node has weight 1. If a string, use this node\n        attribute as the node weight. A node without this attribute is\n        assumed to have weight 1.\n\n    Returns\n    -------\n    min_weighted_cover : set\n        Returns a set of nodes whose weight sum is no more than twice\n        the weight sum of the minimum weight vertex cover.\n\n    Notes\n    -----\n    For a directed graph, a vertex cover has the same definition: a set\n    of nodes such that each edge in the graph is incident to at least\n    one node in the set. Whether the node is the head or tail of the\n    directed edge is ignored.\n\n    This is the local-ratio algorithm for computing an approximate\n    vertex cover. The algorithm greedily reduces the costs over edges,\n    iteratively building a cover. The worst-case runtime of this\n    implementation is $O(m \\log n)$, where $n$ is the number\n    of nodes and $m$ the number of edges in the graph.\n\n    References\n    ----------\n    .. [1] Bar-Yehuda, R., and Even, S. (1985). "A local-ratio theorem for\n       approximating the weighted vertex cover problem."\n       *Annals of Discrete Mathematics*, 25, 27–46\n       <http://www.cs.technion.ac.il/~reuven/PDF/vc_lr.pdf>\n\n    '
    cost = dict(G.nodes(data=weight, default=1))
    cover = set()
    for (u, v) in G.edges():
        if u in cover or v in cover:
            continue
        if cost[u] <= cost[v]:
            cover.add(u)
            cost[v] -= cost[u]
        else:
            cover.add(v)
            cost[u] -= cost[v]
    return cover