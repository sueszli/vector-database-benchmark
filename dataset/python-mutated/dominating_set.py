"""Functions for finding node and edge dominating sets.

A `dominating set`_ for an undirected graph *G* with vertex set *V*
and edge set *E* is a subset *D* of *V* such that every vertex not in
*D* is adjacent to at least one member of *D*. An `edge dominating set`_
is a subset *F* of *E* such that every edge not in *F* is
incident to an endpoint of at least one edge in *F*.

.. _dominating set: https://en.wikipedia.org/wiki/Dominating_set
.. _edge dominating set: https://en.wikipedia.org/wiki/Edge_dominating_set

"""
import networkx as nx
from ...utils import not_implemented_for
from ..matching import maximal_matching
__all__ = ['min_weighted_dominating_set', 'min_edge_dominating_set']

@not_implemented_for('directed')
@nx._dispatch(node_attrs='weight')
def min_weighted_dominating_set(G, weight=None):
    if False:
        while True:
            i = 10
    'Returns a dominating set that approximates the minimum weight node\n    dominating set.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n        Undirected graph.\n\n    weight : string\n        The node attribute storing the weight of an node. If provided,\n        the node attribute with this key must be a number for each\n        node. If not provided, each node is assumed to have weight one.\n\n    Returns\n    -------\n    min_weight_dominating_set : set\n        A set of nodes, the sum of whose weights is no more than `(\\log\n        w(V)) w(V^*)`, where `w(V)` denotes the sum of the weights of\n        each node in the graph and `w(V^*)` denotes the sum of the\n        weights of each node in the minimum weight dominating set.\n\n    Notes\n    -----\n    This algorithm computes an approximate minimum weighted dominating\n    set for the graph `G`. The returned solution has weight `(\\log\n    w(V)) w(V^*)`, where `w(V)` denotes the sum of the weights of each\n    node in the graph and `w(V^*)` denotes the sum of the weights of\n    each node in the minimum weight dominating set for the graph.\n\n    This implementation of the algorithm runs in $O(m)$ time, where $m$\n    is the number of edges in the graph.\n\n    References\n    ----------\n    .. [1] Vazirani, Vijay V.\n           *Approximation Algorithms*.\n           Springer Science & Business Media, 2001.\n\n    '
    if len(G) == 0:
        return set()
    dom_set = set()

    def _cost(node_and_neighborhood):
        if False:
            for i in range(10):
                print('nop')
        'Returns the cost-effectiveness of greedily choosing the given\n        node.\n\n        `node_and_neighborhood` is a two-tuple comprising a node and its\n        closed neighborhood.\n\n        '
        (v, neighborhood) = node_and_neighborhood
        return G.nodes[v].get(weight, 1) / len(neighborhood - dom_set)
    vertices = set(G)
    neighborhoods = {v: {v} | set(G[v]) for v in G}
    while vertices:
        (dom_node, min_set) = min(neighborhoods.items(), key=_cost)
        dom_set.add(dom_node)
        del neighborhoods[dom_node]
        vertices -= min_set
    return dom_set

@nx._dispatch
def min_edge_dominating_set(G):
    if False:
        while True:
            i = 10
    'Returns minimum cardinality edge dominating set.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n      Undirected graph\n\n    Returns\n    -------\n    min_edge_dominating_set : set\n      Returns a set of dominating edges whose size is no more than 2 * OPT.\n\n    Notes\n    -----\n    The algorithm computes an approximate solution to the edge dominating set\n    problem. The result is no more than 2 * OPT in terms of size of the set.\n    Runtime of the algorithm is $O(|E|)$.\n    '
    if not G:
        raise ValueError('Expected non-empty NetworkX graph!')
    return maximal_matching(G)