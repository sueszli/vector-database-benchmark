"""Algorithms to calculate reciprocity in a directed graph."""
import networkx as nx
from networkx import NetworkXError
from ..utils import not_implemented_for
__all__ = ['reciprocity', 'overall_reciprocity']

@not_implemented_for('undirected', 'multigraph')
@nx._dispatch
def reciprocity(G, nodes=None):
    if False:
        while True:
            i = 10
    'Compute the reciprocity in a directed graph.\n\n    The reciprocity of a directed graph is defined as the ratio\n    of the number of edges pointing in both directions to the total\n    number of edges in the graph.\n    Formally, $r = |{(u,v) \\in G|(v,u) \\in G}| / |{(u,v) \\in G}|$.\n\n    The reciprocity of a single node u is defined similarly,\n    it is the ratio of the number of edges in both directions to\n    the total number of edges attached to node u.\n\n    Parameters\n    ----------\n    G : graph\n       A networkx directed graph\n    nodes : container of nodes, optional (default=whole graph)\n       Compute reciprocity for nodes in this container.\n\n    Returns\n    -------\n    out : dictionary\n       Reciprocity keyed by node label.\n\n    Notes\n    -----\n    The reciprocity is not defined for isolated nodes.\n    In such cases this function will return None.\n\n    '
    if nodes is None:
        return overall_reciprocity(G)
    if nodes in G:
        reciprocity = next(_reciprocity_iter(G, nodes))[1]
        if reciprocity is None:
            raise NetworkXError('Not defined for isolated nodes.')
        else:
            return reciprocity
    return dict(_reciprocity_iter(G, nodes))

def _reciprocity_iter(G, nodes):
    if False:
        while True:
            i = 10
    'Return an iterator of (node, reciprocity).'
    n = G.nbunch_iter(nodes)
    for node in n:
        pred = set(G.predecessors(node))
        succ = set(G.successors(node))
        overlap = pred & succ
        n_total = len(pred) + len(succ)
        if n_total == 0:
            yield (node, None)
        else:
            reciprocity = 2 * len(overlap) / n_total
            yield (node, reciprocity)

@not_implemented_for('undirected', 'multigraph')
@nx._dispatch
def overall_reciprocity(G):
    if False:
        for i in range(10):
            print('nop')
    'Compute the reciprocity for the whole graph.\n\n    See the doc of reciprocity for the definition.\n\n    Parameters\n    ----------\n    G : graph\n       A networkx graph\n\n    '
    n_all_edge = G.number_of_edges()
    n_overlap_edge = (n_all_edge - G.to_undirected().number_of_edges()) * 2
    if n_all_edge == 0:
        raise NetworkXError('Not defined for empty graphs')
    return n_overlap_edge / n_all_edge