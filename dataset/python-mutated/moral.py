"""Function for computing the moral graph of a directed graph."""
import itertools
import networkx as nx
from networkx.utils import not_implemented_for
__all__ = ['moral_graph']

@not_implemented_for('undirected')
@nx._dispatch
def moral_graph(G):
    if False:
        for i in range(10):
            print('nop')
    "Return the Moral Graph\n\n    Returns the moralized graph of a given directed graph.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n        Directed graph\n\n    Returns\n    -------\n    H : NetworkX graph\n        The undirected moralized graph of G\n\n    Raises\n    ------\n    NetworkXNotImplemented\n        If `G` is undirected.\n\n    Examples\n    --------\n    >>> G = nx.DiGraph([(1, 2), (2, 3), (2, 5), (3, 4), (4, 3)])\n    >>> G_moral = nx.moral_graph(G)\n    >>> G_moral.edges()\n    EdgeView([(1, 2), (2, 3), (2, 5), (2, 4), (3, 4)])\n\n    Notes\n    -----\n    A moral graph is an undirected graph H = (V, E) generated from a\n    directed Graph, where if a node has more than one parent node, edges\n    between these parent nodes are inserted and all directed edges become\n    undirected.\n\n    https://en.wikipedia.org/wiki/Moral_graph\n\n    References\n    ----------\n    .. [1] Wray L. Buntine. 1995. Chain graphs for learning.\n           In Proceedings of the Eleventh conference on Uncertainty\n           in artificial intelligence (UAI'95)\n    "
    H = G.to_undirected()
    for preds in G.pred.values():
        predecessors_combinations = itertools.combinations(preds, r=2)
        H.add_edges_from(predecessors_combinations)
    return H