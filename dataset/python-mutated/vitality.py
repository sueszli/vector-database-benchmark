"""
Vitality measures.
"""
from functools import partial
import networkx as nx
__all__ = ['closeness_vitality']

@nx._dispatch(edge_attrs='weight')
def closeness_vitality(G, node=None, weight=None, wiener_index=None):
    if False:
        print('Hello World!')
    'Returns the closeness vitality for nodes in the graph.\n\n    The *closeness vitality* of a node, defined in Section 3.6.2 of [1],\n    is the change in the sum of distances between all node pairs when\n    excluding that node.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n        A strongly-connected graph.\n\n    weight : string\n        The name of the edge attribute used as weight. This is passed\n        directly to the :func:`~networkx.wiener_index` function.\n\n    node : object\n        If specified, only the closeness vitality for this node will be\n        returned. Otherwise, a dictionary mapping each node to its\n        closeness vitality will be returned.\n\n    Other parameters\n    ----------------\n    wiener_index : number\n        If you have already computed the Wiener index of the graph\n        `G`, you can provide that value here. Otherwise, it will be\n        computed for you.\n\n    Returns\n    -------\n    dictionary or float\n        If `node` is None, this function returns a dictionary\n        with nodes as keys and closeness vitality as the\n        value. Otherwise, it returns only the closeness vitality for the\n        specified `node`.\n\n        The closeness vitality of a node may be negative infinity if\n        removing that node would disconnect the graph.\n\n    Examples\n    --------\n    >>> G = nx.cycle_graph(3)\n    >>> nx.closeness_vitality(G)\n    {0: 2.0, 1: 2.0, 2: 2.0}\n\n    See Also\n    --------\n    closeness_centrality\n\n    References\n    ----------\n    .. [1] Ulrik Brandes, Thomas Erlebach (eds.).\n           *Network Analysis: Methodological Foundations*.\n           Springer, 2005.\n           <http://books.google.com/books?id=TTNhSm7HYrIC>\n\n    '
    if wiener_index is None:
        wiener_index = nx.wiener_index(G, weight=weight)
    if node is not None:
        after = nx.wiener_index(G.subgraph(set(G) - {node}), weight=weight)
        return wiener_index - after
    vitality = partial(closeness_vitality, G, weight=weight, wiener_index=wiener_index)
    return {v: vitality(node=v) for v in G}