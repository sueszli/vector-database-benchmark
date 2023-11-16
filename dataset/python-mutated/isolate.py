"""
Functions for identifying isolate (degree zero) nodes.
"""
import networkx as nx
__all__ = ['is_isolate', 'isolates', 'number_of_isolates']

@nx._dispatch
def is_isolate(G, n):
    if False:
        i = 10
        return i + 15
    'Determines whether a node is an isolate.\n\n    An *isolate* is a node with no neighbors (that is, with degree\n    zero). For directed graphs, this means no in-neighbors and no\n    out-neighbors.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n\n    n : node\n        A node in `G`.\n\n    Returns\n    -------\n    is_isolate : bool\n       True if and only if `n` has no neighbors.\n\n    Examples\n    --------\n    >>> G = nx.Graph()\n    >>> G.add_edge(1, 2)\n    >>> G.add_node(3)\n    >>> nx.is_isolate(G, 2)\n    False\n    >>> nx.is_isolate(G, 3)\n    True\n    '
    return G.degree(n) == 0

@nx._dispatch
def isolates(G):
    if False:
        print('Hello World!')
    'Iterator over isolates in the graph.\n\n    An *isolate* is a node with no neighbors (that is, with degree\n    zero). For directed graphs, this means no in-neighbors and no\n    out-neighbors.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n\n    Returns\n    -------\n    iterator\n        An iterator over the isolates of `G`.\n\n    Examples\n    --------\n    To get a list of all isolates of a graph, use the :class:`list`\n    constructor::\n\n        >>> G = nx.Graph()\n        >>> G.add_edge(1, 2)\n        >>> G.add_node(3)\n        >>> list(nx.isolates(G))\n        [3]\n\n    To remove all isolates in the graph, first create a list of the\n    isolates, then use :meth:`Graph.remove_nodes_from`::\n\n        >>> G.remove_nodes_from(list(nx.isolates(G)))\n        >>> list(G)\n        [1, 2]\n\n    For digraphs, isolates have zero in-degree and zero out_degre::\n\n        >>> G = nx.DiGraph([(0, 1), (1, 2)])\n        >>> G.add_node(3)\n        >>> list(nx.isolates(G))\n        [3]\n\n    '
    return (n for (n, d) in G.degree() if d == 0)

@nx._dispatch
def number_of_isolates(G):
    if False:
        print('Hello World!')
    'Returns the number of isolates in the graph.\n\n    An *isolate* is a node with no neighbors (that is, with degree\n    zero). For directed graphs, this means no in-neighbors and no\n    out-neighbors.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n\n    Returns\n    -------\n    int\n        The number of degree zero nodes in the graph `G`.\n\n    '
    return sum((1 for v in isolates(G)))