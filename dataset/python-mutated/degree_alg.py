"""Degree centrality measures."""
import networkx as nx
from networkx.utils.decorators import not_implemented_for
__all__ = ['degree_centrality', 'in_degree_centrality', 'out_degree_centrality']

@nx._dispatch
def degree_centrality(G):
    if False:
        for i in range(10):
            print('nop')
    'Compute the degree centrality for nodes.\n\n    The degree centrality for a node v is the fraction of nodes it\n    is connected to.\n\n    Parameters\n    ----------\n    G : graph\n      A networkx graph\n\n    Returns\n    -------\n    nodes : dictionary\n       Dictionary of nodes with degree centrality as the value.\n\n    Examples\n    --------\n    >>> G = nx.Graph([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3)])\n    >>> nx.degree_centrality(G)\n    {0: 1.0, 1: 1.0, 2: 0.6666666666666666, 3: 0.6666666666666666}\n\n    See Also\n    --------\n    betweenness_centrality, load_centrality, eigenvector_centrality\n\n    Notes\n    -----\n    The degree centrality values are normalized by dividing by the maximum\n    possible degree in a simple graph n-1 where n is the number of nodes in G.\n\n    For multigraphs or graphs with self loops the maximum degree might\n    be higher than n-1 and values of degree centrality greater than 1\n    are possible.\n    '
    if len(G) <= 1:
        return {n: 1 for n in G}
    s = 1.0 / (len(G) - 1.0)
    centrality = {n: d * s for (n, d) in G.degree()}
    return centrality

@not_implemented_for('undirected')
@nx._dispatch
def in_degree_centrality(G):
    if False:
        i = 10
        return i + 15
    'Compute the in-degree centrality for nodes.\n\n    The in-degree centrality for a node v is the fraction of nodes its\n    incoming edges are connected to.\n\n    Parameters\n    ----------\n    G : graph\n        A NetworkX graph\n\n    Returns\n    -------\n    nodes : dictionary\n        Dictionary of nodes with in-degree centrality as values.\n\n    Raises\n    ------\n    NetworkXNotImplemented\n        If G is undirected.\n\n    Examples\n    --------\n    >>> G = nx.DiGraph([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3)])\n    >>> nx.in_degree_centrality(G)\n    {0: 0.0, 1: 0.3333333333333333, 2: 0.6666666666666666, 3: 0.6666666666666666}\n\n    See Also\n    --------\n    degree_centrality, out_degree_centrality\n\n    Notes\n    -----\n    The degree centrality values are normalized by dividing by the maximum\n    possible degree in a simple graph n-1 where n is the number of nodes in G.\n\n    For multigraphs or graphs with self loops the maximum degree might\n    be higher than n-1 and values of degree centrality greater than 1\n    are possible.\n    '
    if len(G) <= 1:
        return {n: 1 for n in G}
    s = 1.0 / (len(G) - 1.0)
    centrality = {n: d * s for (n, d) in G.in_degree()}
    return centrality

@not_implemented_for('undirected')
@nx._dispatch
def out_degree_centrality(G):
    if False:
        i = 10
        return i + 15
    'Compute the out-degree centrality for nodes.\n\n    The out-degree centrality for a node v is the fraction of nodes its\n    outgoing edges are connected to.\n\n    Parameters\n    ----------\n    G : graph\n        A NetworkX graph\n\n    Returns\n    -------\n    nodes : dictionary\n        Dictionary of nodes with out-degree centrality as values.\n\n    Raises\n    ------\n    NetworkXNotImplemented\n        If G is undirected.\n\n    Examples\n    --------\n    >>> G = nx.DiGraph([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3)])\n    >>> nx.out_degree_centrality(G)\n    {0: 1.0, 1: 0.6666666666666666, 2: 0.0, 3: 0.0}\n\n    See Also\n    --------\n    degree_centrality, in_degree_centrality\n\n    Notes\n    -----\n    The degree centrality values are normalized by dividing by the maximum\n    possible degree in a simple graph n-1 where n is the number of nodes in G.\n\n    For multigraphs or graphs with self loops the maximum degree might\n    be higher than n-1 and values of degree centrality greater than 1\n    are possible.\n    '
    if len(G) <= 1:
        return {n: 1 for n in G}
    s = 1.0 / (len(G) - 1.0)
    centrality = {n: d * s for (n, d) in G.out_degree()}
    return centrality