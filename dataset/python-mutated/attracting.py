"""Attracting components."""
import networkx as nx
from networkx.utils.decorators import not_implemented_for
__all__ = ['number_attracting_components', 'attracting_components', 'is_attracting_component']

@not_implemented_for('undirected')
@nx._dispatch
def attracting_components(G):
    if False:
        i = 10
        return i + 15
    'Generates the attracting components in `G`.\n\n    An attracting component in a directed graph `G` is a strongly connected\n    component with the property that a random walker on the graph will never\n    leave the component, once it enters the component.\n\n    The nodes in attracting components can also be thought of as recurrent\n    nodes.  If a random walker enters the attractor containing the node, then\n    the node will be visited infinitely often.\n\n    To obtain induced subgraphs on each component use:\n    ``(G.subgraph(c).copy() for c in attracting_components(G))``\n\n    Parameters\n    ----------\n    G : DiGraph, MultiDiGraph\n        The graph to be analyzed.\n\n    Returns\n    -------\n    attractors : generator of sets\n        A generator of sets of nodes, one for each attracting component of G.\n\n    Raises\n    ------\n    NetworkXNotImplemented\n        If the input graph is undirected.\n\n    See Also\n    --------\n    number_attracting_components\n    is_attracting_component\n\n    '
    scc = list(nx.strongly_connected_components(G))
    cG = nx.condensation(G, scc)
    for n in cG:
        if cG.out_degree(n) == 0:
            yield scc[n]

@not_implemented_for('undirected')
@nx._dispatch
def number_attracting_components(G):
    if False:
        while True:
            i = 10
    'Returns the number of attracting components in `G`.\n\n    Parameters\n    ----------\n    G : DiGraph, MultiDiGraph\n        The graph to be analyzed.\n\n    Returns\n    -------\n    n : int\n        The number of attracting components in G.\n\n    Raises\n    ------\n    NetworkXNotImplemented\n        If the input graph is undirected.\n\n    See Also\n    --------\n    attracting_components\n    is_attracting_component\n\n    '
    return sum((1 for ac in attracting_components(G)))

@not_implemented_for('undirected')
@nx._dispatch
def is_attracting_component(G):
    if False:
        return 10
    'Returns True if `G` consists of a single attracting component.\n\n    Parameters\n    ----------\n    G : DiGraph, MultiDiGraph\n        The graph to be analyzed.\n\n    Returns\n    -------\n    attracting : bool\n        True if `G` has a single attracting component. Otherwise, False.\n\n    Raises\n    ------\n    NetworkXNotImplemented\n        If the input graph is undirected.\n\n    See Also\n    --------\n    attracting_components\n    number_attracting_components\n\n    '
    ac = list(attracting_components(G))
    if len(ac) == 1:
        return len(ac[0]) == len(G)
    return False