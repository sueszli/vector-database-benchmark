"""Weakly connected components."""
import networkx as nx
from networkx.utils.decorators import not_implemented_for
__all__ = ['number_weakly_connected_components', 'weakly_connected_components', 'is_weakly_connected']

@not_implemented_for('undirected')
@nx._dispatch
def weakly_connected_components(G):
    if False:
        while True:
            i = 10
    "Generate weakly connected components of G.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n        A directed graph\n\n    Returns\n    -------\n    comp : generator of sets\n        A generator of sets of nodes, one for each weakly connected\n        component of G.\n\n    Raises\n    ------\n    NetworkXNotImplemented\n        If G is undirected.\n\n    Examples\n    --------\n    Generate a sorted list of weakly connected components, largest first.\n\n    >>> G = nx.path_graph(4, create_using=nx.DiGraph())\n    >>> nx.add_path(G, [10, 11, 12])\n    >>> [\n    ...     len(c)\n    ...     for c in sorted(nx.weakly_connected_components(G), key=len, reverse=True)\n    ... ]\n    [4, 3]\n\n    If you only want the largest component, it's more efficient to\n    use max instead of sort:\n\n    >>> largest_cc = max(nx.weakly_connected_components(G), key=len)\n\n    See Also\n    --------\n    connected_components\n    strongly_connected_components\n\n    Notes\n    -----\n    For directed graphs only.\n\n    "
    seen = set()
    for v in G:
        if v not in seen:
            c = set(_plain_bfs(G, v))
            seen.update(c)
            yield c

@not_implemented_for('undirected')
@nx._dispatch
def number_weakly_connected_components(G):
    if False:
        while True:
            i = 10
    'Returns the number of weakly connected components in G.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n        A directed graph.\n\n    Returns\n    -------\n    n : integer\n        Number of weakly connected components\n\n    Raises\n    ------\n    NetworkXNotImplemented\n        If G is undirected.\n\n    Examples\n    --------\n    >>> G = nx.DiGraph([(0, 1), (2, 1), (3, 4)])\n    >>> nx.number_weakly_connected_components(G)\n    2\n\n    See Also\n    --------\n    weakly_connected_components\n    number_connected_components\n    number_strongly_connected_components\n\n    Notes\n    -----\n    For directed graphs only.\n\n    '
    return sum((1 for wcc in weakly_connected_components(G)))

@not_implemented_for('undirected')
@nx._dispatch
def is_weakly_connected(G):
    if False:
        print('Hello World!')
    'Test directed graph for weak connectivity.\n\n    A directed graph is weakly connected if and only if the graph\n    is connected when the direction of the edge between nodes is ignored.\n\n    Note that if a graph is strongly connected (i.e. the graph is connected\n    even when we account for directionality), it is by definition weakly\n    connected as well.\n\n    Parameters\n    ----------\n    G : NetworkX Graph\n        A directed graph.\n\n    Returns\n    -------\n    connected : bool\n        True if the graph is weakly connected, False otherwise.\n\n    Raises\n    ------\n    NetworkXNotImplemented\n        If G is undirected.\n\n    Examples\n    --------\n    >>> G = nx.DiGraph([(0, 1), (2, 1)])\n    >>> G.add_node(3)\n    >>> nx.is_weakly_connected(G)  # node 3 is not connected to the graph\n    False\n    >>> G.add_edge(2, 3)\n    >>> nx.is_weakly_connected(G)\n    True\n\n    See Also\n    --------\n    is_strongly_connected\n    is_semiconnected\n    is_connected\n    is_biconnected\n    weakly_connected_components\n\n    Notes\n    -----\n    For directed graphs only.\n\n    '
    if len(G) == 0:
        raise nx.NetworkXPointlessConcept('Connectivity is undefined for the null graph.')
    return len(next(weakly_connected_components(G))) == len(G)

def _plain_bfs(G, source):
    if False:
        print('Hello World!')
    'A fast BFS node generator\n\n    The direction of the edge between nodes is ignored.\n\n    For directed graphs only.\n\n    '
    n = len(G)
    Gsucc = G._succ
    Gpred = G._pred
    seen = {source}
    nextlevel = [source]
    yield source
    while nextlevel:
        thislevel = nextlevel
        nextlevel = []
        for v in thislevel:
            for w in Gsucc[v]:
                if w not in seen:
                    seen.add(w)
                    nextlevel.append(w)
                    yield w
            for w in Gpred[v]:
                if w not in seen:
                    seen.add(w)
                    nextlevel.append(w)
                    yield w
            if len(seen) == n:
                return