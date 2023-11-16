"""Connected components."""
import networkx as nx
from networkx.utils.decorators import not_implemented_for
from ...utils import arbitrary_element
__all__ = ['number_connected_components', 'connected_components', 'is_connected', 'node_connected_component']

@not_implemented_for('directed')
@nx._dispatch
def connected_components(G):
    if False:
        i = 10
        return i + 15
    "Generate connected components.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n       An undirected graph\n\n    Returns\n    -------\n    comp : generator of sets\n       A generator of sets of nodes, one for each component of G.\n\n    Raises\n    ------\n    NetworkXNotImplemented\n        If G is directed.\n\n    Examples\n    --------\n    Generate a sorted list of connected components, largest first.\n\n    >>> G = nx.path_graph(4)\n    >>> nx.add_path(G, [10, 11, 12])\n    >>> [len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)]\n    [4, 3]\n\n    If you only want the largest connected component, it's more\n    efficient to use max instead of sort.\n\n    >>> largest_cc = max(nx.connected_components(G), key=len)\n\n    To create the induced subgraph of each component use:\n\n    >>> S = [G.subgraph(c).copy() for c in nx.connected_components(G)]\n\n    See Also\n    --------\n    strongly_connected_components\n    weakly_connected_components\n\n    Notes\n    -----\n    For undirected graphs only.\n\n    "
    seen = set()
    for v in G:
        if v not in seen:
            c = _plain_bfs(G, v)
            seen.update(c)
            yield c

@not_implemented_for('directed')
@nx._dispatch
def number_connected_components(G):
    if False:
        for i in range(10):
            print('nop')
    'Returns the number of connected components.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n       An undirected graph.\n\n    Returns\n    -------\n    n : integer\n       Number of connected components\n\n    Raises\n    ------\n    NetworkXNotImplemented\n        If G is directed.\n\n    Examples\n    --------\n    >>> G = nx.Graph([(0, 1), (1, 2), (5, 6), (3, 4)])\n    >>> nx.number_connected_components(G)\n    3\n\n    See Also\n    --------\n    connected_components\n    number_weakly_connected_components\n    number_strongly_connected_components\n\n    Notes\n    -----\n    For undirected graphs only.\n\n    '
    return sum((1 for cc in connected_components(G)))

@not_implemented_for('directed')
@nx._dispatch
def is_connected(G):
    if False:
        for i in range(10):
            print('nop')
    'Returns True if the graph is connected, False otherwise.\n\n    Parameters\n    ----------\n    G : NetworkX Graph\n       An undirected graph.\n\n    Returns\n    -------\n    connected : bool\n      True if the graph is connected, false otherwise.\n\n    Raises\n    ------\n    NetworkXNotImplemented\n        If G is directed.\n\n    Examples\n    --------\n    >>> G = nx.path_graph(4)\n    >>> print(nx.is_connected(G))\n    True\n\n    See Also\n    --------\n    is_strongly_connected\n    is_weakly_connected\n    is_semiconnected\n    is_biconnected\n    connected_components\n\n    Notes\n    -----\n    For undirected graphs only.\n\n    '
    if len(G) == 0:
        raise nx.NetworkXPointlessConcept('Connectivity is undefined for the null graph.')
    return sum((1 for node in _plain_bfs(G, arbitrary_element(G)))) == len(G)

@not_implemented_for('directed')
@nx._dispatch
def node_connected_component(G, n):
    if False:
        return 10
    'Returns the set of nodes in the component of graph containing node n.\n\n    Parameters\n    ----------\n    G : NetworkX Graph\n       An undirected graph.\n\n    n : node label\n       A node in G\n\n    Returns\n    -------\n    comp : set\n       A set of nodes in the component of G containing node n.\n\n    Raises\n    ------\n    NetworkXNotImplemented\n        If G is directed.\n\n    Examples\n    --------\n    >>> G = nx.Graph([(0, 1), (1, 2), (5, 6), (3, 4)])\n    >>> nx.node_connected_component(G, 0)  # nodes of component that contains node 0\n    {0, 1, 2}\n\n    See Also\n    --------\n    connected_components\n\n    Notes\n    -----\n    For undirected graphs only.\n\n    '
    return _plain_bfs(G, n)

def _plain_bfs(G, source):
    if False:
        i = 10
        return i + 15
    'A fast BFS node generator'
    adj = G._adj
    n = len(adj)
    seen = {source}
    nextlevel = [source]
    while nextlevel:
        thislevel = nextlevel
        nextlevel = []
        for v in thislevel:
            for w in adj[v]:
                if w not in seen:
                    seen.add(w)
                    nextlevel.append(w)
            if len(seen) == n:
                return seen
    return seen