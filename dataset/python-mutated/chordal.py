"""
Algorithms for chordal graphs.

A graph is chordal if every cycle of length at least 4 has a chord
(an edge joining two nodes not adjacent in the cycle).
https://en.wikipedia.org/wiki/Chordal_graph
"""
import sys
import networkx as nx
from networkx.algorithms.components import connected_components
from networkx.utils import arbitrary_element, not_implemented_for
__all__ = ['is_chordal', 'find_induced_nodes', 'chordal_graph_cliques', 'chordal_graph_treewidth', 'NetworkXTreewidthBoundExceeded', 'complete_to_chordal_graph']

class NetworkXTreewidthBoundExceeded(nx.NetworkXException):
    """Exception raised when a treewidth bound has been provided and it has
    been exceeded"""

@not_implemented_for('directed')
@not_implemented_for('multigraph')
@nx._dispatch
def is_chordal(G):
    if False:
        while True:
            i = 10
    'Checks whether G is a chordal graph.\n\n    A graph is chordal if every cycle of length at least 4 has a chord\n    (an edge joining two nodes not adjacent in the cycle).\n\n    Parameters\n    ----------\n    G : graph\n      A NetworkX graph.\n\n    Returns\n    -------\n    chordal : bool\n      True if G is a chordal graph and False otherwise.\n\n    Raises\n    ------\n    NetworkXNotImplemented\n        The algorithm does not support DiGraph, MultiGraph and MultiDiGraph.\n\n    Examples\n    --------\n    >>> e = [\n    ...     (1, 2),\n    ...     (1, 3),\n    ...     (2, 3),\n    ...     (2, 4),\n    ...     (3, 4),\n    ...     (3, 5),\n    ...     (3, 6),\n    ...     (4, 5),\n    ...     (4, 6),\n    ...     (5, 6),\n    ... ]\n    >>> G = nx.Graph(e)\n    >>> nx.is_chordal(G)\n    True\n\n    Notes\n    -----\n    The routine tries to go through every node following maximum cardinality\n    search. It returns False when it finds that the separator for any node\n    is not a clique.  Based on the algorithms in [1]_.\n\n    Self loops are ignored.\n\n    References\n    ----------\n    .. [1] R. E. Tarjan and M. Yannakakis, Simple linear-time algorithms\n       to test chordality of graphs, test acyclicity of hypergraphs, and\n       selectively reduce acyclic hypergraphs, SIAM J. Comput., 13 (1984),\n       pp. 566–579.\n    '
    if len(G.nodes) <= 3:
        return True
    return len(_find_chordality_breaker(G)) == 0

@nx._dispatch
def find_induced_nodes(G, s, t, treewidth_bound=sys.maxsize):
    if False:
        for i in range(10):
            print('nop')
    'Returns the set of induced nodes in the path from s to t.\n\n    Parameters\n    ----------\n    G : graph\n      A chordal NetworkX graph\n    s : node\n        Source node to look for induced nodes\n    t : node\n        Destination node to look for induced nodes\n    treewidth_bound: float\n        Maximum treewidth acceptable for the graph H. The search\n        for induced nodes will end as soon as the treewidth_bound is exceeded.\n\n    Returns\n    -------\n    induced_nodes : Set of nodes\n        The set of induced nodes in the path from s to t in G\n\n    Raises\n    ------\n    NetworkXError\n        The algorithm does not support DiGraph, MultiGraph and MultiDiGraph.\n        If the input graph is an instance of one of these classes, a\n        :exc:`NetworkXError` is raised.\n        The algorithm can only be applied to chordal graphs. If the input\n        graph is found to be non-chordal, a :exc:`NetworkXError` is raised.\n\n    Examples\n    --------\n    >>> G = nx.Graph()\n    >>> G = nx.generators.classic.path_graph(10)\n    >>> induced_nodes = nx.find_induced_nodes(G, 1, 9, 2)\n    >>> sorted(induced_nodes)\n    [1, 2, 3, 4, 5, 6, 7, 8, 9]\n\n    Notes\n    -----\n    G must be a chordal graph and (s,t) an edge that is not in G.\n\n    If a treewidth_bound is provided, the search for induced nodes will end\n    as soon as the treewidth_bound is exceeded.\n\n    The algorithm is inspired by Algorithm 4 in [1]_.\n    A formal definition of induced node can also be found on that reference.\n\n    Self Loops are ignored\n\n    References\n    ----------\n    .. [1] Learning Bounded Treewidth Bayesian Networks.\n       Gal Elidan, Stephen Gould; JMLR, 9(Dec):2699--2731, 2008.\n       http://jmlr.csail.mit.edu/papers/volume9/elidan08a/elidan08a.pdf\n    '
    if not is_chordal(G):
        raise nx.NetworkXError('Input graph is not chordal.')
    H = nx.Graph(G)
    H.add_edge(s, t)
    induced_nodes = set()
    triplet = _find_chordality_breaker(H, s, treewidth_bound)
    while triplet:
        (u, v, w) = triplet
        induced_nodes.update(triplet)
        for n in triplet:
            if n != s:
                H.add_edge(s, n)
        triplet = _find_chordality_breaker(H, s, treewidth_bound)
    if induced_nodes:
        induced_nodes.add(t)
        for u in G[s]:
            if len(induced_nodes & set(G[u])) == 2:
                induced_nodes.add(u)
                break
    return induced_nodes

@nx._dispatch
def chordal_graph_cliques(G):
    if False:
        return 10
    'Returns all maximal cliques of a chordal graph.\n\n    The algorithm breaks the graph in connected components and performs a\n    maximum cardinality search in each component to get the cliques.\n\n    Parameters\n    ----------\n    G : graph\n      A NetworkX graph\n\n    Yields\n    ------\n    frozenset of nodes\n        Maximal cliques, each of which is a frozenset of\n        nodes in `G`. The order of cliques is arbitrary.\n\n    Raises\n    ------\n    NetworkXError\n        The algorithm does not support DiGraph, MultiGraph and MultiDiGraph.\n        The algorithm can only be applied to chordal graphs. If the input\n        graph is found to be non-chordal, a :exc:`NetworkXError` is raised.\n\n    Examples\n    --------\n    >>> e = [\n    ...     (1, 2),\n    ...     (1, 3),\n    ...     (2, 3),\n    ...     (2, 4),\n    ...     (3, 4),\n    ...     (3, 5),\n    ...     (3, 6),\n    ...     (4, 5),\n    ...     (4, 6),\n    ...     (5, 6),\n    ...     (7, 8),\n    ... ]\n    >>> G = nx.Graph(e)\n    >>> G.add_node(9)\n    >>> cliques = [c for c in chordal_graph_cliques(G)]\n    >>> cliques[0]\n    frozenset({1, 2, 3})\n    '
    for C in (G.subgraph(c).copy() for c in connected_components(G)):
        if C.number_of_nodes() == 1:
            if nx.number_of_selfloops(C) > 0:
                raise nx.NetworkXError('Input graph is not chordal.')
            yield frozenset(C.nodes())
        else:
            unnumbered = set(C.nodes())
            v = arbitrary_element(C)
            unnumbered.remove(v)
            numbered = {v}
            clique_wanna_be = {v}
            while unnumbered:
                v = _max_cardinality_node(C, unnumbered, numbered)
                unnumbered.remove(v)
                numbered.add(v)
                new_clique_wanna_be = set(C.neighbors(v)) & numbered
                sg = C.subgraph(clique_wanna_be)
                if _is_complete_graph(sg):
                    new_clique_wanna_be.add(v)
                    if not new_clique_wanna_be >= clique_wanna_be:
                        yield frozenset(clique_wanna_be)
                    clique_wanna_be = new_clique_wanna_be
                else:
                    raise nx.NetworkXError('Input graph is not chordal.')
            yield frozenset(clique_wanna_be)

@nx._dispatch
def chordal_graph_treewidth(G):
    if False:
        i = 10
        return i + 15
    'Returns the treewidth of the chordal graph G.\n\n    Parameters\n    ----------\n    G : graph\n      A NetworkX graph\n\n    Returns\n    -------\n    treewidth : int\n        The size of the largest clique in the graph minus one.\n\n    Raises\n    ------\n    NetworkXError\n        The algorithm does not support DiGraph, MultiGraph and MultiDiGraph.\n        The algorithm can only be applied to chordal graphs. If the input\n        graph is found to be non-chordal, a :exc:`NetworkXError` is raised.\n\n    Examples\n    --------\n    >>> e = [\n    ...     (1, 2),\n    ...     (1, 3),\n    ...     (2, 3),\n    ...     (2, 4),\n    ...     (3, 4),\n    ...     (3, 5),\n    ...     (3, 6),\n    ...     (4, 5),\n    ...     (4, 6),\n    ...     (5, 6),\n    ...     (7, 8),\n    ... ]\n    >>> G = nx.Graph(e)\n    >>> G.add_node(9)\n    >>> nx.chordal_graph_treewidth(G)\n    3\n\n    References\n    ----------\n    .. [1] https://en.wikipedia.org/wiki/Tree_decomposition#Treewidth\n    '
    if not is_chordal(G):
        raise nx.NetworkXError('Input graph is not chordal.')
    max_clique = -1
    for clique in nx.chordal_graph_cliques(G):
        max_clique = max(max_clique, len(clique))
    return max_clique - 1

def _is_complete_graph(G):
    if False:
        print('Hello World!')
    'Returns True if G is a complete graph.'
    if nx.number_of_selfloops(G) > 0:
        raise nx.NetworkXError('Self loop found in _is_complete_graph()')
    n = G.number_of_nodes()
    if n < 2:
        return True
    e = G.number_of_edges()
    max_edges = n * (n - 1) / 2
    return e == max_edges

def _find_missing_edge(G):
    if False:
        while True:
            i = 10
    'Given a non-complete graph G, returns a missing edge.'
    nodes = set(G)
    for u in G:
        missing = nodes - set(list(G[u].keys()) + [u])
        if missing:
            return (u, missing.pop())

def _max_cardinality_node(G, choices, wanna_connect):
    if False:
        while True:
            i = 10
    'Returns a the node in choices that has more connections in G\n    to nodes in wanna_connect.\n    '
    max_number = -1
    for x in choices:
        number = len([y for y in G[x] if y in wanna_connect])
        if number > max_number:
            max_number = number
            max_cardinality_node = x
    return max_cardinality_node

def _find_chordality_breaker(G, s=None, treewidth_bound=sys.maxsize):
    if False:
        for i in range(10):
            print('nop')
    'Given a graph G, starts a max cardinality search\n    (starting from s if s is given and from an arbitrary node otherwise)\n    trying to find a non-chordal cycle.\n\n    If it does find one, it returns (u,v,w) where u,v,w are the three\n    nodes that together with s are involved in the cycle.\n\n    It ignores any self loops.\n    '
    unnumbered = set(G)
    if s is None:
        s = arbitrary_element(G)
    unnumbered.remove(s)
    numbered = {s}
    current_treewidth = -1
    while unnumbered:
        v = _max_cardinality_node(G, unnumbered, numbered)
        unnumbered.remove(v)
        numbered.add(v)
        clique_wanna_be = set(G[v]) & numbered
        sg = G.subgraph(clique_wanna_be)
        if _is_complete_graph(sg):
            current_treewidth = max(current_treewidth, len(clique_wanna_be))
            if current_treewidth > treewidth_bound:
                raise nx.NetworkXTreewidthBoundExceeded(f'treewidth_bound exceeded: {current_treewidth}')
        else:
            (u, w) = _find_missing_edge(sg)
            return (u, v, w)
    return ()

@not_implemented_for('directed')
@nx._dispatch
def complete_to_chordal_graph(G):
    if False:
        for i in range(10):
            print('nop')
    'Return a copy of G completed to a chordal graph\n\n    Adds edges to a copy of G to create a chordal graph. A graph G=(V,E) is\n    called chordal if for each cycle with length bigger than 3, there exist\n    two non-adjacent nodes connected by an edge (called a chord).\n\n    Parameters\n    ----------\n    G : NetworkX graph\n        Undirected graph\n\n    Returns\n    -------\n    H : NetworkX graph\n        The chordal enhancement of G\n    alpha : Dictionary\n            The elimination ordering of nodes of G\n\n    Notes\n    -----\n    There are different approaches to calculate the chordal\n    enhancement of a graph. The algorithm used here is called\n    MCS-M and gives at least minimal (local) triangulation of graph. Note\n    that this triangulation is not necessarily a global minimum.\n\n    https://en.wikipedia.org/wiki/Chordal_graph\n\n    References\n    ----------\n    .. [1] Berry, Anne & Blair, Jean & Heggernes, Pinar & Peyton, Barry. (2004)\n           Maximum Cardinality Search for Computing Minimal Triangulations of\n           Graphs.  Algorithmica. 39. 287-298. 10.1007/s00453-004-1084-3.\n\n    Examples\n    --------\n    >>> from networkx.algorithms.chordal import complete_to_chordal_graph\n    >>> G = nx.wheel_graph(10)\n    >>> H, alpha = complete_to_chordal_graph(G)\n    '
    H = G.copy()
    alpha = {node: 0 for node in H}
    if nx.is_chordal(H):
        return (H, alpha)
    chords = set()
    weight = {node: 0 for node in H.nodes()}
    unnumbered_nodes = list(H.nodes())
    for i in range(len(H.nodes()), 0, -1):
        z = max(unnumbered_nodes, key=lambda node: weight[node])
        unnumbered_nodes.remove(z)
        alpha[z] = i
        update_nodes = []
        for y in unnumbered_nodes:
            if G.has_edge(y, z):
                update_nodes.append(y)
            else:
                y_weight = weight[y]
                lower_nodes = [node for node in unnumbered_nodes if weight[node] < y_weight]
                if nx.has_path(H.subgraph(lower_nodes + [z, y]), y, z):
                    update_nodes.append(y)
                    chords.add((z, y))
        for node in update_nodes:
            weight[node] += 1
    H.add_edges_from(chords)
    return (H, alpha)