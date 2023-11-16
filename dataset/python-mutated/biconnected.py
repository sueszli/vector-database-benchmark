"""Biconnected components and articulation points."""
from itertools import chain
import networkx as nx
from networkx.utils.decorators import not_implemented_for
__all__ = ['biconnected_components', 'biconnected_component_edges', 'is_biconnected', 'articulation_points']

@not_implemented_for('directed')
@nx._dispatch
def is_biconnected(G):
    if False:
        while True:
            i = 10
    'Returns True if the graph is biconnected, False otherwise.\n\n    A graph is biconnected if, and only if, it cannot be disconnected by\n    removing only one node (and all edges incident on that node).  If\n    removing a node increases the number of disconnected components\n    in the graph, that node is called an articulation point, or cut\n    vertex.  A biconnected graph has no articulation points.\n\n    Parameters\n    ----------\n    G : NetworkX Graph\n        An undirected graph.\n\n    Returns\n    -------\n    biconnected : bool\n        True if the graph is biconnected, False otherwise.\n\n    Raises\n    ------\n    NetworkXNotImplemented\n        If the input graph is not undirected.\n\n    Examples\n    --------\n    >>> G = nx.path_graph(4)\n    >>> print(nx.is_biconnected(G))\n    False\n    >>> G.add_edge(0, 3)\n    >>> print(nx.is_biconnected(G))\n    True\n\n    See Also\n    --------\n    biconnected_components\n    articulation_points\n    biconnected_component_edges\n    is_strongly_connected\n    is_weakly_connected\n    is_connected\n    is_semiconnected\n\n    Notes\n    -----\n    The algorithm to find articulation points and biconnected\n    components is implemented using a non-recursive depth-first-search\n    (DFS) that keeps track of the highest level that back edges reach\n    in the DFS tree.  A node `n` is an articulation point if, and only\n    if, there exists a subtree rooted at `n` such that there is no\n    back edge from any successor of `n` that links to a predecessor of\n    `n` in the DFS tree.  By keeping track of all the edges traversed\n    by the DFS we can obtain the biconnected components because all\n    edges of a bicomponent will be traversed consecutively between\n    articulation points.\n\n    References\n    ----------\n    .. [1] Hopcroft, J.; Tarjan, R. (1973).\n       "Efficient algorithms for graph manipulation".\n       Communications of the ACM 16: 372–378. doi:10.1145/362248.362272\n\n    '
    bccs = biconnected_components(G)
    try:
        bcc = next(bccs)
    except StopIteration:
        return False
    try:
        next(bccs)
    except StopIteration:
        return len(bcc) == len(G)
    else:
        return False

@not_implemented_for('directed')
@nx._dispatch
def biconnected_component_edges(G):
    if False:
        for i in range(10):
            print('nop')
    'Returns a generator of lists of edges, one list for each biconnected\n    component of the input graph.\n\n    Biconnected components are maximal subgraphs such that the removal of a\n    node (and all edges incident on that node) will not disconnect the\n    subgraph.  Note that nodes may be part of more than one biconnected\n    component.  Those nodes are articulation points, or cut vertices.\n    However, each edge belongs to one, and only one, biconnected component.\n\n    Notice that by convention a dyad is considered a biconnected component.\n\n    Parameters\n    ----------\n    G : NetworkX Graph\n        An undirected graph.\n\n    Returns\n    -------\n    edges : generator of lists\n        Generator of lists of edges, one list for each bicomponent.\n\n    Raises\n    ------\n    NetworkXNotImplemented\n        If the input graph is not undirected.\n\n    Examples\n    --------\n    >>> G = nx.barbell_graph(4, 2)\n    >>> print(nx.is_biconnected(G))\n    False\n    >>> bicomponents_edges = list(nx.biconnected_component_edges(G))\n    >>> len(bicomponents_edges)\n    5\n    >>> G.add_edge(2, 8)\n    >>> print(nx.is_biconnected(G))\n    True\n    >>> bicomponents_edges = list(nx.biconnected_component_edges(G))\n    >>> len(bicomponents_edges)\n    1\n\n    See Also\n    --------\n    is_biconnected,\n    biconnected_components,\n    articulation_points,\n\n    Notes\n    -----\n    The algorithm to find articulation points and biconnected\n    components is implemented using a non-recursive depth-first-search\n    (DFS) that keeps track of the highest level that back edges reach\n    in the DFS tree.  A node `n` is an articulation point if, and only\n    if, there exists a subtree rooted at `n` such that there is no\n    back edge from any successor of `n` that links to a predecessor of\n    `n` in the DFS tree.  By keeping track of all the edges traversed\n    by the DFS we can obtain the biconnected components because all\n    edges of a bicomponent will be traversed consecutively between\n    articulation points.\n\n    References\n    ----------\n    .. [1] Hopcroft, J.; Tarjan, R. (1973).\n           "Efficient algorithms for graph manipulation".\n           Communications of the ACM 16: 372–378. doi:10.1145/362248.362272\n\n    '
    yield from _biconnected_dfs(G, components=True)

@not_implemented_for('directed')
@nx._dispatch
def biconnected_components(G):
    if False:
        while True:
            i = 10
    'Returns a generator of sets of nodes, one set for each biconnected\n    component of the graph\n\n    Biconnected components are maximal subgraphs such that the removal of a\n    node (and all edges incident on that node) will not disconnect the\n    subgraph. Note that nodes may be part of more than one biconnected\n    component.  Those nodes are articulation points, or cut vertices.  The\n    removal of articulation points will increase the number of connected\n    components of the graph.\n\n    Notice that by convention a dyad is considered a biconnected component.\n\n    Parameters\n    ----------\n    G : NetworkX Graph\n        An undirected graph.\n\n    Returns\n    -------\n    nodes : generator\n        Generator of sets of nodes, one set for each biconnected component.\n\n    Raises\n    ------\n    NetworkXNotImplemented\n        If the input graph is not undirected.\n\n    Examples\n    --------\n    >>> G = nx.lollipop_graph(5, 1)\n    >>> print(nx.is_biconnected(G))\n    False\n    >>> bicomponents = list(nx.biconnected_components(G))\n    >>> len(bicomponents)\n    2\n    >>> G.add_edge(0, 5)\n    >>> print(nx.is_biconnected(G))\n    True\n    >>> bicomponents = list(nx.biconnected_components(G))\n    >>> len(bicomponents)\n    1\n\n    You can generate a sorted list of biconnected components, largest\n    first, using sort.\n\n    >>> G.remove_edge(0, 5)\n    >>> [len(c) for c in sorted(nx.biconnected_components(G), key=len, reverse=True)]\n    [5, 2]\n\n    If you only want the largest connected component, it\'s more\n    efficient to use max instead of sort.\n\n    >>> Gc = max(nx.biconnected_components(G), key=len)\n\n    To create the components as subgraphs use:\n    ``(G.subgraph(c).copy() for c in biconnected_components(G))``\n\n    See Also\n    --------\n    is_biconnected\n    articulation_points\n    biconnected_component_edges\n    k_components : this function is a special case where k=2\n    bridge_components : similar to this function, but is defined using\n        2-edge-connectivity instead of 2-node-connectivity.\n\n    Notes\n    -----\n    The algorithm to find articulation points and biconnected\n    components is implemented using a non-recursive depth-first-search\n    (DFS) that keeps track of the highest level that back edges reach\n    in the DFS tree.  A node `n` is an articulation point if, and only\n    if, there exists a subtree rooted at `n` such that there is no\n    back edge from any successor of `n` that links to a predecessor of\n    `n` in the DFS tree.  By keeping track of all the edges traversed\n    by the DFS we can obtain the biconnected components because all\n    edges of a bicomponent will be traversed consecutively between\n    articulation points.\n\n    References\n    ----------\n    .. [1] Hopcroft, J.; Tarjan, R. (1973).\n           "Efficient algorithms for graph manipulation".\n           Communications of the ACM 16: 372–378. doi:10.1145/362248.362272\n\n    '
    for comp in _biconnected_dfs(G, components=True):
        yield set(chain.from_iterable(comp))

@not_implemented_for('directed')
@nx._dispatch
def articulation_points(G):
    if False:
        return 10
    'Yield the articulation points, or cut vertices, of a graph.\n\n    An articulation point or cut vertex is any node whose removal (along with\n    all its incident edges) increases the number of connected components of\n    a graph.  An undirected connected graph without articulation points is\n    biconnected. Articulation points belong to more than one biconnected\n    component of a graph.\n\n    Notice that by convention a dyad is considered a biconnected component.\n\n    Parameters\n    ----------\n    G : NetworkX Graph\n        An undirected graph.\n\n    Yields\n    ------\n    node\n        An articulation point in the graph.\n\n    Raises\n    ------\n    NetworkXNotImplemented\n        If the input graph is not undirected.\n\n    Examples\n    --------\n\n    >>> G = nx.barbell_graph(4, 2)\n    >>> print(nx.is_biconnected(G))\n    False\n    >>> len(list(nx.articulation_points(G)))\n    4\n    >>> G.add_edge(2, 8)\n    >>> print(nx.is_biconnected(G))\n    True\n    >>> len(list(nx.articulation_points(G)))\n    0\n\n    See Also\n    --------\n    is_biconnected\n    biconnected_components\n    biconnected_component_edges\n\n    Notes\n    -----\n    The algorithm to find articulation points and biconnected\n    components is implemented using a non-recursive depth-first-search\n    (DFS) that keeps track of the highest level that back edges reach\n    in the DFS tree.  A node `n` is an articulation point if, and only\n    if, there exists a subtree rooted at `n` such that there is no\n    back edge from any successor of `n` that links to a predecessor of\n    `n` in the DFS tree.  By keeping track of all the edges traversed\n    by the DFS we can obtain the biconnected components because all\n    edges of a bicomponent will be traversed consecutively between\n    articulation points.\n\n    References\n    ----------\n    .. [1] Hopcroft, J.; Tarjan, R. (1973).\n           "Efficient algorithms for graph manipulation".\n           Communications of the ACM 16: 372–378. doi:10.1145/362248.362272\n\n    '
    seen = set()
    for articulation in _biconnected_dfs(G, components=False):
        if articulation not in seen:
            seen.add(articulation)
            yield articulation

@not_implemented_for('directed')
def _biconnected_dfs(G, components=True):
    if False:
        for i in range(10):
            print('nop')
    visited = set()
    for start in G:
        if start in visited:
            continue
        discovery = {start: 0}
        low = {start: 0}
        root_children = 0
        visited.add(start)
        edge_stack = []
        stack = [(start, start, iter(G[start]))]
        edge_index = {}
        while stack:
            (grandparent, parent, children) = stack[-1]
            try:
                child = next(children)
                if grandparent == child:
                    continue
                if child in visited:
                    if discovery[child] <= discovery[parent]:
                        low[parent] = min(low[parent], discovery[child])
                        if components:
                            edge_index[parent, child] = len(edge_stack)
                            edge_stack.append((parent, child))
                else:
                    low[child] = discovery[child] = len(discovery)
                    visited.add(child)
                    stack.append((parent, child, iter(G[child])))
                    if components:
                        edge_index[parent, child] = len(edge_stack)
                        edge_stack.append((parent, child))
            except StopIteration:
                stack.pop()
                if len(stack) > 1:
                    if low[parent] >= discovery[grandparent]:
                        if components:
                            ind = edge_index[grandparent, parent]
                            yield edge_stack[ind:]
                            del edge_stack[ind:]
                        else:
                            yield grandparent
                    low[grandparent] = min(low[parent], low[grandparent])
                elif stack:
                    root_children += 1
                    if components:
                        ind = edge_index[grandparent, parent]
                        yield edge_stack[ind:]
                        del edge_stack[ind:]
        if not components:
            if root_children > 1:
                yield start