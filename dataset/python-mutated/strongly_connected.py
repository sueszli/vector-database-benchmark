"""Strongly connected components."""
import networkx as nx
from networkx.utils.decorators import not_implemented_for
__all__ = ['number_strongly_connected_components', 'strongly_connected_components', 'is_strongly_connected', 'strongly_connected_components_recursive', 'kosaraju_strongly_connected_components', 'condensation']

@not_implemented_for('undirected')
@nx._dispatch
def strongly_connected_components(G):
    if False:
        return 10
    "Generate nodes in strongly connected components of graph.\n\n    Parameters\n    ----------\n    G : NetworkX Graph\n        A directed graph.\n\n    Returns\n    -------\n    comp : generator of sets\n        A generator of sets of nodes, one for each strongly connected\n        component of G.\n\n    Raises\n    ------\n    NetworkXNotImplemented\n        If G is undirected.\n\n    Examples\n    --------\n    Generate a sorted list of strongly connected components, largest first.\n\n    >>> G = nx.cycle_graph(4, create_using=nx.DiGraph())\n    >>> nx.add_cycle(G, [10, 11, 12])\n    >>> [\n    ...     len(c)\n    ...     for c in sorted(nx.strongly_connected_components(G), key=len, reverse=True)\n    ... ]\n    [4, 3]\n\n    If you only want the largest component, it's more efficient to\n    use max instead of sort.\n\n    >>> largest = max(nx.strongly_connected_components(G), key=len)\n\n    See Also\n    --------\n    connected_components\n    weakly_connected_components\n    kosaraju_strongly_connected_components\n\n    Notes\n    -----\n    Uses Tarjan's algorithm[1]_ with Nuutila's modifications[2]_.\n    Nonrecursive version of algorithm.\n\n    References\n    ----------\n    .. [1] Depth-first search and linear graph algorithms, R. Tarjan\n       SIAM Journal of Computing 1(2):146-160, (1972).\n\n    .. [2] On finding the strongly connected components in a directed graph.\n       E. Nuutila and E. Soisalon-Soinen\n       Information Processing Letters 49(1): 9-14, (1994)..\n\n    "
    preorder = {}
    lowlink = {}
    scc_found = set()
    scc_queue = []
    i = 0
    neighbors = {v: iter(G[v]) for v in G}
    for source in G:
        if source not in scc_found:
            queue = [source]
            while queue:
                v = queue[-1]
                if v not in preorder:
                    i = i + 1
                    preorder[v] = i
                done = True
                for w in neighbors[v]:
                    if w not in preorder:
                        queue.append(w)
                        done = False
                        break
                if done:
                    lowlink[v] = preorder[v]
                    for w in G[v]:
                        if w not in scc_found:
                            if preorder[w] > preorder[v]:
                                lowlink[v] = min([lowlink[v], lowlink[w]])
                            else:
                                lowlink[v] = min([lowlink[v], preorder[w]])
                    queue.pop()
                    if lowlink[v] == preorder[v]:
                        scc = {v}
                        while scc_queue and preorder[scc_queue[-1]] > preorder[v]:
                            k = scc_queue.pop()
                            scc.add(k)
                        scc_found.update(scc)
                        yield scc
                    else:
                        scc_queue.append(v)

@not_implemented_for('undirected')
@nx._dispatch
def kosaraju_strongly_connected_components(G, source=None):
    if False:
        while True:
            i = 10
    "Generate nodes in strongly connected components of graph.\n\n    Parameters\n    ----------\n    G : NetworkX Graph\n        A directed graph.\n\n    Returns\n    -------\n    comp : generator of sets\n        A generator of sets of nodes, one for each strongly connected\n        component of G.\n\n    Raises\n    ------\n    NetworkXNotImplemented\n        If G is undirected.\n\n    Examples\n    --------\n    Generate a sorted list of strongly connected components, largest first.\n\n    >>> G = nx.cycle_graph(4, create_using=nx.DiGraph())\n    >>> nx.add_cycle(G, [10, 11, 12])\n    >>> [\n    ...     len(c)\n    ...     for c in sorted(\n    ...         nx.kosaraju_strongly_connected_components(G), key=len, reverse=True\n    ...     )\n    ... ]\n    [4, 3]\n\n    If you only want the largest component, it's more efficient to\n    use max instead of sort.\n\n    >>> largest = max(nx.kosaraju_strongly_connected_components(G), key=len)\n\n    See Also\n    --------\n    strongly_connected_components\n\n    Notes\n    -----\n    Uses Kosaraju's algorithm.\n\n    "
    post = list(nx.dfs_postorder_nodes(G.reverse(copy=False), source=source))
    seen = set()
    while post:
        r = post.pop()
        if r in seen:
            continue
        c = nx.dfs_preorder_nodes(G, r)
        new = {v for v in c if v not in seen}
        seen.update(new)
        yield new

@not_implemented_for('undirected')
@nx._dispatch
def strongly_connected_components_recursive(G):
    if False:
        for i in range(10):
            print('nop')
    "Generate nodes in strongly connected components of graph.\n\n    .. deprecated:: 3.2\n\n       This function is deprecated and will be removed in a future version of\n       NetworkX. Use `strongly_connected_components` instead.\n\n    Recursive version of algorithm.\n\n    Parameters\n    ----------\n    G : NetworkX Graph\n        A directed graph.\n\n    Returns\n    -------\n    comp : generator of sets\n        A generator of sets of nodes, one for each strongly connected\n        component of G.\n\n    Raises\n    ------\n    NetworkXNotImplemented\n        If G is undirected.\n\n    Examples\n    --------\n    Generate a sorted list of strongly connected components, largest first.\n\n    >>> G = nx.cycle_graph(4, create_using=nx.DiGraph())\n    >>> nx.add_cycle(G, [10, 11, 12])\n    >>> [\n    ...     len(c)\n    ...     for c in sorted(\n    ...         nx.strongly_connected_components_recursive(G), key=len, reverse=True\n    ...     )\n    ... ]\n    [4, 3]\n\n    If you only want the largest component, it's more efficient to\n    use max instead of sort.\n\n    >>> largest = max(nx.strongly_connected_components_recursive(G), key=len)\n\n    To create the induced subgraph of the components use:\n    >>> S = [G.subgraph(c).copy() for c in nx.weakly_connected_components(G)]\n\n    See Also\n    --------\n    connected_components\n\n    Notes\n    -----\n    Uses Tarjan's algorithm[1]_ with Nuutila's modifications[2]_.\n\n    References\n    ----------\n    .. [1] Depth-first search and linear graph algorithms, R. Tarjan\n       SIAM Journal of Computing 1(2):146-160, (1972).\n\n    .. [2] On finding the strongly connected components in a directed graph.\n       E. Nuutila and E. Soisalon-Soinen\n       Information Processing Letters 49(1): 9-14, (1994)..\n\n    "
    import warnings
    warnings.warn('\n\nstrongly_connected_components_recursive is deprecated and will be\nremoved in the future. Use strongly_connected_components instead.', category=DeprecationWarning, stacklevel=2)
    yield from strongly_connected_components(G)

@not_implemented_for('undirected')
@nx._dispatch
def number_strongly_connected_components(G):
    if False:
        for i in range(10):
            print('nop')
    'Returns number of strongly connected components in graph.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n       A directed graph.\n\n    Returns\n    -------\n    n : integer\n       Number of strongly connected components\n\n    Raises\n    ------\n    NetworkXNotImplemented\n        If G is undirected.\n\n    Examples\n    --------\n    >>> G = nx.DiGraph([(0, 1), (1, 2), (2, 0), (2, 3), (4, 5), (3, 4), (5, 6), (6, 3), (6, 7)])\n    >>> nx.number_strongly_connected_components(G)\n    3\n\n    See Also\n    --------\n    strongly_connected_components\n    number_connected_components\n    number_weakly_connected_components\n\n    Notes\n    -----\n    For directed graphs only.\n    '
    return sum((1 for scc in strongly_connected_components(G)))

@not_implemented_for('undirected')
@nx._dispatch
def is_strongly_connected(G):
    if False:
        print('Hello World!')
    'Test directed graph for strong connectivity.\n\n    A directed graph is strongly connected if and only if every vertex in\n    the graph is reachable from every other vertex.\n\n    Parameters\n    ----------\n    G : NetworkX Graph\n       A directed graph.\n\n    Returns\n    -------\n    connected : bool\n      True if the graph is strongly connected, False otherwise.\n\n    Examples\n    --------\n    >>> G = nx.DiGraph([(0, 1), (1, 2), (2, 3), (3, 0), (2, 4), (4, 2)])\n    >>> nx.is_strongly_connected(G)\n    True\n    >>> G.remove_edge(2, 3)\n    >>> nx.is_strongly_connected(G)\n    False\n\n    Raises\n    ------\n    NetworkXNotImplemented\n        If G is undirected.\n\n    See Also\n    --------\n    is_weakly_connected\n    is_semiconnected\n    is_connected\n    is_biconnected\n    strongly_connected_components\n\n    Notes\n    -----\n    For directed graphs only.\n    '
    if len(G) == 0:
        raise nx.NetworkXPointlessConcept('Connectivity is undefined for the null graph.')
    return len(next(strongly_connected_components(G))) == len(G)

@not_implemented_for('undirected')
@nx._dispatch
def condensation(G, scc=None):
    if False:
        print('Hello World!')
    "Returns the condensation of G.\n\n    The condensation of G is the graph with each of the strongly connected\n    components contracted into a single node.\n\n    Parameters\n    ----------\n    G : NetworkX DiGraph\n       A directed graph.\n\n    scc:  list or generator (optional, default=None)\n       Strongly connected components. If provided, the elements in\n       `scc` must partition the nodes in `G`. If not provided, it will be\n       calculated as scc=nx.strongly_connected_components(G).\n\n    Returns\n    -------\n    C : NetworkX DiGraph\n       The condensation graph C of G.  The node labels are integers\n       corresponding to the index of the component in the list of\n       strongly connected components of G.  C has a graph attribute named\n       'mapping' with a dictionary mapping the original nodes to the\n       nodes in C to which they belong.  Each node in C also has a node\n       attribute 'members' with the set of original nodes in G that\n       form the SCC that the node in C represents.\n\n    Raises\n    ------\n    NetworkXNotImplemented\n        If G is undirected.\n\n    Examples\n    --------\n    Contracting two sets of strongly connected nodes into two distinct SCC\n    using the barbell graph.\n\n    >>> G = nx.barbell_graph(4, 0)\n    >>> G.remove_edge(3, 4)\n    >>> G = nx.DiGraph(G)\n    >>> H = nx.condensation(G)\n    >>> H.nodes.data()\n    NodeDataView({0: {'members': {0, 1, 2, 3}}, 1: {'members': {4, 5, 6, 7}}})\n    >>> H.graph['mapping']\n    {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 1}\n\n    Contracting a complete graph into one single SCC.\n\n    >>> G = nx.complete_graph(7, create_using=nx.DiGraph)\n    >>> H = nx.condensation(G)\n    >>> H.nodes\n    NodeView((0,))\n    >>> H.nodes.data()\n    NodeDataView({0: {'members': {0, 1, 2, 3, 4, 5, 6}}})\n\n    Notes\n    -----\n    After contracting all strongly connected components to a single node,\n    the resulting graph is a directed acyclic graph.\n\n    "
    if scc is None:
        scc = nx.strongly_connected_components(G)
    mapping = {}
    members = {}
    C = nx.DiGraph()
    C.graph['mapping'] = mapping
    if len(G) == 0:
        return C
    for (i, component) in enumerate(scc):
        members[i] = component
        mapping.update(((n, i) for n in component))
    number_of_components = i + 1
    C.add_nodes_from(range(number_of_components))
    C.add_edges_from(((mapping[u], mapping[v]) for (u, v) in G.edges() if mapping[u] != mapping[v]))
    nx.set_node_attributes(C, members, 'members')
    return C