"""Functions for finding chains in a graph."""
import networkx as nx
from networkx.utils import not_implemented_for
__all__ = ['chain_decomposition']

@not_implemented_for('directed')
@not_implemented_for('multigraph')
@nx._dispatch
def chain_decomposition(G, root=None):
    if False:
        print('Hello World!')
    'Returns the chain decomposition of a graph.\n\n    The *chain decomposition* of a graph with respect a depth-first\n    search tree is a set of cycles or paths derived from the set of\n    fundamental cycles of the tree in the following manner. Consider\n    each fundamental cycle with respect to the given tree, represented\n    as a list of edges beginning with the nontree edge oriented away\n    from the root of the tree. For each fundamental cycle, if it\n    overlaps with any previous fundamental cycle, just take the initial\n    non-overlapping segment, which is a path instead of a cycle. Each\n    cycle or path is called a *chain*. For more information, see [1]_.\n\n    Parameters\n    ----------\n    G : undirected graph\n\n    root : node (optional)\n       A node in the graph `G`. If specified, only the chain\n       decomposition for the connected component containing this node\n       will be returned. This node indicates the root of the depth-first\n       search tree.\n\n    Yields\n    ------\n    chain : list\n       A list of edges representing a chain. There is no guarantee on\n       the orientation of the edges in each chain (for example, if a\n       chain includes the edge joining nodes 1 and 2, the chain may\n       include either (1, 2) or (2, 1)).\n\n    Raises\n    ------\n    NodeNotFound\n       If `root` is not in the graph `G`.\n\n    Examples\n    --------\n    >>> G = nx.Graph([(0, 1), (1, 4), (3, 4), (3, 5), (4, 5)])\n    >>> list(nx.chain_decomposition(G))\n    [[(4, 5), (5, 3), (3, 4)]]\n\n    Notes\n    -----\n    The worst-case running time of this implementation is linear in the\n    number of nodes and number of edges [1]_.\n\n    References\n    ----------\n    .. [1] Jens M. Schmidt (2013). "A simple test on 2-vertex-\n       and 2-edge-connectivity." *Information Processing Letters*,\n       113, 241–244. Elsevier. <https://doi.org/10.1016/j.ipl.2013.01.016>\n\n    '

    def _dfs_cycle_forest(G, root=None):
        if False:
            for i in range(10):
                print('nop')
        'Builds a directed graph composed of cycles from the given graph.\n\n        `G` is an undirected simple graph. `root` is a node in the graph\n        from which the depth-first search is started.\n\n        This function returns both the depth-first search cycle graph\n        (as a :class:`~networkx.DiGraph`) and the list of nodes in\n        depth-first preorder. The depth-first search cycle graph is a\n        directed graph whose edges are the edges of `G` oriented toward\n        the root if the edge is a tree edge and away from the root if\n        the edge is a non-tree edge. If `root` is not specified, this\n        performs a depth-first search on each connected component of `G`\n        and returns a directed forest instead.\n\n        If `root` is not in the graph, this raises :exc:`KeyError`.\n\n        '
        H = nx.DiGraph()
        nodes = []
        for (u, v, d) in nx.dfs_labeled_edges(G, source=root):
            if d == 'forward':
                if u == v:
                    H.add_node(v, parent=None)
                    nodes.append(v)
                else:
                    H.add_node(v, parent=u)
                    H.add_edge(v, u, nontree=False)
                    nodes.append(v)
            elif d == 'nontree' and v not in H[u]:
                H.add_edge(v, u, nontree=True)
            else:
                pass
        return (H, nodes)

    def _build_chain(G, u, v, visited):
        if False:
            return 10
        "Generate the chain starting from the given nontree edge.\n\n        `G` is a DFS cycle graph as constructed by\n        :func:`_dfs_cycle_graph`. The edge (`u`, `v`) is a nontree edge\n        that begins a chain. `visited` is a set representing the nodes\n        in `G` that have already been visited.\n\n        This function yields the edges in an initial segment of the\n        fundamental cycle of `G` starting with the nontree edge (`u`,\n        `v`) that includes all the edges up until the first node that\n        appears in `visited`. The tree edges are given by the 'parent'\n        node attribute. The `visited` set is updated to add each node in\n        an edge yielded by this function.\n\n        "
        while v not in visited:
            yield (u, v)
            visited.add(v)
            (u, v) = (v, G.nodes[v]['parent'])
        yield (u, v)
    if root is not None and root not in G:
        raise nx.NodeNotFound(f'Root node {root} is not in graph')
    (H, nodes) = _dfs_cycle_forest(G, root)
    visited = set()
    for u in nodes:
        visited.add(u)
        edges = ((u, v) for (u, v, d) in H.out_edges(u, data='nontree') if d)
        for (u, v) in edges:
            chain = list(_build_chain(H, u, v, visited))
            yield chain