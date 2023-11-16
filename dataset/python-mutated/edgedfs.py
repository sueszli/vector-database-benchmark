"""
===========================
Depth First Search on Edges
===========================

Algorithms for a depth-first traversal of edges in a graph.

"""
import networkx as nx
FORWARD = 'forward'
REVERSE = 'reverse'
__all__ = ['edge_dfs']

@nx._dispatch
def edge_dfs(G, source=None, orientation=None):
    if False:
        for i in range(10):
            print('nop')
    'A directed, depth-first-search of edges in `G`, beginning at `source`.\n\n    Yield the edges of G in a depth-first-search order continuing until\n    all edges are generated.\n\n    Parameters\n    ----------\n    G : graph\n        A directed/undirected graph/multigraph.\n\n    source : node, list of nodes\n        The node from which the traversal begins. If None, then a source\n        is chosen arbitrarily and repeatedly until all edges from each node in\n        the graph are searched.\n\n    orientation : None | \'original\' | \'reverse\' | \'ignore\' (default: None)\n        For directed graphs and directed multigraphs, edge traversals need not\n        respect the original orientation of the edges.\n        When set to \'reverse\' every edge is traversed in the reverse direction.\n        When set to \'ignore\', every edge is treated as undirected.\n        When set to \'original\', every edge is treated as directed.\n        In all three cases, the yielded edge tuples add a last entry to\n        indicate the direction in which that edge was traversed.\n        If orientation is None, the yielded edge has no direction indicated.\n        The direction is respected, but not reported.\n\n    Yields\n    ------\n    edge : directed edge\n        A directed edge indicating the path taken by the depth-first traversal.\n        For graphs, `edge` is of the form `(u, v)` where `u` and `v`\n        are the tail and head of the edge as determined by the traversal.\n        For multigraphs, `edge` is of the form `(u, v, key)`, where `key` is\n        the key of the edge. When the graph is directed, then `u` and `v`\n        are always in the order of the actual directed edge.\n        If orientation is not None then the edge tuple is extended to include\n        the direction of traversal (\'forward\' or \'reverse\') on that edge.\n\n    Examples\n    --------\n    >>> nodes = [0, 1, 2, 3]\n    >>> edges = [(0, 1), (1, 0), (1, 0), (2, 1), (3, 1)]\n\n    >>> list(nx.edge_dfs(nx.Graph(edges), nodes))\n    [(0, 1), (1, 2), (1, 3)]\n\n    >>> list(nx.edge_dfs(nx.DiGraph(edges), nodes))\n    [(0, 1), (1, 0), (2, 1), (3, 1)]\n\n    >>> list(nx.edge_dfs(nx.MultiGraph(edges), nodes))\n    [(0, 1, 0), (1, 0, 1), (0, 1, 2), (1, 2, 0), (1, 3, 0)]\n\n    >>> list(nx.edge_dfs(nx.MultiDiGraph(edges), nodes))\n    [(0, 1, 0), (1, 0, 0), (1, 0, 1), (2, 1, 0), (3, 1, 0)]\n\n    >>> list(nx.edge_dfs(nx.DiGraph(edges), nodes, orientation="ignore"))\n    [(0, 1, \'forward\'), (1, 0, \'forward\'), (2, 1, \'reverse\'), (3, 1, \'reverse\')]\n\n    >>> list(nx.edge_dfs(nx.MultiDiGraph(edges), nodes, orientation="ignore"))\n    [(0, 1, 0, \'forward\'), (1, 0, 0, \'forward\'), (1, 0, 1, \'reverse\'), (2, 1, 0, \'reverse\'), (3, 1, 0, \'reverse\')]\n\n    Notes\n    -----\n    The goal of this function is to visit edges. It differs from the more\n    familiar depth-first traversal of nodes, as provided by\n    :func:`~networkx.algorithms.traversal.depth_first_search.dfs_edges`, in\n    that it does not stop once every node has been visited. In a directed graph\n    with edges [(0, 1), (1, 2), (2, 1)], the edge (2, 1) would not be visited\n    if not for the functionality provided by this function.\n\n    See Also\n    --------\n    :func:`~networkx.algorithms.traversal.depth_first_search.dfs_edges`\n\n    '
    nodes = list(G.nbunch_iter(source))
    if not nodes:
        return
    directed = G.is_directed()
    kwds = {'data': False}
    if G.is_multigraph() is True:
        kwds['keys'] = True
    if orientation is None:

        def edges_from(node):
            if False:
                while True:
                    i = 10
            return iter(G.edges(node, **kwds))
    elif not directed or orientation == 'original':

        def edges_from(node):
            if False:
                return 10
            for e in G.edges(node, **kwds):
                yield (e + (FORWARD,))
    elif orientation == 'reverse':

        def edges_from(node):
            if False:
                print('Hello World!')
            for e in G.in_edges(node, **kwds):
                yield (e + (REVERSE,))
    elif orientation == 'ignore':

        def edges_from(node):
            if False:
                while True:
                    i = 10
            for e in G.edges(node, **kwds):
                yield (e + (FORWARD,))
            for e in G.in_edges(node, **kwds):
                yield (e + (REVERSE,))
    else:
        raise nx.NetworkXError('invalid orientation argument.')
    if directed:

        def edge_id(edge):
            if False:
                for i in range(10):
                    print('nop')
            return edge[:-1] if orientation is not None else edge
    else:

        def edge_id(edge):
            if False:
                return 10
            return (frozenset(edge[:2]),) + edge[2:]
    check_reverse = directed and orientation in ('reverse', 'ignore')
    visited_edges = set()
    visited_nodes = set()
    edges = {}
    for start_node in nodes:
        stack = [start_node]
        while stack:
            current_node = stack[-1]
            if current_node not in visited_nodes:
                edges[current_node] = edges_from(current_node)
                visited_nodes.add(current_node)
            try:
                edge = next(edges[current_node])
            except StopIteration:
                stack.pop()
            else:
                edgeid = edge_id(edge)
                if edgeid not in visited_edges:
                    visited_edges.add(edgeid)
                    if check_reverse and edge[-1] == REVERSE:
                        stack.append(edge[0])
                    else:
                        stack.append(edge[1])
                    yield edge