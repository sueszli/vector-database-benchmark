"""
Shortest path algorithms for weighted graphs.
"""
from collections import deque
from heapq import heappop, heappush
from itertools import count
import networkx as nx
from networkx.algorithms.shortest_paths.generic import _build_paths_from_predecessors
__all__ = ['dijkstra_path', 'dijkstra_path_length', 'bidirectional_dijkstra', 'single_source_dijkstra', 'single_source_dijkstra_path', 'single_source_dijkstra_path_length', 'multi_source_dijkstra', 'multi_source_dijkstra_path', 'multi_source_dijkstra_path_length', 'all_pairs_dijkstra', 'all_pairs_dijkstra_path', 'all_pairs_dijkstra_path_length', 'dijkstra_predecessor_and_distance', 'bellman_ford_path', 'bellman_ford_path_length', 'single_source_bellman_ford', 'single_source_bellman_ford_path', 'single_source_bellman_ford_path_length', 'all_pairs_bellman_ford_path', 'all_pairs_bellman_ford_path_length', 'bellman_ford_predecessor_and_distance', 'negative_edge_cycle', 'find_negative_cycle', 'goldberg_radzik', 'johnson']

def _weight_function(G, weight):
    if False:
        while True:
            i = 10
    'Returns a function that returns the weight of an edge.\n\n    The returned function is specifically suitable for input to\n    functions :func:`_dijkstra` and :func:`_bellman_ford_relaxation`.\n\n    Parameters\n    ----------\n    G : NetworkX graph.\n\n    weight : string or function\n        If it is callable, `weight` itself is returned. If it is a string,\n        it is assumed to be the name of the edge attribute that represents\n        the weight of an edge. In that case, a function is returned that\n        gets the edge weight according to the specified edge attribute.\n\n    Returns\n    -------\n    function\n        This function returns a callable that accepts exactly three inputs:\n        a node, an node adjacent to the first one, and the edge attribute\n        dictionary for the eedge joining those nodes. That function returns\n        a number representing the weight of an edge.\n\n    If `G` is a multigraph, and `weight` is not callable, the\n    minimum edge weight over all parallel edges is returned. If any edge\n    does not have an attribute with key `weight`, it is assumed to\n    have weight one.\n\n    '
    if callable(weight):
        return weight
    if G.is_multigraph():
        return lambda u, v, d: min((attr.get(weight, 1) for attr in d.values()))
    return lambda u, v, data: data.get(weight, 1)

@nx._dispatch(edge_attrs='weight')
def dijkstra_path(G, source, target, weight='weight'):
    if False:
        for i in range(10):
            print('nop')
    'Returns the shortest weighted path from source to target in G.\n\n    Uses Dijkstra\'s Method to compute the shortest weighted path\n    between two nodes in a graph.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n\n    source : node\n        Starting node\n\n    target : node\n        Ending node\n\n    weight : string or function\n        If this is a string, then edge weights will be accessed via the\n        edge attribute with this key (that is, the weight of the edge\n        joining `u` to `v` will be ``G.edges[u, v][weight]``). If no\n        such edge attribute exists, the weight of the edge is assumed to\n        be one.\n\n        If this is a function, the weight of an edge is the value\n        returned by the function. The function must accept exactly three\n        positional arguments: the two endpoints of an edge and the\n        dictionary of edge attributes for that edge. The function must\n        return a number or None to indicate a hidden edge.\n\n    Returns\n    -------\n    path : list\n        List of nodes in a shortest path.\n\n    Raises\n    ------\n    NodeNotFound\n        If `source` is not in `G`.\n\n    NetworkXNoPath\n        If no path exists between source and target.\n\n    Examples\n    --------\n    >>> G = nx.path_graph(5)\n    >>> print(nx.dijkstra_path(G, 0, 4))\n    [0, 1, 2, 3, 4]\n\n    Find edges of shortest path in Multigraph\n\n    >>> G = nx.MultiDiGraph()\n    >>> G.add_weighted_edges_from([(1, 2, 0.75), (1, 2, 0.5), (2, 3, 0.5), (1, 3, 1.5)])\n    >>> nodes = nx.dijkstra_path(G, 1, 3)\n    >>> edges = nx.utils.pairwise(nodes)\n    >>> list((u, v, min(G[u][v], key=lambda k: G[u][v][k].get(\'weight\', 1))) for u, v in edges)\n    [(1, 2, 1), (2, 3, 0)]\n\n    Notes\n    -----\n    Edge weight attributes must be numerical.\n    Distances are calculated as sums of weighted edges traversed.\n\n    The weight function can be used to hide edges by returning None.\n    So ``weight = lambda u, v, d: 1 if d[\'color\']=="red" else None``\n    will find the shortest red path.\n\n    The weight function can be used to include node weights.\n\n    >>> def func(u, v, d):\n    ...     node_u_wt = G.nodes[u].get("node_weight", 1)\n    ...     node_v_wt = G.nodes[v].get("node_weight", 1)\n    ...     edge_wt = d.get("weight", 1)\n    ...     return node_u_wt / 2 + node_v_wt / 2 + edge_wt\n\n    In this example we take the average of start and end node\n    weights of an edge and add it to the weight of the edge.\n\n    The function :func:`single_source_dijkstra` computes both\n    path and length-of-path if you need both, use that.\n\n    See Also\n    --------\n    bidirectional_dijkstra\n    bellman_ford_path\n    single_source_dijkstra\n    '
    (length, path) = single_source_dijkstra(G, source, target=target, weight=weight)
    return path

@nx._dispatch(edge_attrs='weight')
def dijkstra_path_length(G, source, target, weight='weight'):
    if False:
        return 10
    'Returns the shortest weighted path length in G from source to target.\n\n    Uses Dijkstra\'s Method to compute the shortest weighted path length\n    between two nodes in a graph.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n\n    source : node label\n        starting node for path\n\n    target : node label\n        ending node for path\n\n    weight : string or function\n        If this is a string, then edge weights will be accessed via the\n        edge attribute with this key (that is, the weight of the edge\n        joining `u` to `v` will be ``G.edges[u, v][weight]``). If no\n        such edge attribute exists, the weight of the edge is assumed to\n        be one.\n\n        If this is a function, the weight of an edge is the value\n        returned by the function. The function must accept exactly three\n        positional arguments: the two endpoints of an edge and the\n        dictionary of edge attributes for that edge. The function must\n        return a number or None to indicate a hidden edge.\n\n    Returns\n    -------\n    length : number\n        Shortest path length.\n\n    Raises\n    ------\n    NodeNotFound\n        If `source` is not in `G`.\n\n    NetworkXNoPath\n        If no path exists between source and target.\n\n    Examples\n    --------\n    >>> G = nx.path_graph(5)\n    >>> nx.dijkstra_path_length(G, 0, 4)\n    4\n\n    Notes\n    -----\n    Edge weight attributes must be numerical.\n    Distances are calculated as sums of weighted edges traversed.\n\n    The weight function can be used to hide edges by returning None.\n    So ``weight = lambda u, v, d: 1 if d[\'color\']=="red" else None``\n    will find the shortest red path.\n\n    The function :func:`single_source_dijkstra` computes both\n    path and length-of-path if you need both, use that.\n\n    See Also\n    --------\n    bidirectional_dijkstra\n    bellman_ford_path_length\n    single_source_dijkstra\n\n    '
    if source not in G:
        raise nx.NodeNotFound(f'Node {source} not found in graph')
    if source == target:
        return 0
    weight = _weight_function(G, weight)
    length = _dijkstra(G, source, weight, target=target)
    try:
        return length[target]
    except KeyError as err:
        raise nx.NetworkXNoPath(f'Node {target} not reachable from {source}') from err

@nx._dispatch(edge_attrs='weight')
def single_source_dijkstra_path(G, source, cutoff=None, weight='weight'):
    if False:
        while True:
            i = 10
    'Find shortest weighted paths in G from a source node.\n\n    Compute shortest path between source and all other reachable\n    nodes for a weighted graph.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n\n    source : node\n        Starting node for path.\n\n    cutoff : integer or float, optional\n        Length (sum of edge weights) at which the search is stopped.\n        If cutoff is provided, only return paths with summed weight <= cutoff.\n\n    weight : string or function\n        If this is a string, then edge weights will be accessed via the\n        edge attribute with this key (that is, the weight of the edge\n        joining `u` to `v` will be ``G.edges[u, v][weight]``). If no\n        such edge attribute exists, the weight of the edge is assumed to\n        be one.\n\n        If this is a function, the weight of an edge is the value\n        returned by the function. The function must accept exactly three\n        positional arguments: the two endpoints of an edge and the\n        dictionary of edge attributes for that edge. The function must\n        return a number or None to indicate a hidden edge.\n\n    Returns\n    -------\n    paths : dictionary\n        Dictionary of shortest path lengths keyed by target.\n\n    Raises\n    ------\n    NodeNotFound\n        If `source` is not in `G`.\n\n    Examples\n    --------\n    >>> G = nx.path_graph(5)\n    >>> path = nx.single_source_dijkstra_path(G, 0)\n    >>> path[4]\n    [0, 1, 2, 3, 4]\n\n    Notes\n    -----\n    Edge weight attributes must be numerical.\n    Distances are calculated as sums of weighted edges traversed.\n\n    The weight function can be used to hide edges by returning None.\n    So ``weight = lambda u, v, d: 1 if d[\'color\']=="red" else None``\n    will find the shortest red path.\n\n    See Also\n    --------\n    single_source_dijkstra, single_source_bellman_ford\n\n    '
    return multi_source_dijkstra_path(G, {source}, cutoff=cutoff, weight=weight)

@nx._dispatch(edge_attrs='weight')
def single_source_dijkstra_path_length(G, source, cutoff=None, weight='weight'):
    if False:
        for i in range(10):
            print('nop')
    'Find shortest weighted path lengths in G from a source node.\n\n    Compute the shortest path length between source and all other\n    reachable nodes for a weighted graph.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n\n    source : node label\n        Starting node for path\n\n    cutoff : integer or float, optional\n        Length (sum of edge weights) at which the search is stopped.\n        If cutoff is provided, only return paths with summed weight <= cutoff.\n\n    weight : string or function\n        If this is a string, then edge weights will be accessed via the\n        edge attribute with this key (that is, the weight of the edge\n        joining `u` to `v` will be ``G.edges[u, v][weight]``). If no\n        such edge attribute exists, the weight of the edge is assumed to\n        be one.\n\n        If this is a function, the weight of an edge is the value\n        returned by the function. The function must accept exactly three\n        positional arguments: the two endpoints of an edge and the\n        dictionary of edge attributes for that edge. The function must\n        return a number or None to indicate a hidden edge.\n\n    Returns\n    -------\n    length : dict\n        Dict keyed by node to shortest path length from source.\n\n    Raises\n    ------\n    NodeNotFound\n        If `source` is not in `G`.\n\n    Examples\n    --------\n    >>> G = nx.path_graph(5)\n    >>> length = nx.single_source_dijkstra_path_length(G, 0)\n    >>> length[4]\n    4\n    >>> for node in [0, 1, 2, 3, 4]:\n    ...     print(f"{node}: {length[node]}")\n    0: 0\n    1: 1\n    2: 2\n    3: 3\n    4: 4\n\n    Notes\n    -----\n    Edge weight attributes must be numerical.\n    Distances are calculated as sums of weighted edges traversed.\n\n    The weight function can be used to hide edges by returning None.\n    So ``weight = lambda u, v, d: 1 if d[\'color\']=="red" else None``\n    will find the shortest red path.\n\n    See Also\n    --------\n    single_source_dijkstra, single_source_bellman_ford_path_length\n\n    '
    return multi_source_dijkstra_path_length(G, {source}, cutoff=cutoff, weight=weight)

@nx._dispatch(edge_attrs='weight')
def single_source_dijkstra(G, source, target=None, cutoff=None, weight='weight'):
    if False:
        print('Hello World!')
    'Find shortest weighted paths and lengths from a source node.\n\n    Compute the shortest path length between source and all other\n    reachable nodes for a weighted graph.\n\n    Uses Dijkstra\'s algorithm to compute shortest paths and lengths\n    between a source and all other reachable nodes in a weighted graph.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n\n    source : node label\n        Starting node for path\n\n    target : node label, optional\n        Ending node for path\n\n    cutoff : integer or float, optional\n        Length (sum of edge weights) at which the search is stopped.\n        If cutoff is provided, only return paths with summed weight <= cutoff.\n\n\n    weight : string or function\n        If this is a string, then edge weights will be accessed via the\n        edge attribute with this key (that is, the weight of the edge\n        joining `u` to `v` will be ``G.edges[u, v][weight]``). If no\n        such edge attribute exists, the weight of the edge is assumed to\n        be one.\n\n        If this is a function, the weight of an edge is the value\n        returned by the function. The function must accept exactly three\n        positional arguments: the two endpoints of an edge and the\n        dictionary of edge attributes for that edge. The function must\n        return a number or None to indicate a hidden edge.\n\n    Returns\n    -------\n    distance, path : pair of dictionaries, or numeric and list.\n        If target is None, paths and lengths to all nodes are computed.\n        The return value is a tuple of two dictionaries keyed by target nodes.\n        The first dictionary stores distance to each target node.\n        The second stores the path to each target node.\n        If target is not None, returns a tuple (distance, path), where\n        distance is the distance from source to target and path is a list\n        representing the path from source to target.\n\n    Raises\n    ------\n    NodeNotFound\n        If `source` is not in `G`.\n\n    Examples\n    --------\n    >>> G = nx.path_graph(5)\n    >>> length, path = nx.single_source_dijkstra(G, 0)\n    >>> length[4]\n    4\n    >>> for node in [0, 1, 2, 3, 4]:\n    ...     print(f"{node}: {length[node]}")\n    0: 0\n    1: 1\n    2: 2\n    3: 3\n    4: 4\n    >>> path[4]\n    [0, 1, 2, 3, 4]\n    >>> length, path = nx.single_source_dijkstra(G, 0, 1)\n    >>> length\n    1\n    >>> path\n    [0, 1]\n\n    Notes\n    -----\n    Edge weight attributes must be numerical.\n    Distances are calculated as sums of weighted edges traversed.\n\n    The weight function can be used to hide edges by returning None.\n    So ``weight = lambda u, v, d: 1 if d[\'color\']=="red" else None``\n    will find the shortest red path.\n\n    Based on the Python cookbook recipe (119466) at\n    https://code.activestate.com/recipes/119466/\n\n    This algorithm is not guaranteed to work if edge weights\n    are negative or are floating point numbers\n    (overflows and roundoff errors can cause problems).\n\n    See Also\n    --------\n    single_source_dijkstra_path\n    single_source_dijkstra_path_length\n    single_source_bellman_ford\n    '
    return multi_source_dijkstra(G, {source}, cutoff=cutoff, target=target, weight=weight)

@nx._dispatch(edge_attrs='weight')
def multi_source_dijkstra_path(G, sources, cutoff=None, weight='weight'):
    if False:
        print('Hello World!')
    'Find shortest weighted paths in G from a given set of source\n    nodes.\n\n    Compute shortest path between any of the source nodes and all other\n    reachable nodes for a weighted graph.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n\n    sources : non-empty set of nodes\n        Starting nodes for paths. If this is just a set containing a\n        single node, then all paths computed by this function will start\n        from that node. If there are two or more nodes in the set, the\n        computed paths may begin from any one of the start nodes.\n\n    cutoff : integer or float, optional\n        Length (sum of edge weights) at which the search is stopped.\n        If cutoff is provided, only return paths with summed weight <= cutoff.\n\n    weight : string or function\n        If this is a string, then edge weights will be accessed via the\n        edge attribute with this key (that is, the weight of the edge\n        joining `u` to `v` will be ``G.edges[u, v][weight]``). If no\n        such edge attribute exists, the weight of the edge is assumed to\n        be one.\n\n        If this is a function, the weight of an edge is the value\n        returned by the function. The function must accept exactly three\n        positional arguments: the two endpoints of an edge and the\n        dictionary of edge attributes for that edge. The function must\n        return a number or None to indicate a hidden edge.\n\n    Returns\n    -------\n    paths : dictionary\n        Dictionary of shortest paths keyed by target.\n\n    Examples\n    --------\n    >>> G = nx.path_graph(5)\n    >>> path = nx.multi_source_dijkstra_path(G, {0, 4})\n    >>> path[1]\n    [0, 1]\n    >>> path[3]\n    [4, 3]\n\n    Notes\n    -----\n    Edge weight attributes must be numerical.\n    Distances are calculated as sums of weighted edges traversed.\n\n    The weight function can be used to hide edges by returning None.\n    So ``weight = lambda u, v, d: 1 if d[\'color\']=="red" else None``\n    will find the shortest red path.\n\n    Raises\n    ------\n    ValueError\n        If `sources` is empty.\n    NodeNotFound\n        If any of `sources` is not in `G`.\n\n    See Also\n    --------\n    multi_source_dijkstra, multi_source_bellman_ford\n\n    '
    (length, path) = multi_source_dijkstra(G, sources, cutoff=cutoff, weight=weight)
    return path

@nx._dispatch(edge_attrs='weight')
def multi_source_dijkstra_path_length(G, sources, cutoff=None, weight='weight'):
    if False:
        for i in range(10):
            print('nop')
    'Find shortest weighted path lengths in G from a given set of\n    source nodes.\n\n    Compute the shortest path length between any of the source nodes and\n    all other reachable nodes for a weighted graph.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n\n    sources : non-empty set of nodes\n        Starting nodes for paths. If this is just a set containing a\n        single node, then all paths computed by this function will start\n        from that node. If there are two or more nodes in the set, the\n        computed paths may begin from any one of the start nodes.\n\n    cutoff : integer or float, optional\n        Length (sum of edge weights) at which the search is stopped.\n        If cutoff is provided, only return paths with summed weight <= cutoff.\n\n    weight : string or function\n        If this is a string, then edge weights will be accessed via the\n        edge attribute with this key (that is, the weight of the edge\n        joining `u` to `v` will be ``G.edges[u, v][weight]``). If no\n        such edge attribute exists, the weight of the edge is assumed to\n        be one.\n\n        If this is a function, the weight of an edge is the value\n        returned by the function. The function must accept exactly three\n        positional arguments: the two endpoints of an edge and the\n        dictionary of edge attributes for that edge. The function must\n        return a number or None to indicate a hidden edge.\n\n    Returns\n    -------\n    length : dict\n        Dict keyed by node to shortest path length to nearest source.\n\n    Examples\n    --------\n    >>> G = nx.path_graph(5)\n    >>> length = nx.multi_source_dijkstra_path_length(G, {0, 4})\n    >>> for node in [0, 1, 2, 3, 4]:\n    ...     print(f"{node}: {length[node]}")\n    0: 0\n    1: 1\n    2: 2\n    3: 1\n    4: 0\n\n    Notes\n    -----\n    Edge weight attributes must be numerical.\n    Distances are calculated as sums of weighted edges traversed.\n\n    The weight function can be used to hide edges by returning None.\n    So ``weight = lambda u, v, d: 1 if d[\'color\']=="red" else None``\n    will find the shortest red path.\n\n    Raises\n    ------\n    ValueError\n        If `sources` is empty.\n    NodeNotFound\n        If any of `sources` is not in `G`.\n\n    See Also\n    --------\n    multi_source_dijkstra\n\n    '
    if not sources:
        raise ValueError('sources must not be empty')
    for s in sources:
        if s not in G:
            raise nx.NodeNotFound(f'Node {s} not found in graph')
    weight = _weight_function(G, weight)
    return _dijkstra_multisource(G, sources, weight, cutoff=cutoff)

@nx._dispatch(edge_attrs='weight')
def multi_source_dijkstra(G, sources, target=None, cutoff=None, weight='weight'):
    if False:
        while True:
            i = 10
    'Find shortest weighted paths and lengths from a given set of\n    source nodes.\n\n    Uses Dijkstra\'s algorithm to compute the shortest paths and lengths\n    between one of the source nodes and the given `target`, or all other\n    reachable nodes if not specified, for a weighted graph.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n\n    sources : non-empty set of nodes\n        Starting nodes for paths. If this is just a set containing a\n        single node, then all paths computed by this function will start\n        from that node. If there are two or more nodes in the set, the\n        computed paths may begin from any one of the start nodes.\n\n    target : node label, optional\n        Ending node for path\n\n    cutoff : integer or float, optional\n        Length (sum of edge weights) at which the search is stopped.\n        If cutoff is provided, only return paths with summed weight <= cutoff.\n\n    weight : string or function\n        If this is a string, then edge weights will be accessed via the\n        edge attribute with this key (that is, the weight of the edge\n        joining `u` to `v` will be ``G.edges[u, v][weight]``). If no\n        such edge attribute exists, the weight of the edge is assumed to\n        be one.\n\n        If this is a function, the weight of an edge is the value\n        returned by the function. The function must accept exactly three\n        positional arguments: the two endpoints of an edge and the\n        dictionary of edge attributes for that edge. The function must\n        return a number or None to indicate a hidden edge.\n\n    Returns\n    -------\n    distance, path : pair of dictionaries, or numeric and list\n        If target is None, returns a tuple of two dictionaries keyed by node.\n        The first dictionary stores distance from one of the source nodes.\n        The second stores the path from one of the sources to that node.\n        If target is not None, returns a tuple of (distance, path) where\n        distance is the distance from source to target and path is a list\n        representing the path from source to target.\n\n    Examples\n    --------\n    >>> G = nx.path_graph(5)\n    >>> length, path = nx.multi_source_dijkstra(G, {0, 4})\n    >>> for node in [0, 1, 2, 3, 4]:\n    ...     print(f"{node}: {length[node]}")\n    0: 0\n    1: 1\n    2: 2\n    3: 1\n    4: 0\n    >>> path[1]\n    [0, 1]\n    >>> path[3]\n    [4, 3]\n\n    >>> length, path = nx.multi_source_dijkstra(G, {0, 4}, 1)\n    >>> length\n    1\n    >>> path\n    [0, 1]\n\n    Notes\n    -----\n    Edge weight attributes must be numerical.\n    Distances are calculated as sums of weighted edges traversed.\n\n    The weight function can be used to hide edges by returning None.\n    So ``weight = lambda u, v, d: 1 if d[\'color\']=="red" else None``\n    will find the shortest red path.\n\n    Based on the Python cookbook recipe (119466) at\n    https://code.activestate.com/recipes/119466/\n\n    This algorithm is not guaranteed to work if edge weights\n    are negative or are floating point numbers\n    (overflows and roundoff errors can cause problems).\n\n    Raises\n    ------\n    ValueError\n        If `sources` is empty.\n    NodeNotFound\n        If any of `sources` is not in `G`.\n\n    See Also\n    --------\n    multi_source_dijkstra_path\n    multi_source_dijkstra_path_length\n\n    '
    if not sources:
        raise ValueError('sources must not be empty')
    for s in sources:
        if s not in G:
            raise nx.NodeNotFound(f'Node {s} not found in graph')
    if target in sources:
        return (0, [target])
    weight = _weight_function(G, weight)
    paths = {source: [source] for source in sources}
    dist = _dijkstra_multisource(G, sources, weight, paths=paths, cutoff=cutoff, target=target)
    if target is None:
        return (dist, paths)
    try:
        return (dist[target], paths[target])
    except KeyError as err:
        raise nx.NetworkXNoPath(f'No path to {target}.') from err

def _dijkstra(G, source, weight, pred=None, paths=None, cutoff=None, target=None):
    if False:
        i = 10
        return i + 15
    "Uses Dijkstra's algorithm to find shortest weighted paths from a\n    single source.\n\n    This is a convenience function for :func:`_dijkstra_multisource`\n    with all the arguments the same, except the keyword argument\n    `sources` set to ``[source]``.\n\n    "
    return _dijkstra_multisource(G, [source], weight, pred=pred, paths=paths, cutoff=cutoff, target=target)

def _dijkstra_multisource(G, sources, weight, pred=None, paths=None, cutoff=None, target=None):
    if False:
        while True:
            i = 10
    "Uses Dijkstra's algorithm to find shortest weighted paths\n\n    Parameters\n    ----------\n    G : NetworkX graph\n\n    sources : non-empty iterable of nodes\n        Starting nodes for paths. If this is just an iterable containing\n        a single node, then all paths computed by this function will\n        start from that node. If there are two or more nodes in this\n        iterable, the computed paths may begin from any one of the start\n        nodes.\n\n    weight: function\n        Function with (u, v, data) input that returns that edge's weight\n        or None to indicate a hidden edge\n\n    pred: dict of lists, optional(default=None)\n        dict to store a list of predecessors keyed by that node\n        If None, predecessors are not stored.\n\n    paths: dict, optional (default=None)\n        dict to store the path list from source to each node, keyed by node.\n        If None, paths are not stored.\n\n    target : node label, optional\n        Ending node for path. Search is halted when target is found.\n\n    cutoff : integer or float, optional\n        Length (sum of edge weights) at which the search is stopped.\n        If cutoff is provided, only return paths with summed weight <= cutoff.\n\n    Returns\n    -------\n    distance : dictionary\n        A mapping from node to shortest distance to that node from one\n        of the source nodes.\n\n    Raises\n    ------\n    NodeNotFound\n        If any of `sources` is not in `G`.\n\n    Notes\n    -----\n    The optional predecessor and path dictionaries can be accessed by\n    the caller through the original pred and paths objects passed\n    as arguments. No need to explicitly return pred or paths.\n\n    "
    G_succ = G._adj
    push = heappush
    pop = heappop
    dist = {}
    seen = {}
    c = count()
    fringe = []
    for source in sources:
        seen[source] = 0
        push(fringe, (0, next(c), source))
    while fringe:
        (d, _, v) = pop(fringe)
        if v in dist:
            continue
        dist[v] = d
        if v == target:
            break
        for (u, e) in G_succ[v].items():
            cost = weight(v, u, e)
            if cost is None:
                continue
            vu_dist = dist[v] + cost
            if cutoff is not None:
                if vu_dist > cutoff:
                    continue
            if u in dist:
                u_dist = dist[u]
                if vu_dist < u_dist:
                    raise ValueError('Contradictory paths found:', 'negative weights?')
                elif pred is not None and vu_dist == u_dist:
                    pred[u].append(v)
            elif u not in seen or vu_dist < seen[u]:
                seen[u] = vu_dist
                push(fringe, (vu_dist, next(c), u))
                if paths is not None:
                    paths[u] = paths[v] + [u]
                if pred is not None:
                    pred[u] = [v]
            elif vu_dist == seen[u]:
                if pred is not None:
                    pred[u].append(v)
    return dist

@nx._dispatch(edge_attrs='weight')
def dijkstra_predecessor_and_distance(G, source, cutoff=None, weight='weight'):
    if False:
        return 10
    "Compute weighted shortest path length and predecessors.\n\n    Uses Dijkstra's Method to obtain the shortest weighted paths\n    and return dictionaries of predecessors for each node and\n    distance for each node from the `source`.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n\n    source : node label\n        Starting node for path\n\n    cutoff : integer or float, optional\n        Length (sum of edge weights) at which the search is stopped.\n        If cutoff is provided, only return paths with summed weight <= cutoff.\n\n    weight : string or function\n        If this is a string, then edge weights will be accessed via the\n        edge attribute with this key (that is, the weight of the edge\n        joining `u` to `v` will be ``G.edges[u, v][weight]``). If no\n        such edge attribute exists, the weight of the edge is assumed to\n        be one.\n\n        If this is a function, the weight of an edge is the value\n        returned by the function. The function must accept exactly three\n        positional arguments: the two endpoints of an edge and the\n        dictionary of edge attributes for that edge. The function must\n        return a number or None to indicate a hidden edge.\n\n    Returns\n    -------\n    pred, distance : dictionaries\n        Returns two dictionaries representing a list of predecessors\n        of a node and the distance to each node.\n\n    Raises\n    ------\n    NodeNotFound\n        If `source` is not in `G`.\n\n    Notes\n    -----\n    Edge weight attributes must be numerical.\n    Distances are calculated as sums of weighted edges traversed.\n\n    The list of predecessors contains more than one element only when\n    there are more than one shortest paths to the key node.\n\n    Examples\n    --------\n    >>> G = nx.path_graph(5, create_using=nx.DiGraph())\n    >>> pred, dist = nx.dijkstra_predecessor_and_distance(G, 0)\n    >>> sorted(pred.items())\n    [(0, []), (1, [0]), (2, [1]), (3, [2]), (4, [3])]\n    >>> sorted(dist.items())\n    [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]\n\n    >>> pred, dist = nx.dijkstra_predecessor_and_distance(G, 0, 1)\n    >>> sorted(pred.items())\n    [(0, []), (1, [0])]\n    >>> sorted(dist.items())\n    [(0, 0), (1, 1)]\n    "
    if source not in G:
        raise nx.NodeNotFound(f'Node {source} is not found in the graph')
    weight = _weight_function(G, weight)
    pred = {source: []}
    return (pred, _dijkstra(G, source, weight, pred=pred, cutoff=cutoff))

@nx._dispatch(edge_attrs='weight')
def all_pairs_dijkstra(G, cutoff=None, weight='weight'):
    if False:
        i = 10
        return i + 15
    'Find shortest weighted paths and lengths between all nodes.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n\n    cutoff : integer or float, optional\n        Length (sum of edge weights) at which the search is stopped.\n        If cutoff is provided, only return paths with summed weight <= cutoff.\n\n    weight : string or function\n        If this is a string, then edge weights will be accessed via the\n        edge attribute with this key (that is, the weight of the edge\n        joining `u` to `v` will be ``G.edge[u][v][weight]``). If no\n        such edge attribute exists, the weight of the edge is assumed to\n        be one.\n\n        If this is a function, the weight of an edge is the value\n        returned by the function. The function must accept exactly three\n        positional arguments: the two endpoints of an edge and the\n        dictionary of edge attributes for that edge. The function must\n        return a number or None to indicate a hidden edge.\n\n    Yields\n    ------\n    (node, (distance, path)) : (node obj, (dict, dict))\n        Each source node has two associated dicts. The first holds distance\n        keyed by target and the second holds paths keyed by target.\n        (See single_source_dijkstra for the source/target node terminology.)\n        If desired you can apply `dict()` to this function to create a dict\n        keyed by source node to the two dicts.\n\n    Examples\n    --------\n    >>> G = nx.path_graph(5)\n    >>> len_path = dict(nx.all_pairs_dijkstra(G))\n    >>> len_path[3][0][1]\n    2\n    >>> for node in [0, 1, 2, 3, 4]:\n    ...     print(f"3 - {node}: {len_path[3][0][node]}")\n    3 - 0: 3\n    3 - 1: 2\n    3 - 2: 1\n    3 - 3: 0\n    3 - 4: 1\n    >>> len_path[3][1][1]\n    [3, 2, 1]\n    >>> for n, (dist, path) in nx.all_pairs_dijkstra(G):\n    ...     print(path[1])\n    [0, 1]\n    [1]\n    [2, 1]\n    [3, 2, 1]\n    [4, 3, 2, 1]\n\n    Notes\n    -----\n    Edge weight attributes must be numerical.\n    Distances are calculated as sums of weighted edges traversed.\n\n    The yielded dicts only have keys for reachable nodes.\n    '
    for n in G:
        (dist, path) = single_source_dijkstra(G, n, cutoff=cutoff, weight=weight)
        yield (n, (dist, path))

@nx._dispatch(edge_attrs='weight')
def all_pairs_dijkstra_path_length(G, cutoff=None, weight='weight'):
    if False:
        while True:
            i = 10
    'Compute shortest path lengths between all nodes in a weighted graph.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n\n    cutoff : integer or float, optional\n        Length (sum of edge weights) at which the search is stopped.\n        If cutoff is provided, only return paths with summed weight <= cutoff.\n\n    weight : string or function\n        If this is a string, then edge weights will be accessed via the\n        edge attribute with this key (that is, the weight of the edge\n        joining `u` to `v` will be ``G.edges[u, v][weight]``). If no\n        such edge attribute exists, the weight of the edge is assumed to\n        be one.\n\n        If this is a function, the weight of an edge is the value\n        returned by the function. The function must accept exactly three\n        positional arguments: the two endpoints of an edge and the\n        dictionary of edge attributes for that edge. The function must\n        return a number or None to indicate a hidden edge.\n\n    Returns\n    -------\n    distance : iterator\n        (source, dictionary) iterator with dictionary keyed by target and\n        shortest path length as the key value.\n\n    Examples\n    --------\n    >>> G = nx.path_graph(5)\n    >>> length = dict(nx.all_pairs_dijkstra_path_length(G))\n    >>> for node in [0, 1, 2, 3, 4]:\n    ...     print(f"1 - {node}: {length[1][node]}")\n    1 - 0: 1\n    1 - 1: 0\n    1 - 2: 1\n    1 - 3: 2\n    1 - 4: 3\n    >>> length[3][2]\n    1\n    >>> length[2][2]\n    0\n\n    Notes\n    -----\n    Edge weight attributes must be numerical.\n    Distances are calculated as sums of weighted edges traversed.\n\n    The dictionary returned only has keys for reachable node pairs.\n    '
    length = single_source_dijkstra_path_length
    for n in G:
        yield (n, length(G, n, cutoff=cutoff, weight=weight))

@nx._dispatch(edge_attrs='weight')
def all_pairs_dijkstra_path(G, cutoff=None, weight='weight'):
    if False:
        i = 10
        return i + 15
    'Compute shortest paths between all nodes in a weighted graph.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n\n    cutoff : integer or float, optional\n        Length (sum of edge weights) at which the search is stopped.\n        If cutoff is provided, only return paths with summed weight <= cutoff.\n\n    weight : string or function\n        If this is a string, then edge weights will be accessed via the\n        edge attribute with this key (that is, the weight of the edge\n        joining `u` to `v` will be ``G.edges[u, v][weight]``). If no\n        such edge attribute exists, the weight of the edge is assumed to\n        be one.\n\n        If this is a function, the weight of an edge is the value\n        returned by the function. The function must accept exactly three\n        positional arguments: the two endpoints of an edge and the\n        dictionary of edge attributes for that edge. The function must\n        return a number or None to indicate a hidden edge.\n\n    Returns\n    -------\n    paths : iterator\n        (source, dictionary) iterator with dictionary keyed by target and\n        shortest path as the key value.\n\n    Examples\n    --------\n    >>> G = nx.path_graph(5)\n    >>> path = dict(nx.all_pairs_dijkstra_path(G))\n    >>> path[0][4]\n    [0, 1, 2, 3, 4]\n\n    Notes\n    -----\n    Edge weight attributes must be numerical.\n    Distances are calculated as sums of weighted edges traversed.\n\n    See Also\n    --------\n    floyd_warshall, all_pairs_bellman_ford_path\n\n    '
    path = single_source_dijkstra_path
    for n in G:
        yield (n, path(G, n, cutoff=cutoff, weight=weight))

@nx._dispatch(edge_attrs='weight')
def bellman_ford_predecessor_and_distance(G, source, target=None, weight='weight', heuristic=False):
    if False:
        i = 10
        return i + 15
    'Compute shortest path lengths and predecessors on shortest paths\n    in weighted graphs.\n\n    The algorithm has a running time of $O(mn)$ where $n$ is the number of\n    nodes and $m$ is the number of edges.  It is slower than Dijkstra but\n    can handle negative edge weights.\n\n    If a negative cycle is detected, you can use :func:`find_negative_cycle`\n    to return the cycle and examine it. Shortest paths are not defined when\n    a negative cycle exists because once reached, the path can cycle forever\n    to build up arbitrarily low weights.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n        The algorithm works for all types of graphs, including directed\n        graphs and multigraphs.\n\n    source: node label\n        Starting node for path\n\n    target : node label, optional\n        Ending node for path\n\n    weight : string or function\n        If this is a string, then edge weights will be accessed via the\n        edge attribute with this key (that is, the weight of the edge\n        joining `u` to `v` will be ``G.edges[u, v][weight]``). If no\n        such edge attribute exists, the weight of the edge is assumed to\n        be one.\n\n        If this is a function, the weight of an edge is the value\n        returned by the function. The function must accept exactly three\n        positional arguments: the two endpoints of an edge and the\n        dictionary of edge attributes for that edge. The function must\n        return a number.\n\n    heuristic : bool\n        Determines whether to use a heuristic to early detect negative\n        cycles at a hopefully negligible cost.\n\n    Returns\n    -------\n    pred, dist : dictionaries\n        Returns two dictionaries keyed by node to predecessor in the\n        path and to the distance from the source respectively.\n\n    Raises\n    ------\n    NodeNotFound\n        If `source` is not in `G`.\n\n    NetworkXUnbounded\n        If the (di)graph contains a negative (di)cycle, the\n        algorithm raises an exception to indicate the presence of the\n        negative (di)cycle.  Note: any negative weight edge in an\n        undirected graph is a negative cycle.\n\n    Examples\n    --------\n    >>> G = nx.path_graph(5, create_using=nx.DiGraph())\n    >>> pred, dist = nx.bellman_ford_predecessor_and_distance(G, 0)\n    >>> sorted(pred.items())\n    [(0, []), (1, [0]), (2, [1]), (3, [2]), (4, [3])]\n    >>> sorted(dist.items())\n    [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]\n\n    >>> pred, dist = nx.bellman_ford_predecessor_and_distance(G, 0, 1)\n    >>> sorted(pred.items())\n    [(0, []), (1, [0]), (2, [1]), (3, [2]), (4, [3])]\n    >>> sorted(dist.items())\n    [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]\n\n    >>> G = nx.cycle_graph(5, create_using=nx.DiGraph())\n    >>> G[1][2]["weight"] = -7\n    >>> nx.bellman_ford_predecessor_and_distance(G, 0)\n    Traceback (most recent call last):\n        ...\n    networkx.exception.NetworkXUnbounded: Negative cycle detected.\n\n    See Also\n    --------\n    find_negative_cycle\n\n    Notes\n    -----\n    Edge weight attributes must be numerical.\n    Distances are calculated as sums of weighted edges traversed.\n\n    The dictionaries returned only have keys for nodes reachable from\n    the source.\n\n    In the case where the (di)graph is not connected, if a component\n    not containing the source contains a negative (di)cycle, it\n    will not be detected.\n\n    In NetworkX v2.1 and prior, the source node had predecessor `[None]`.\n    In NetworkX v2.2 this changed to the source node having predecessor `[]`\n    '
    if source not in G:
        raise nx.NodeNotFound(f'Node {source} is not found in the graph')
    weight = _weight_function(G, weight)
    if G.is_multigraph():
        if any((weight(u, v, {k: d}) < 0 for (u, v, k, d) in nx.selfloop_edges(G, keys=True, data=True))):
            raise nx.NetworkXUnbounded('Negative cycle detected.')
    elif any((weight(u, v, d) < 0 for (u, v, d) in nx.selfloop_edges(G, data=True))):
        raise nx.NetworkXUnbounded('Negative cycle detected.')
    dist = {source: 0}
    pred = {source: []}
    if len(G) == 1:
        return (pred, dist)
    weight = _weight_function(G, weight)
    dist = _bellman_ford(G, [source], weight, pred=pred, dist=dist, target=target, heuristic=heuristic)
    return (pred, dist)

def _bellman_ford(G, source, weight, pred=None, paths=None, dist=None, target=None, heuristic=True):
    if False:
        return 10
    'Calls relaxation loop for Bellman–Ford algorithm and builds paths\n\n    This is an implementation of the SPFA variant.\n    See https://en.wikipedia.org/wiki/Shortest_Path_Faster_Algorithm\n\n    Parameters\n    ----------\n    G : NetworkX graph\n\n    source: list\n        List of source nodes. The shortest path from any of the source\n        nodes will be found if multiple sources are provided.\n\n    weight : function\n        The weight of an edge is the value returned by the function. The\n        function must accept exactly three positional arguments: the two\n        endpoints of an edge and the dictionary of edge attributes for\n        that edge. The function must return a number.\n\n    pred: dict of lists, optional (default=None)\n        dict to store a list of predecessors keyed by that node\n        If None, predecessors are not stored\n\n    paths: dict, optional (default=None)\n        dict to store the path list from source to each node, keyed by node\n        If None, paths are not stored\n\n    dist: dict, optional (default=None)\n        dict to store distance from source to the keyed node\n        If None, returned dist dict contents default to 0 for every node in the\n        source list\n\n    target: node label, optional\n        Ending node for path. Path lengths to other destinations may (and\n        probably will) be incorrect.\n\n    heuristic : bool\n        Determines whether to use a heuristic to early detect negative\n        cycles at a hopefully negligible cost.\n\n    Returns\n    -------\n    dist : dict\n        Returns a dict keyed by node to the distance from the source.\n        Dicts for paths and pred are in the mutated input dicts by those names.\n\n    Raises\n    ------\n    NodeNotFound\n        If any of `source` is not in `G`.\n\n    NetworkXUnbounded\n        If the (di)graph contains a negative (di)cycle, the\n        algorithm raises an exception to indicate the presence of the\n        negative (di)cycle.  Note: any negative weight edge in an\n        undirected graph is a negative cycle\n    '
    if pred is None:
        pred = {v: [] for v in source}
    if dist is None:
        dist = {v: 0 for v in source}
    negative_cycle_found = _inner_bellman_ford(G, source, weight, pred, dist, heuristic)
    if negative_cycle_found is not None:
        raise nx.NetworkXUnbounded('Negative cycle detected.')
    if paths is not None:
        sources = set(source)
        dsts = [target] if target is not None else pred
        for dst in dsts:
            gen = _build_paths_from_predecessors(sources, dst, pred)
            paths[dst] = next(gen)
    return dist

def _inner_bellman_ford(G, sources, weight, pred, dist=None, heuristic=True):
    if False:
        for i in range(10):
            print('nop')
    'Inner Relaxation loop for Bellman–Ford algorithm.\n\n    This is an implementation of the SPFA variant.\n    See https://en.wikipedia.org/wiki/Shortest_Path_Faster_Algorithm\n\n    Parameters\n    ----------\n    G : NetworkX graph\n\n    source: list\n        List of source nodes. The shortest path from any of the source\n        nodes will be found if multiple sources are provided.\n\n    weight : function\n        The weight of an edge is the value returned by the function. The\n        function must accept exactly three positional arguments: the two\n        endpoints of an edge and the dictionary of edge attributes for\n        that edge. The function must return a number.\n\n    pred: dict of lists\n        dict to store a list of predecessors keyed by that node\n\n    dist: dict, optional (default=None)\n        dict to store distance from source to the keyed node\n        If None, returned dist dict contents default to 0 for every node in the\n        source list\n\n    heuristic : bool\n        Determines whether to use a heuristic to early detect negative\n        cycles at a hopefully negligible cost.\n\n    Returns\n    -------\n    node or None\n        Return a node `v` where processing discovered a negative cycle.\n        If no negative cycle found, return None.\n\n    Raises\n    ------\n    NodeNotFound\n        If any of `source` is not in `G`.\n    '
    for s in sources:
        if s not in G:
            raise nx.NodeNotFound(f'Source {s} not in G')
    if pred is None:
        pred = {v: [] for v in sources}
    if dist is None:
        dist = {v: 0 for v in sources}
    nonexistent_edge = (None, None)
    pred_edge = {v: None for v in sources}
    recent_update = {v: nonexistent_edge for v in sources}
    G_succ = G._adj
    inf = float('inf')
    n = len(G)
    count = {}
    q = deque(sources)
    in_q = set(sources)
    while q:
        u = q.popleft()
        in_q.remove(u)
        if all((pred_u not in in_q for pred_u in pred[u])):
            dist_u = dist[u]
            for (v, e) in G_succ[u].items():
                dist_v = dist_u + weight(u, v, e)
                if dist_v < dist.get(v, inf):
                    if heuristic:
                        if v in recent_update[u]:
                            pred[v].append(u)
                            return v
                        if v in pred_edge and pred_edge[v] == u:
                            recent_update[v] = recent_update[u]
                        else:
                            recent_update[v] = (u, v)
                    if v not in in_q:
                        q.append(v)
                        in_q.add(v)
                        count_v = count.get(v, 0) + 1
                        if count_v == n:
                            return v
                        count[v] = count_v
                    dist[v] = dist_v
                    pred[v] = [u]
                    pred_edge[v] = u
                elif dist.get(v) is not None and dist_v == dist.get(v):
                    pred[v].append(u)
    return None

@nx._dispatch(edge_attrs='weight')
def bellman_ford_path(G, source, target, weight='weight'):
    if False:
        for i in range(10):
            print('nop')
    'Returns the shortest path from source to target in a weighted graph G.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n\n    source : node\n        Starting node\n\n    target : node\n        Ending node\n\n    weight : string or function (default="weight")\n        If this is a string, then edge weights will be accessed via the\n        edge attribute with this key (that is, the weight of the edge\n        joining `u` to `v` will be ``G.edges[u, v][weight]``). If no\n        such edge attribute exists, the weight of the edge is assumed to\n        be one.\n\n        If this is a function, the weight of an edge is the value\n        returned by the function. The function must accept exactly three\n        positional arguments: the two endpoints of an edge and the\n        dictionary of edge attributes for that edge. The function must\n        return a number.\n\n    Returns\n    -------\n    path : list\n        List of nodes in a shortest path.\n\n    Raises\n    ------\n    NodeNotFound\n        If `source` is not in `G`.\n\n    NetworkXNoPath\n        If no path exists between source and target.\n\n    Examples\n    --------\n    >>> G = nx.path_graph(5)\n    >>> nx.bellman_ford_path(G, 0, 4)\n    [0, 1, 2, 3, 4]\n\n    Notes\n    -----\n    Edge weight attributes must be numerical.\n    Distances are calculated as sums of weighted edges traversed.\n\n    See Also\n    --------\n    dijkstra_path, bellman_ford_path_length\n    '
    (length, path) = single_source_bellman_ford(G, source, target=target, weight=weight)
    return path

@nx._dispatch(edge_attrs='weight')
def bellman_ford_path_length(G, source, target, weight='weight'):
    if False:
        while True:
            i = 10
    'Returns the shortest path length from source to target\n    in a weighted graph.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n\n    source : node label\n        starting node for path\n\n    target : node label\n        ending node for path\n\n    weight : string or function (default="weight")\n        If this is a string, then edge weights will be accessed via the\n        edge attribute with this key (that is, the weight of the edge\n        joining `u` to `v` will be ``G.edges[u, v][weight]``). If no\n        such edge attribute exists, the weight of the edge is assumed to\n        be one.\n\n        If this is a function, the weight of an edge is the value\n        returned by the function. The function must accept exactly three\n        positional arguments: the two endpoints of an edge and the\n        dictionary of edge attributes for that edge. The function must\n        return a number.\n\n    Returns\n    -------\n    length : number\n        Shortest path length.\n\n    Raises\n    ------\n    NodeNotFound\n        If `source` is not in `G`.\n\n    NetworkXNoPath\n        If no path exists between source and target.\n\n    Examples\n    --------\n    >>> G = nx.path_graph(5)\n    >>> nx.bellman_ford_path_length(G, 0, 4)\n    4\n\n    Notes\n    -----\n    Edge weight attributes must be numerical.\n    Distances are calculated as sums of weighted edges traversed.\n\n    See Also\n    --------\n    dijkstra_path_length, bellman_ford_path\n    '
    if source == target:
        if source not in G:
            raise nx.NodeNotFound(f'Node {source} not found in graph')
        return 0
    weight = _weight_function(G, weight)
    length = _bellman_ford(G, [source], weight, target=target)
    try:
        return length[target]
    except KeyError as err:
        raise nx.NetworkXNoPath(f'node {target} not reachable from {source}') from err

@nx._dispatch(edge_attrs='weight')
def single_source_bellman_ford_path(G, source, weight='weight'):
    if False:
        while True:
            i = 10
    'Compute shortest path between source and all other reachable\n    nodes for a weighted graph.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n\n    source : node\n        Starting node for path.\n\n    weight : string or function (default="weight")\n        If this is a string, then edge weights will be accessed via the\n        edge attribute with this key (that is, the weight of the edge\n        joining `u` to `v` will be ``G.edges[u, v][weight]``). If no\n        such edge attribute exists, the weight of the edge is assumed to\n        be one.\n\n        If this is a function, the weight of an edge is the value\n        returned by the function. The function must accept exactly three\n        positional arguments: the two endpoints of an edge and the\n        dictionary of edge attributes for that edge. The function must\n        return a number.\n\n    Returns\n    -------\n    paths : dictionary\n        Dictionary of shortest path lengths keyed by target.\n\n    Raises\n    ------\n    NodeNotFound\n        If `source` is not in `G`.\n\n    Examples\n    --------\n    >>> G = nx.path_graph(5)\n    >>> path = nx.single_source_bellman_ford_path(G, 0)\n    >>> path[4]\n    [0, 1, 2, 3, 4]\n\n    Notes\n    -----\n    Edge weight attributes must be numerical.\n    Distances are calculated as sums of weighted edges traversed.\n\n    See Also\n    --------\n    single_source_dijkstra, single_source_bellman_ford\n\n    '
    (length, path) = single_source_bellman_ford(G, source, weight=weight)
    return path

@nx._dispatch(edge_attrs='weight')
def single_source_bellman_ford_path_length(G, source, weight='weight'):
    if False:
        print('Hello World!')
    'Compute the shortest path length between source and all other\n    reachable nodes for a weighted graph.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n\n    source : node label\n        Starting node for path\n\n    weight : string or function (default="weight")\n        If this is a string, then edge weights will be accessed via the\n        edge attribute with this key (that is, the weight of the edge\n        joining `u` to `v` will be ``G.edges[u, v][weight]``). If no\n        such edge attribute exists, the weight of the edge is assumed to\n        be one.\n\n        If this is a function, the weight of an edge is the value\n        returned by the function. The function must accept exactly three\n        positional arguments: the two endpoints of an edge and the\n        dictionary of edge attributes for that edge. The function must\n        return a number.\n\n    Returns\n    -------\n    length : dictionary\n        Dictionary of shortest path length keyed by target\n\n    Raises\n    ------\n    NodeNotFound\n        If `source` is not in `G`.\n\n    Examples\n    --------\n    >>> G = nx.path_graph(5)\n    >>> length = nx.single_source_bellman_ford_path_length(G, 0)\n    >>> length[4]\n    4\n    >>> for node in [0, 1, 2, 3, 4]:\n    ...     print(f"{node}: {length[node]}")\n    0: 0\n    1: 1\n    2: 2\n    3: 3\n    4: 4\n\n    Notes\n    -----\n    Edge weight attributes must be numerical.\n    Distances are calculated as sums of weighted edges traversed.\n\n    See Also\n    --------\n    single_source_dijkstra, single_source_bellman_ford\n\n    '
    weight = _weight_function(G, weight)
    return _bellman_ford(G, [source], weight)

@nx._dispatch(edge_attrs='weight')
def single_source_bellman_ford(G, source, target=None, weight='weight'):
    if False:
        i = 10
        return i + 15
    'Compute shortest paths and lengths in a weighted graph G.\n\n    Uses Bellman-Ford algorithm for shortest paths.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n\n    source : node label\n        Starting node for path\n\n    target : node label, optional\n        Ending node for path\n\n    weight : string or function\n        If this is a string, then edge weights will be accessed via the\n        edge attribute with this key (that is, the weight of the edge\n        joining `u` to `v` will be ``G.edges[u, v][weight]``). If no\n        such edge attribute exists, the weight of the edge is assumed to\n        be one.\n\n        If this is a function, the weight of an edge is the value\n        returned by the function. The function must accept exactly three\n        positional arguments: the two endpoints of an edge and the\n        dictionary of edge attributes for that edge. The function must\n        return a number.\n\n    Returns\n    -------\n    distance, path : pair of dictionaries, or numeric and list\n        If target is None, returns a tuple of two dictionaries keyed by node.\n        The first dictionary stores distance from one of the source nodes.\n        The second stores the path from one of the sources to that node.\n        If target is not None, returns a tuple of (distance, path) where\n        distance is the distance from source to target and path is a list\n        representing the path from source to target.\n\n    Raises\n    ------\n    NodeNotFound\n        If `source` is not in `G`.\n\n    Examples\n    --------\n    >>> G = nx.path_graph(5)\n    >>> length, path = nx.single_source_bellman_ford(G, 0)\n    >>> length[4]\n    4\n    >>> for node in [0, 1, 2, 3, 4]:\n    ...     print(f"{node}: {length[node]}")\n    0: 0\n    1: 1\n    2: 2\n    3: 3\n    4: 4\n    >>> path[4]\n    [0, 1, 2, 3, 4]\n    >>> length, path = nx.single_source_bellman_ford(G, 0, 1)\n    >>> length\n    1\n    >>> path\n    [0, 1]\n\n    Notes\n    -----\n    Edge weight attributes must be numerical.\n    Distances are calculated as sums of weighted edges traversed.\n\n    See Also\n    --------\n    single_source_dijkstra\n    single_source_bellman_ford_path\n    single_source_bellman_ford_path_length\n    '
    if source == target:
        if source not in G:
            raise nx.NodeNotFound(f'Node {source} is not found in the graph')
        return (0, [source])
    weight = _weight_function(G, weight)
    paths = {source: [source]}
    dist = _bellman_ford(G, [source], weight, paths=paths, target=target)
    if target is None:
        return (dist, paths)
    try:
        return (dist[target], paths[target])
    except KeyError as err:
        msg = f'Node {target} not reachable from {source}'
        raise nx.NetworkXNoPath(msg) from err

@nx._dispatch(edge_attrs='weight')
def all_pairs_bellman_ford_path_length(G, weight='weight'):
    if False:
        return 10
    'Compute shortest path lengths between all nodes in a weighted graph.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n\n    weight : string or function (default="weight")\n        If this is a string, then edge weights will be accessed via the\n        edge attribute with this key (that is, the weight of the edge\n        joining `u` to `v` will be ``G.edges[u, v][weight]``). If no\n        such edge attribute exists, the weight of the edge is assumed to\n        be one.\n\n        If this is a function, the weight of an edge is the value\n        returned by the function. The function must accept exactly three\n        positional arguments: the two endpoints of an edge and the\n        dictionary of edge attributes for that edge. The function must\n        return a number.\n\n    Returns\n    -------\n    distance : iterator\n        (source, dictionary) iterator with dictionary keyed by target and\n        shortest path length as the key value.\n\n    Examples\n    --------\n    >>> G = nx.path_graph(5)\n    >>> length = dict(nx.all_pairs_bellman_ford_path_length(G))\n    >>> for node in [0, 1, 2, 3, 4]:\n    ...     print(f"1 - {node}: {length[1][node]}")\n    1 - 0: 1\n    1 - 1: 0\n    1 - 2: 1\n    1 - 3: 2\n    1 - 4: 3\n    >>> length[3][2]\n    1\n    >>> length[2][2]\n    0\n\n    Notes\n    -----\n    Edge weight attributes must be numerical.\n    Distances are calculated as sums of weighted edges traversed.\n\n    The dictionary returned only has keys for reachable node pairs.\n    '
    length = single_source_bellman_ford_path_length
    for n in G:
        yield (n, dict(length(G, n, weight=weight)))

@nx._dispatch(edge_attrs='weight')
def all_pairs_bellman_ford_path(G, weight='weight'):
    if False:
        for i in range(10):
            print('nop')
    'Compute shortest paths between all nodes in a weighted graph.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n\n    weight : string or function (default="weight")\n        If this is a string, then edge weights will be accessed via the\n        edge attribute with this key (that is, the weight of the edge\n        joining `u` to `v` will be ``G.edges[u, v][weight]``). If no\n        such edge attribute exists, the weight of the edge is assumed to\n        be one.\n\n        If this is a function, the weight of an edge is the value\n        returned by the function. The function must accept exactly three\n        positional arguments: the two endpoints of an edge and the\n        dictionary of edge attributes for that edge. The function must\n        return a number.\n\n    Returns\n    -------\n    paths : iterator\n        (source, dictionary) iterator with dictionary keyed by target and\n        shortest path as the key value.\n\n    Examples\n    --------\n    >>> G = nx.path_graph(5)\n    >>> path = dict(nx.all_pairs_bellman_ford_path(G))\n    >>> path[0][4]\n    [0, 1, 2, 3, 4]\n\n    Notes\n    -----\n    Edge weight attributes must be numerical.\n    Distances are calculated as sums of weighted edges traversed.\n\n    See Also\n    --------\n    floyd_warshall, all_pairs_dijkstra_path\n\n    '
    path = single_source_bellman_ford_path
    for n in G:
        yield (n, path(G, n, weight=weight))

@nx._dispatch(edge_attrs='weight')
def goldberg_radzik(G, source, weight='weight'):
    if False:
        print('Hello World!')
    'Compute shortest path lengths and predecessors on shortest paths\n    in weighted graphs.\n\n    The algorithm has a running time of $O(mn)$ where $n$ is the number of\n    nodes and $m$ is the number of edges.  It is slower than Dijkstra but\n    can handle negative edge weights.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n        The algorithm works for all types of graphs, including directed\n        graphs and multigraphs.\n\n    source: node label\n        Starting node for path\n\n    weight : string or function\n        If this is a string, then edge weights will be accessed via the\n        edge attribute with this key (that is, the weight of the edge\n        joining `u` to `v` will be ``G.edges[u, v][weight]``). If no\n        such edge attribute exists, the weight of the edge is assumed to\n        be one.\n\n        If this is a function, the weight of an edge is the value\n        returned by the function. The function must accept exactly three\n        positional arguments: the two endpoints of an edge and the\n        dictionary of edge attributes for that edge. The function must\n        return a number.\n\n    Returns\n    -------\n    pred, dist : dictionaries\n        Returns two dictionaries keyed by node to predecessor in the\n        path and to the distance from the source respectively.\n\n    Raises\n    ------\n    NodeNotFound\n        If `source` is not in `G`.\n\n    NetworkXUnbounded\n        If the (di)graph contains a negative (di)cycle, the\n        algorithm raises an exception to indicate the presence of the\n        negative (di)cycle.  Note: any negative weight edge in an\n        undirected graph is a negative cycle.\n\n        As of NetworkX v3.2, a zero weight cycle is no longer\n        incorrectly reported as a negative weight cycle.\n\n\n    Examples\n    --------\n    >>> G = nx.path_graph(5, create_using=nx.DiGraph())\n    >>> pred, dist = nx.goldberg_radzik(G, 0)\n    >>> sorted(pred.items())\n    [(0, None), (1, 0), (2, 1), (3, 2), (4, 3)]\n    >>> sorted(dist.items())\n    [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]\n\n    >>> G = nx.cycle_graph(5, create_using=nx.DiGraph())\n    >>> G[1][2]["weight"] = -7\n    >>> nx.goldberg_radzik(G, 0)\n    Traceback (most recent call last):\n        ...\n    networkx.exception.NetworkXUnbounded: Negative cycle detected.\n\n    Notes\n    -----\n    Edge weight attributes must be numerical.\n    Distances are calculated as sums of weighted edges traversed.\n\n    The dictionaries returned only have keys for nodes reachable from\n    the source.\n\n    In the case where the (di)graph is not connected, if a component\n    not containing the source contains a negative (di)cycle, it\n    will not be detected.\n\n    '
    if source not in G:
        raise nx.NodeNotFound(f'Node {source} is not found in the graph')
    weight = _weight_function(G, weight)
    if G.is_multigraph():
        if any((weight(u, v, {k: d}) < 0 for (u, v, k, d) in nx.selfloop_edges(G, keys=True, data=True))):
            raise nx.NetworkXUnbounded('Negative cycle detected.')
    elif any((weight(u, v, d) < 0 for (u, v, d) in nx.selfloop_edges(G, data=True))):
        raise nx.NetworkXUnbounded('Negative cycle detected.')
    if len(G) == 1:
        return ({source: None}, {source: 0})
    G_succ = G._adj
    inf = float('inf')
    d = {u: inf for u in G}
    d[source] = 0
    pred = {source: None}

    def topo_sort(relabeled):
        if False:
            return 10
        'Topologically sort nodes relabeled in the previous round and detect\n        negative cycles.\n        '
        to_scan = []
        neg_count = {}
        for u in relabeled:
            if u in neg_count:
                continue
            d_u = d[u]
            if all((d_u + weight(u, v, e) >= d[v] for (v, e) in G_succ[u].items())):
                continue
            stack = [(u, iter(G_succ[u].items()))]
            in_stack = {u}
            neg_count[u] = 0
            while stack:
                (u, it) = stack[-1]
                try:
                    (v, e) = next(it)
                except StopIteration:
                    to_scan.append(u)
                    stack.pop()
                    in_stack.remove(u)
                    continue
                t = d[u] + weight(u, v, e)
                d_v = d[v]
                if t < d_v:
                    is_neg = t < d_v
                    d[v] = t
                    pred[v] = u
                    if v not in neg_count:
                        neg_count[v] = neg_count[u] + int(is_neg)
                        stack.append((v, iter(G_succ[v].items())))
                        in_stack.add(v)
                    elif v in in_stack and neg_count[u] + int(is_neg) > neg_count[v]:
                        raise nx.NetworkXUnbounded('Negative cycle detected.')
        to_scan.reverse()
        return to_scan

    def relax(to_scan):
        if False:
            return 10
        'Relax out-edges of relabeled nodes.'
        relabeled = set()
        for u in to_scan:
            d_u = d[u]
            for (v, e) in G_succ[u].items():
                w_e = weight(u, v, e)
                if d_u + w_e < d[v]:
                    d[v] = d_u + w_e
                    pred[v] = u
                    relabeled.add(v)
        return relabeled
    relabeled = {source}
    while relabeled:
        to_scan = topo_sort(relabeled)
        relabeled = relax(to_scan)
    d = {u: d[u] for u in pred}
    return (pred, d)

@nx._dispatch(edge_attrs='weight')
def negative_edge_cycle(G, weight='weight', heuristic=True):
    if False:
        i = 10
        return i + 15
    'Returns True if there exists a negative edge cycle anywhere in G.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n\n    weight : string or function\n        If this is a string, then edge weights will be accessed via the\n        edge attribute with this key (that is, the weight of the edge\n        joining `u` to `v` will be ``G.edges[u, v][weight]``). If no\n        such edge attribute exists, the weight of the edge is assumed to\n        be one.\n\n        If this is a function, the weight of an edge is the value\n        returned by the function. The function must accept exactly three\n        positional arguments: the two endpoints of an edge and the\n        dictionary of edge attributes for that edge. The function must\n        return a number.\n\n    heuristic : bool\n        Determines whether to use a heuristic to early detect negative\n        cycles at a negligible cost. In case of graphs with a negative cycle,\n        the performance of detection increases by at least an order of magnitude.\n\n    Returns\n    -------\n    negative_cycle : bool\n        True if a negative edge cycle exists, otherwise False.\n\n    Examples\n    --------\n    >>> G = nx.cycle_graph(5, create_using=nx.DiGraph())\n    >>> print(nx.negative_edge_cycle(G))\n    False\n    >>> G[1][2]["weight"] = -7\n    >>> print(nx.negative_edge_cycle(G))\n    True\n\n    Notes\n    -----\n    Edge weight attributes must be numerical.\n    Distances are calculated as sums of weighted edges traversed.\n\n    This algorithm uses bellman_ford_predecessor_and_distance() but finds\n    negative cycles on any component by first adding a new node connected to\n    every node, and starting bellman_ford_predecessor_and_distance on that\n    node.  It then removes that extra node.\n    '
    if G.size() == 0:
        return False
    newnode = -1
    while newnode in G:
        newnode -= 1
    G.add_edges_from([(newnode, n) for n in G])
    try:
        bellman_ford_predecessor_and_distance(G, newnode, weight=weight, heuristic=heuristic)
    except nx.NetworkXUnbounded:
        return True
    finally:
        G.remove_node(newnode)
    return False

@nx._dispatch(edge_attrs='weight')
def find_negative_cycle(G, source, weight='weight'):
    if False:
        for i in range(10):
            print('nop')
    'Returns a cycle with negative total weight if it exists.\n\n    Bellman-Ford is used to find shortest_paths. That algorithm\n    stops if there exists a negative cycle. This algorithm\n    picks up from there and returns the found negative cycle.\n\n    The cycle consists of a list of nodes in the cycle order. The last\n    node equals the first to make it a cycle.\n    You can look up the edge weights in the original graph. In the case\n    of multigraphs the relevant edge is the minimal weight edge between\n    the nodes in the 2-tuple.\n\n    If the graph has no negative cycle, a NetworkXError is raised.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n\n    source: node label\n        The search for the negative cycle will start from this node.\n\n    weight : string or function\n        If this is a string, then edge weights will be accessed via the\n        edge attribute with this key (that is, the weight of the edge\n        joining `u` to `v` will be ``G.edges[u, v][weight]``). If no\n        such edge attribute exists, the weight of the edge is assumed to\n        be one.\n\n        If this is a function, the weight of an edge is the value\n        returned by the function. The function must accept exactly three\n        positional arguments: the two endpoints of an edge and the\n        dictionary of edge attributes for that edge. The function must\n        return a number.\n\n    Examples\n    --------\n    >>> G = nx.DiGraph()\n    >>> G.add_weighted_edges_from([(0, 1, 2), (1, 2, 2), (2, 0, 1), (1, 4, 2), (4, 0, -5)])\n    >>> nx.find_negative_cycle(G, 0)\n    [4, 0, 1, 4]\n\n    Returns\n    -------\n    cycle : list\n        A list of nodes in the order of the cycle found. The last node\n        equals the first to indicate a cycle.\n\n    Raises\n    ------\n    NetworkXError\n        If no negative cycle is found.\n    '
    weight = _weight_function(G, weight)
    pred = {source: []}
    v = _inner_bellman_ford(G, [source], weight, pred=pred)
    if v is None:
        raise nx.NetworkXError('No negative cycles detected.')
    neg_cycle = []
    stack = [(v, list(pred[v]))]
    seen = {v}
    while stack:
        (node, preds) = stack[-1]
        if v in preds:
            neg_cycle.extend([node, v])
            neg_cycle = list(reversed(neg_cycle))
            return neg_cycle
        if preds:
            nbr = preds.pop()
            if nbr not in seen:
                stack.append((nbr, list(pred[nbr])))
                neg_cycle.append(node)
                seen.add(nbr)
        else:
            stack.pop()
            if neg_cycle:
                neg_cycle.pop()
            else:
                if v in G[v] and weight(G, v, v) < 0:
                    return [v, v]
                raise nx.NetworkXError('Negative cycle is detected but not found')
    msg = 'negative cycle detected but not identified'
    raise nx.NetworkXUnbounded(msg)

@nx._dispatch(edge_attrs='weight')
def bidirectional_dijkstra(G, source, target, weight='weight'):
    if False:
        for i in range(10):
            print('nop')
    'Dijkstra\'s algorithm for shortest paths using bidirectional search.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n\n    source : node\n        Starting node.\n\n    target : node\n        Ending node.\n\n    weight : string or function\n        If this is a string, then edge weights will be accessed via the\n        edge attribute with this key (that is, the weight of the edge\n        joining `u` to `v` will be ``G.edges[u, v][weight]``). If no\n        such edge attribute exists, the weight of the edge is assumed to\n        be one.\n\n        If this is a function, the weight of an edge is the value\n        returned by the function. The function must accept exactly three\n        positional arguments: the two endpoints of an edge and the\n        dictionary of edge attributes for that edge. The function must\n        return a number or None to indicate a hidden edge.\n\n    Returns\n    -------\n    length, path : number and list\n        length is the distance from source to target.\n        path is a list of nodes on a path from source to target.\n\n    Raises\n    ------\n    NodeNotFound\n        If either `source` or `target` is not in `G`.\n\n    NetworkXNoPath\n        If no path exists between source and target.\n\n    Examples\n    --------\n    >>> G = nx.path_graph(5)\n    >>> length, path = nx.bidirectional_dijkstra(G, 0, 4)\n    >>> print(length)\n    4\n    >>> print(path)\n    [0, 1, 2, 3, 4]\n\n    Notes\n    -----\n    Edge weight attributes must be numerical.\n    Distances are calculated as sums of weighted edges traversed.\n\n    The weight function can be used to hide edges by returning None.\n    So ``weight = lambda u, v, d: 1 if d[\'color\']=="red" else None``\n    will find the shortest red path.\n\n    In practice  bidirectional Dijkstra is much more than twice as fast as\n    ordinary Dijkstra.\n\n    Ordinary Dijkstra expands nodes in a sphere-like manner from the\n    source. The radius of this sphere will eventually be the length\n    of the shortest path. Bidirectional Dijkstra will expand nodes\n    from both the source and the target, making two spheres of half\n    this radius. Volume of the first sphere is `\\pi*r*r` while the\n    others are `2*\\pi*r/2*r/2`, making up half the volume.\n\n    This algorithm is not guaranteed to work if edge weights\n    are negative or are floating point numbers\n    (overflows and roundoff errors can cause problems).\n\n    See Also\n    --------\n    shortest_path\n    shortest_path_length\n    '
    if source not in G or target not in G:
        msg = f'Either source {source} or target {target} is not in G'
        raise nx.NodeNotFound(msg)
    if source == target:
        return (0, [source])
    weight = _weight_function(G, weight)
    push = heappush
    pop = heappop
    dists = [{}, {}]
    paths = [{source: [source]}, {target: [target]}]
    fringe = [[], []]
    seen = [{source: 0}, {target: 0}]
    c = count()
    push(fringe[0], (0, next(c), source))
    push(fringe[1], (0, next(c), target))
    if G.is_directed():
        neighs = [G._succ, G._pred]
    else:
        neighs = [G._adj, G._adj]
    finalpath = []
    dir = 1
    while fringe[0] and fringe[1]:
        dir = 1 - dir
        (dist, _, v) = pop(fringe[dir])
        if v in dists[dir]:
            continue
        dists[dir][v] = dist
        if v in dists[1 - dir]:
            return (finaldist, finalpath)
        for (w, d) in neighs[dir][v].items():
            cost = weight(v, w, d) if dir == 0 else weight(w, v, d)
            if cost is None:
                continue
            vwLength = dists[dir][v] + cost
            if w in dists[dir]:
                if vwLength < dists[dir][w]:
                    raise ValueError('Contradictory paths found: negative weights?')
            elif w not in seen[dir] or vwLength < seen[dir][w]:
                seen[dir][w] = vwLength
                push(fringe[dir], (vwLength, next(c), w))
                paths[dir][w] = paths[dir][v] + [w]
                if w in seen[0] and w in seen[1]:
                    totaldist = seen[0][w] + seen[1][w]
                    if finalpath == [] or finaldist > totaldist:
                        finaldist = totaldist
                        revpath = paths[1][w][:]
                        revpath.reverse()
                        finalpath = paths[0][w] + revpath[1:]
    raise nx.NetworkXNoPath(f'No path between {source} and {target}.')

@nx._dispatch(edge_attrs='weight')
def johnson(G, weight='weight'):
    if False:
        while True:
            i = 10
    'Uses Johnson\'s Algorithm to compute shortest paths.\n\n    Johnson\'s Algorithm finds a shortest path between each pair of\n    nodes in a weighted graph even if negative weights are present.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n\n    weight : string or function\n        If this is a string, then edge weights will be accessed via the\n        edge attribute with this key (that is, the weight of the edge\n        joining `u` to `v` will be ``G.edges[u, v][weight]``). If no\n        such edge attribute exists, the weight of the edge is assumed to\n        be one.\n\n        If this is a function, the weight of an edge is the value\n        returned by the function. The function must accept exactly three\n        positional arguments: the two endpoints of an edge and the\n        dictionary of edge attributes for that edge. The function must\n        return a number.\n\n    Returns\n    -------\n    distance : dictionary\n        Dictionary, keyed by source and target, of shortest paths.\n\n    Examples\n    --------\n    >>> graph = nx.DiGraph()\n    >>> graph.add_weighted_edges_from(\n    ...     [("0", "3", 3), ("0", "1", -5), ("0", "2", 2), ("1", "2", 4), ("2", "3", 1)]\n    ... )\n    >>> paths = nx.johnson(graph, weight="weight")\n    >>> paths["0"]["2"]\n    [\'0\', \'1\', \'2\']\n\n    Notes\n    -----\n    Johnson\'s algorithm is suitable even for graphs with negative weights. It\n    works by using the Bellman–Ford algorithm to compute a transformation of\n    the input graph that removes all negative weights, allowing Dijkstra\'s\n    algorithm to be used on the transformed graph.\n\n    The time complexity of this algorithm is $O(n^2 \\log n + n m)$,\n    where $n$ is the number of nodes and $m$ the number of edges in the\n    graph. For dense graphs, this may be faster than the Floyd–Warshall\n    algorithm.\n\n    See Also\n    --------\n    floyd_warshall_predecessor_and_distance\n    floyd_warshall_numpy\n    all_pairs_shortest_path\n    all_pairs_shortest_path_length\n    all_pairs_dijkstra_path\n    bellman_ford_predecessor_and_distance\n    all_pairs_bellman_ford_path\n    all_pairs_bellman_ford_path_length\n\n    '
    dist = {v: 0 for v in G}
    pred = {v: [] for v in G}
    weight = _weight_function(G, weight)
    dist_bellman = _bellman_ford(G, list(G), weight, pred=pred, dist=dist)

    def new_weight(u, v, d):
        if False:
            return 10
        return weight(u, v, d) + dist_bellman[u] - dist_bellman[v]

    def dist_path(v):
        if False:
            return 10
        paths = {v: [v]}
        _dijkstra(G, v, new_weight, paths=paths)
        return paths
    return {v: dist_path(v) for v in G}