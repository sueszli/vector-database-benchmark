"""
Shortest path algorithms for unweighted graphs.
"""
import warnings
import networkx as nx
__all__ = ['bidirectional_shortest_path', 'single_source_shortest_path', 'single_source_shortest_path_length', 'single_target_shortest_path', 'single_target_shortest_path_length', 'all_pairs_shortest_path', 'all_pairs_shortest_path_length', 'predecessor']

@nx._dispatch
def single_source_shortest_path_length(G, source, cutoff=None):
    if False:
        print('Hello World!')
    'Compute the shortest path lengths from source to all reachable nodes.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n\n    source : node\n       Starting node for path\n\n    cutoff : integer, optional\n        Depth to stop the search. Only paths of length <= cutoff are returned.\n\n    Returns\n    -------\n    lengths : dict\n        Dict keyed by node to shortest path length to source.\n\n    Examples\n    --------\n    >>> G = nx.path_graph(5)\n    >>> length = nx.single_source_shortest_path_length(G, 0)\n    >>> length[4]\n    4\n    >>> for node in length:\n    ...     print(f"{node}: {length[node]}")\n    0: 0\n    1: 1\n    2: 2\n    3: 3\n    4: 4\n\n    See Also\n    --------\n    shortest_path_length\n    '
    if source not in G:
        raise nx.NodeNotFound(f'Source {source} is not in G')
    if cutoff is None:
        cutoff = float('inf')
    nextlevel = [source]
    return dict(_single_shortest_path_length(G._adj, nextlevel, cutoff))

def _single_shortest_path_length(adj, firstlevel, cutoff):
    if False:
        while True:
            i = 10
    'Yields (node, level) in a breadth first search\n\n    Shortest Path Length helper function\n    Parameters\n    ----------\n        adj : dict\n            Adjacency dict or view\n        firstlevel : list\n            starting nodes, e.g. [source] or [target]\n        cutoff : int or float\n            level at which we stop the process\n    '
    seen = set(firstlevel)
    nextlevel = firstlevel
    level = 0
    n = len(adj)
    for v in nextlevel:
        yield (v, level)
    while nextlevel and cutoff > level:
        level += 1
        thislevel = nextlevel
        nextlevel = []
        for v in thislevel:
            for w in adj[v]:
                if w not in seen:
                    seen.add(w)
                    nextlevel.append(w)
                    yield (w, level)
            if len(seen) == n:
                return

@nx._dispatch
def single_target_shortest_path_length(G, target, cutoff=None):
    if False:
        return 10
    'Compute the shortest path lengths to target from all reachable nodes.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n\n    target : node\n       Target node for path\n\n    cutoff : integer, optional\n        Depth to stop the search. Only paths of length <= cutoff are returned.\n\n    Returns\n    -------\n    lengths : iterator\n        (source, shortest path length) iterator\n\n    Examples\n    --------\n    >>> G = nx.path_graph(5, create_using=nx.DiGraph())\n    >>> length = dict(nx.single_target_shortest_path_length(G, 4))\n    >>> length[0]\n    4\n    >>> for node in range(5):\n    ...     print(f"{node}: {length[node]}")\n    0: 4\n    1: 3\n    2: 2\n    3: 1\n    4: 0\n\n    See Also\n    --------\n    single_source_shortest_path_length, shortest_path_length\n    '
    if target not in G:
        raise nx.NodeNotFound(f'Target {target} is not in G')
    msg = 'single_target_shortest_path_length will return a dict starting in v3.3'
    warnings.warn(msg, DeprecationWarning)
    if cutoff is None:
        cutoff = float('inf')
    adj = G._pred if G.is_directed() else G._adj
    nextlevel = [target]
    return _single_shortest_path_length(adj, nextlevel, cutoff)

@nx._dispatch
def all_pairs_shortest_path_length(G, cutoff=None):
    if False:
        while True:
            i = 10
    'Computes the shortest path lengths between all nodes in `G`.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n\n    cutoff : integer, optional\n        Depth at which to stop the search. Only paths of length at most\n        `cutoff` are returned.\n\n    Returns\n    -------\n    lengths : iterator\n        (source, dictionary) iterator with dictionary keyed by target and\n        shortest path length as the key value.\n\n    Notes\n    -----\n    The iterator returned only has reachable node pairs.\n\n    Examples\n    --------\n    >>> G = nx.path_graph(5)\n    >>> length = dict(nx.all_pairs_shortest_path_length(G))\n    >>> for node in [0, 1, 2, 3, 4]:\n    ...     print(f"1 - {node}: {length[1][node]}")\n    1 - 0: 1\n    1 - 1: 0\n    1 - 2: 1\n    1 - 3: 2\n    1 - 4: 3\n    >>> length[3][2]\n    1\n    >>> length[2][2]\n    0\n\n    '
    length = single_source_shortest_path_length
    for n in G:
        yield (n, length(G, n, cutoff=cutoff))

@nx._dispatch
def bidirectional_shortest_path(G, source, target):
    if False:
        return 10
    'Returns a list of nodes in a shortest path between source and target.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n\n    source : node label\n       starting node for path\n\n    target : node label\n       ending node for path\n\n    Returns\n    -------\n    path: list\n       List of nodes in a path from source to target.\n\n    Raises\n    ------\n    NetworkXNoPath\n       If no path exists between source and target.\n\n    Examples\n    --------\n    >>> G = nx.Graph()\n    >>> nx.add_path(G, [0, 1, 2, 3, 0, 4, 5, 6, 7, 4])\n    >>> nx.bidirectional_shortest_path(G, 2, 6)\n    [2, 1, 0, 4, 5, 6]\n\n    See Also\n    --------\n    shortest_path\n\n    Notes\n    -----\n    This algorithm is used by shortest_path(G, source, target).\n    '
    if source not in G or target not in G:
        msg = f'Either source {source} or target {target} is not in G'
        raise nx.NodeNotFound(msg)
    results = _bidirectional_pred_succ(G, source, target)
    (pred, succ, w) = results
    path = []
    while w is not None:
        path.append(w)
        w = pred[w]
    path.reverse()
    w = succ[path[-1]]
    while w is not None:
        path.append(w)
        w = succ[w]
    return path

def _bidirectional_pred_succ(G, source, target):
    if False:
        print('Hello World!')
    'Bidirectional shortest path helper.\n\n    Returns (pred, succ, w) where\n    pred is a dictionary of predecessors from w to the source, and\n    succ is a dictionary of successors from w to the target.\n    '
    if target == source:
        return ({target: None}, {source: None}, source)
    if G.is_directed():
        Gpred = G.pred
        Gsucc = G.succ
    else:
        Gpred = G.adj
        Gsucc = G.adj
    pred = {source: None}
    succ = {target: None}
    forward_fringe = [source]
    reverse_fringe = [target]
    while forward_fringe and reverse_fringe:
        if len(forward_fringe) <= len(reverse_fringe):
            this_level = forward_fringe
            forward_fringe = []
            for v in this_level:
                for w in Gsucc[v]:
                    if w not in pred:
                        forward_fringe.append(w)
                        pred[w] = v
                    if w in succ:
                        return (pred, succ, w)
        else:
            this_level = reverse_fringe
            reverse_fringe = []
            for v in this_level:
                for w in Gpred[v]:
                    if w not in succ:
                        succ[w] = v
                        reverse_fringe.append(w)
                    if w in pred:
                        return (pred, succ, w)
    raise nx.NetworkXNoPath(f'No path between {source} and {target}.')

@nx._dispatch
def single_source_shortest_path(G, source, cutoff=None):
    if False:
        i = 10
        return i + 15
    "Compute shortest path between source\n    and all other nodes reachable from source.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n\n    source : node label\n       Starting node for path\n\n    cutoff : integer, optional\n        Depth to stop the search. Only paths of length <= cutoff are returned.\n\n    Returns\n    -------\n    paths : dictionary\n        Dictionary, keyed by target, of shortest paths.\n\n    Examples\n    --------\n    >>> G = nx.path_graph(5)\n    >>> path = nx.single_source_shortest_path(G, 0)\n    >>> path[4]\n    [0, 1, 2, 3, 4]\n\n    Notes\n    -----\n    The shortest path is not necessarily unique. So there can be multiple\n    paths between the source and each target node, all of which have the\n    same 'shortest' length. For each target node, this function returns\n    only one of those paths.\n\n    See Also\n    --------\n    shortest_path\n    "
    if source not in G:
        raise nx.NodeNotFound(f'Source {source} not in G')

    def join(p1, p2):
        if False:
            return 10
        return p1 + p2
    if cutoff is None:
        cutoff = float('inf')
    nextlevel = {source: 1}
    paths = {source: [source]}
    return dict(_single_shortest_path(G.adj, nextlevel, paths, cutoff, join))

def _single_shortest_path(adj, firstlevel, paths, cutoff, join):
    if False:
        for i in range(10):
            print('nop')
    'Returns shortest paths\n\n    Shortest Path helper function\n    Parameters\n    ----------\n        adj : dict\n            Adjacency dict or view\n        firstlevel : dict\n            starting nodes, e.g. {source: 1} or {target: 1}\n        paths : dict\n            paths for starting nodes, e.g. {source: [source]}\n        cutoff : int or float\n            level at which we stop the process\n        join : function\n            function to construct a path from two partial paths. Requires two\n            list inputs `p1` and `p2`, and returns a list. Usually returns\n            `p1 + p2` (forward from source) or `p2 + p1` (backward from target)\n    '
    level = 0
    nextlevel = firstlevel
    while nextlevel and cutoff > level:
        thislevel = nextlevel
        nextlevel = {}
        for v in thislevel:
            for w in adj[v]:
                if w not in paths:
                    paths[w] = join(paths[v], [w])
                    nextlevel[w] = 1
        level += 1
    return paths

@nx._dispatch
def single_target_shortest_path(G, target, cutoff=None):
    if False:
        i = 10
        return i + 15
    "Compute shortest path to target from all nodes that reach target.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n\n    target : node label\n       Target node for path\n\n    cutoff : integer, optional\n        Depth to stop the search. Only paths of length <= cutoff are returned.\n\n    Returns\n    -------\n    paths : dictionary\n        Dictionary, keyed by target, of shortest paths.\n\n    Examples\n    --------\n    >>> G = nx.path_graph(5, create_using=nx.DiGraph())\n    >>> path = nx.single_target_shortest_path(G, 4)\n    >>> path[0]\n    [0, 1, 2, 3, 4]\n\n    Notes\n    -----\n    The shortest path is not necessarily unique. So there can be multiple\n    paths between the source and each target node, all of which have the\n    same 'shortest' length. For each target node, this function returns\n    only one of those paths.\n\n    See Also\n    --------\n    shortest_path, single_source_shortest_path\n    "
    if target not in G:
        raise nx.NodeNotFound(f'Target {target} not in G')

    def join(p1, p2):
        if False:
            return 10
        return p2 + p1
    adj = G.pred if G.is_directed() else G.adj
    if cutoff is None:
        cutoff = float('inf')
    nextlevel = {target: 1}
    paths = {target: [target]}
    return dict(_single_shortest_path(adj, nextlevel, paths, cutoff, join))

@nx._dispatch
def all_pairs_shortest_path(G, cutoff=None):
    if False:
        print('Hello World!')
    'Compute shortest paths between all nodes.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n\n    cutoff : integer, optional\n        Depth at which to stop the search. Only paths of length at most\n        `cutoff` are returned.\n\n    Returns\n    -------\n    paths : iterator\n        Dictionary, keyed by source and target, of shortest paths.\n\n    Examples\n    --------\n    >>> G = nx.path_graph(5)\n    >>> path = dict(nx.all_pairs_shortest_path(G))\n    >>> print(path[0][4])\n    [0, 1, 2, 3, 4]\n\n    Notes\n    -----\n    There may be multiple shortest paths with the same length between\n    two nodes. For each pair, this function returns only one of those paths.\n\n    See Also\n    --------\n    floyd_warshall\n    all_pairs_all_shortest_paths\n\n    '
    for n in G:
        yield (n, single_source_shortest_path(G, n, cutoff=cutoff))

@nx._dispatch
def predecessor(G, source, target=None, cutoff=None, return_seen=None):
    if False:
        print('Hello World!')
    'Returns dict of predecessors for the path from source to all nodes in G.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n\n    source : node label\n       Starting node for path\n\n    target : node label, optional\n       Ending node for path. If provided only predecessors between\n       source and target are returned\n\n    cutoff : integer, optional\n        Depth to stop the search. Only paths of length <= cutoff are returned.\n\n    return_seen : bool, optional (default=None)\n        Whether to return a dictionary, keyed by node, of the level (number of\n        hops) to reach the node (as seen during breadth-first-search).\n\n    Returns\n    -------\n    pred : dictionary\n        Dictionary, keyed by node, of predecessors in the shortest path.\n\n\n    (pred, seen): tuple of dictionaries\n        If `return_seen` argument is set to `True`, then a tuple of dictionaries\n        is returned. The first element is the dictionary, keyed by node, of\n        predecessors in the shortest path. The second element is the dictionary,\n        keyed by node, of the level (number of hops) to reach the node (as seen\n        during breadth-first-search).\n\n    Examples\n    --------\n    >>> G = nx.path_graph(4)\n    >>> list(G)\n    [0, 1, 2, 3]\n    >>> nx.predecessor(G, 0)\n    {0: [], 1: [0], 2: [1], 3: [2]}\n    >>> nx.predecessor(G, 0, return_seen=True)\n    ({0: [], 1: [0], 2: [1], 3: [2]}, {0: 0, 1: 1, 2: 2, 3: 3})\n\n\n    '
    if source not in G:
        raise nx.NodeNotFound(f'Source {source} not in G')
    level = 0
    nextlevel = [source]
    seen = {source: level}
    pred = {source: []}
    while nextlevel:
        level = level + 1
        thislevel = nextlevel
        nextlevel = []
        for v in thislevel:
            for w in G[v]:
                if w not in seen:
                    pred[w] = [v]
                    seen[w] = level
                    nextlevel.append(w)
                elif seen[w] == level:
                    pred[w].append(v)
        if cutoff and cutoff <= level:
            break
    if target is not None:
        if return_seen:
            if target not in pred:
                return ([], -1)
            return (pred[target], seen[target])
        else:
            if target not in pred:
                return []
            return pred[target]
    elif return_seen:
        return (pred, seen)
    else:
        return pred