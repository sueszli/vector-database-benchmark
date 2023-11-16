from heapq import heappop, heappush
from itertools import count
import networkx as nx
from networkx.algorithms.shortest_paths.weighted import _weight_function
from networkx.utils import not_implemented_for, pairwise
__all__ = ['all_simple_paths', 'is_simple_path', 'shortest_simple_paths', 'all_simple_edge_paths']

@nx._dispatch
def is_simple_path(G, nodes):
    if False:
        for i in range(10):
            print('nop')
    "Returns True if and only if `nodes` form a simple path in `G`.\n\n    A *simple path* in a graph is a nonempty sequence of nodes in which\n    no node appears more than once in the sequence, and each adjacent\n    pair of nodes in the sequence is adjacent in the graph.\n\n    Parameters\n    ----------\n    G : graph\n        A NetworkX graph.\n    nodes : list\n        A list of one or more nodes in the graph `G`.\n\n    Returns\n    -------\n    bool\n        Whether the given list of nodes represents a simple path in `G`.\n\n    Notes\n    -----\n    An empty list of nodes is not a path but a list of one node is a\n    path. Here's an explanation why.\n\n    This function operates on *node paths*. One could also consider\n    *edge paths*. There is a bijection between node paths and edge\n    paths.\n\n    The *length of a path* is the number of edges in the path, so a list\n    of nodes of length *n* corresponds to a path of length *n* - 1.\n    Thus the smallest edge path would be a list of zero edges, the empty\n    path. This corresponds to a list of one node.\n\n    To convert between a node path and an edge path, you can use code\n    like the following::\n\n        >>> from networkx.utils import pairwise\n        >>> nodes = [0, 1, 2, 3]\n        >>> edges = list(pairwise(nodes))\n        >>> edges\n        [(0, 1), (1, 2), (2, 3)]\n        >>> nodes = [edges[0][0]] + [v for u, v in edges]\n        >>> nodes\n        [0, 1, 2, 3]\n\n    Examples\n    --------\n    >>> G = nx.cycle_graph(4)\n    >>> nx.is_simple_path(G, [2, 3, 0])\n    True\n    >>> nx.is_simple_path(G, [0, 2])\n    False\n\n    "
    if len(nodes) == 0:
        return False
    if len(nodes) == 1:
        return nodes[0] in G
    if not all((n in G for n in nodes)):
        return False
    if len(set(nodes)) != len(nodes):
        return False
    return all((v in G[u] for (u, v) in pairwise(nodes)))

@nx._dispatch
def all_simple_paths(G, source, target, cutoff=None):
    if False:
        return 10
    'Generate all simple paths in the graph G from source to target.\n\n    A simple path is a path with no repeated nodes.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n\n    source : node\n       Starting node for path\n\n    target : nodes\n       Single node or iterable of nodes at which to end path\n\n    cutoff : integer, optional\n        Depth to stop the search. Only paths of length <= cutoff are returned.\n\n    Returns\n    -------\n    path_generator: generator\n       A generator that produces lists of simple paths.  If there are no paths\n       between the source and target within the given cutoff the generator\n       produces no output. If it is possible to traverse the same sequence of\n       nodes in multiple ways, namely through parallel edges, then it will be\n       returned multiple times (once for each viable edge combination).\n\n    Examples\n    --------\n    This iterator generates lists of nodes::\n\n        >>> G = nx.complete_graph(4)\n        >>> for path in nx.all_simple_paths(G, source=0, target=3):\n        ...     print(path)\n        ...\n        [0, 1, 2, 3]\n        [0, 1, 3]\n        [0, 2, 1, 3]\n        [0, 2, 3]\n        [0, 3]\n\n    You can generate only those paths that are shorter than a certain\n    length by using the `cutoff` keyword argument::\n\n        >>> paths = nx.all_simple_paths(G, source=0, target=3, cutoff=2)\n        >>> print(list(paths))\n        [[0, 1, 3], [0, 2, 3], [0, 3]]\n\n    To get each path as the corresponding list of edges, you can use the\n    :func:`networkx.utils.pairwise` helper function::\n\n        >>> paths = nx.all_simple_paths(G, source=0, target=3)\n        >>> for path in map(nx.utils.pairwise, paths):\n        ...     print(list(path))\n        [(0, 1), (1, 2), (2, 3)]\n        [(0, 1), (1, 3)]\n        [(0, 2), (2, 1), (1, 3)]\n        [(0, 2), (2, 3)]\n        [(0, 3)]\n\n    Pass an iterable of nodes as target to generate all paths ending in any of several nodes::\n\n        >>> G = nx.complete_graph(4)\n        >>> for path in nx.all_simple_paths(G, source=0, target=[3, 2]):\n        ...     print(path)\n        ...\n        [0, 1, 2]\n        [0, 1, 2, 3]\n        [0, 1, 3]\n        [0, 1, 3, 2]\n        [0, 2]\n        [0, 2, 1, 3]\n        [0, 2, 3]\n        [0, 3]\n        [0, 3, 1, 2]\n        [0, 3, 2]\n\n    Iterate over each path from the root nodes to the leaf nodes in a\n    directed acyclic graph using a functional programming approach::\n\n        >>> from itertools import chain\n        >>> from itertools import product\n        >>> from itertools import starmap\n        >>> from functools import partial\n        >>>\n        >>> chaini = chain.from_iterable\n        >>>\n        >>> G = nx.DiGraph([(0, 1), (1, 2), (0, 3), (3, 2)])\n        >>> roots = (v for v, d in G.in_degree() if d == 0)\n        >>> leaves = (v for v, d in G.out_degree() if d == 0)\n        >>> all_paths = partial(nx.all_simple_paths, G)\n        >>> list(chaini(starmap(all_paths, product(roots, leaves))))\n        [[0, 1, 2], [0, 3, 2]]\n\n    The same list computed using an iterative approach::\n\n        >>> G = nx.DiGraph([(0, 1), (1, 2), (0, 3), (3, 2)])\n        >>> roots = (v for v, d in G.in_degree() if d == 0)\n        >>> leaves = (v for v, d in G.out_degree() if d == 0)\n        >>> all_paths = []\n        >>> for root in roots:\n        ...     for leaf in leaves:\n        ...         paths = nx.all_simple_paths(G, root, leaf)\n        ...         all_paths.extend(paths)\n        >>> all_paths\n        [[0, 1, 2], [0, 3, 2]]\n\n    Iterate over each path from the root nodes to the leaf nodes in a\n    directed acyclic graph passing all leaves together to avoid unnecessary\n    compute::\n\n        >>> G = nx.DiGraph([(0, 1), (2, 1), (1, 3), (1, 4)])\n        >>> roots = (v for v, d in G.in_degree() if d == 0)\n        >>> leaves = [v for v, d in G.out_degree() if d == 0]\n        >>> all_paths = []\n        >>> for root in roots:\n        ...     paths = nx.all_simple_paths(G, root, leaves)\n        ...     all_paths.extend(paths)\n        >>> all_paths\n        [[0, 1, 3], [0, 1, 4], [2, 1, 3], [2, 1, 4]]\n\n    If parallel edges offer multiple ways to traverse a given sequence of\n    nodes, this sequence of nodes will be returned multiple times:\n\n        >>> G = nx.MultiDiGraph([(0, 1), (0, 1), (1, 2)])\n        >>> list(nx.all_simple_paths(G, 0, 2))\n        [[0, 1, 2], [0, 1, 2]]\n\n    Notes\n    -----\n    This algorithm uses a modified depth-first search to generate the\n    paths [1]_.  A single path can be found in $O(V+E)$ time but the\n    number of simple paths in a graph can be very large, e.g. $O(n!)$ in\n    the complete graph of order $n$.\n\n    This function does not check that a path exists between `source` and\n    `target`. For large graphs, this may result in very long runtimes.\n    Consider using `has_path` to check that a path exists between `source` and\n    `target` before calling this function on large graphs.\n\n    References\n    ----------\n    .. [1] R. Sedgewick, "Algorithms in C, Part 5: Graph Algorithms",\n       Addison Wesley Professional, 3rd ed., 2001.\n\n    See Also\n    --------\n    all_shortest_paths, shortest_path, has_path\n\n    '
    if source not in G:
        raise nx.NodeNotFound(f'source node {source} not in graph')
    if target in G:
        targets = {target}
    else:
        try:
            targets = set(target)
        except TypeError as err:
            raise nx.NodeNotFound(f'target node {target} not in graph') from err
    if source in targets:
        return _empty_generator()
    if cutoff is None:
        cutoff = len(G) - 1
    if cutoff < 1:
        return _empty_generator()
    if G.is_multigraph():
        return _all_simple_paths_multigraph(G, source, targets, cutoff)
    else:
        return _all_simple_paths_graph(G, source, targets, cutoff)

def _empty_generator():
    if False:
        i = 10
        return i + 15
    yield from ()

def _all_simple_paths_graph(G, source, targets, cutoff):
    if False:
        while True:
            i = 10
    visited = {source: True}
    stack = [iter(G[source])]
    while stack:
        children = stack[-1]
        child = next(children, None)
        if child is None:
            stack.pop()
            visited.popitem()
        elif len(visited) < cutoff:
            if child in visited:
                continue
            if child in targets:
                yield (list(visited) + [child])
            visited[child] = True
            if targets - set(visited.keys()):
                stack.append(iter(G[child]))
            else:
                visited.popitem()
        else:
            for target in (targets & (set(children) | {child})) - set(visited.keys()):
                yield (list(visited) + [target])
            stack.pop()
            visited.popitem()

def _all_simple_paths_multigraph(G, source, targets, cutoff):
    if False:
        for i in range(10):
            print('nop')
    visited = {source: True}
    stack = [(v for (u, v) in G.edges(source))]
    while stack:
        children = stack[-1]
        child = next(children, None)
        if child is None:
            stack.pop()
            visited.popitem()
        elif len(visited) < cutoff:
            if child in visited:
                continue
            if child in targets:
                yield (list(visited) + [child])
            visited[child] = True
            if targets - set(visited.keys()):
                stack.append((v for (u, v) in G.edges(child)))
            else:
                visited.popitem()
        else:
            for target in targets - set(visited.keys()):
                count = ([child] + list(children)).count(target)
                for i in range(count):
                    yield (list(visited) + [target])
            stack.pop()
            visited.popitem()

@nx._dispatch
def all_simple_edge_paths(G, source, target, cutoff=None):
    if False:
        for i in range(10):
            print('nop')
    'Generate lists of edges for all simple paths in G from source to target.\n\n    A simple path is a path with no repeated nodes.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n\n    source : node\n       Starting node for path\n\n    target : nodes\n       Single node or iterable of nodes at which to end path\n\n    cutoff : integer, optional\n        Depth to stop the search. Only paths of length <= cutoff are returned.\n\n    Returns\n    -------\n    path_generator: generator\n       A generator that produces lists of simple paths.  If there are no paths\n       between the source and target within the given cutoff the generator\n       produces no output.\n       For multigraphs, the list of edges have elements of the form `(u,v,k)`.\n       Where `k` corresponds to the edge key.\n\n    Examples\n    --------\n\n    Print the simple path edges of a Graph::\n\n        >>> g = nx.Graph([(1, 2), (2, 4), (1, 3), (3, 4)])\n        >>> for path in sorted(nx.all_simple_edge_paths(g, 1, 4)):\n        ...     print(path)\n        [(1, 2), (2, 4)]\n        [(1, 3), (3, 4)]\n\n    Print the simple path edges of a MultiGraph. Returned edges come with\n    their associated keys::\n\n        >>> mg = nx.MultiGraph()\n        >>> mg.add_edge(1, 2, key="k0")\n        \'k0\'\n        >>> mg.add_edge(1, 2, key="k1")\n        \'k1\'\n        >>> mg.add_edge(2, 3, key="k0")\n        \'k0\'\n        >>> for path in sorted(nx.all_simple_edge_paths(mg, 1, 3)):\n        ...     print(path)\n        [(1, 2, \'k0\'), (2, 3, \'k0\')]\n        [(1, 2, \'k1\'), (2, 3, \'k0\')]\n\n\n    Notes\n    -----\n    This algorithm uses a modified depth-first search to generate the\n    paths [1]_.  A single path can be found in $O(V+E)$ time but the\n    number of simple paths in a graph can be very large, e.g. $O(n!)$ in\n    the complete graph of order $n$.\n\n    References\n    ----------\n    .. [1] R. Sedgewick, "Algorithms in C, Part 5: Graph Algorithms",\n       Addison Wesley Professional, 3rd ed., 2001.\n\n    See Also\n    --------\n    all_shortest_paths, shortest_path, all_simple_paths\n\n    '
    if source not in G:
        raise nx.NodeNotFound('source node %s not in graph' % source)
    if target in G:
        targets = {target}
    else:
        try:
            targets = set(target)
        except TypeError:
            raise nx.NodeNotFound('target node %s not in graph' % target)
    if source in targets:
        return []
    if cutoff is None:
        cutoff = len(G) - 1
    if cutoff < 1:
        return []
    if G.is_multigraph():
        for simp_path in _all_simple_edge_paths_multigraph(G, source, targets, cutoff):
            yield simp_path
    else:
        for simp_path in _all_simple_paths_graph(G, source, targets, cutoff):
            yield list(zip(simp_path[:-1], simp_path[1:]))

def _all_simple_edge_paths_multigraph(G, source, targets, cutoff):
    if False:
        return 10
    if not cutoff or cutoff < 1:
        return []
    visited = [source]
    stack = [iter(G.edges(source, keys=True))]
    while stack:
        children = stack[-1]
        child = next(children, None)
        if child is None:
            stack.pop()
            visited.pop()
        elif len(visited) < cutoff:
            if child[1] in targets:
                yield (visited[1:] + [child])
            elif child[1] not in [v[0] for v in visited[1:]]:
                visited.append(child)
                stack.append(iter(G.edges(child[1], keys=True)))
        else:
            for (u, v, k) in [child] + list(children):
                if v in targets:
                    yield (visited[1:] + [(u, v, k)])
            stack.pop()
            visited.pop()

@not_implemented_for('multigraph')
@nx._dispatch(edge_attrs='weight')
def shortest_simple_paths(G, source, target, weight=None):
    if False:
        while True:
            i = 10
    'Generate all simple paths in the graph G from source to target,\n       starting from shortest ones.\n\n    A simple path is a path with no repeated nodes.\n\n    If a weighted shortest path search is to be used, no negative weights\n    are allowed.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n\n    source : node\n       Starting node for path\n\n    target : node\n       Ending node for path\n\n    weight : string or function\n        If it is a string, it is the name of the edge attribute to be\n        used as a weight.\n\n        If it is a function, the weight of an edge is the value returned\n        by the function. The function must accept exactly three positional\n        arguments: the two endpoints of an edge and the dictionary of edge\n        attributes for that edge. The function must return a number.\n\n        If None all edges are considered to have unit weight. Default\n        value None.\n\n    Returns\n    -------\n    path_generator: generator\n       A generator that produces lists of simple paths, in order from\n       shortest to longest.\n\n    Raises\n    ------\n    NetworkXNoPath\n       If no path exists between source and target.\n\n    NetworkXError\n       If source or target nodes are not in the input graph.\n\n    NetworkXNotImplemented\n       If the input graph is a Multi[Di]Graph.\n\n    Examples\n    --------\n\n    >>> G = nx.cycle_graph(7)\n    >>> paths = list(nx.shortest_simple_paths(G, 0, 3))\n    >>> print(paths)\n    [[0, 1, 2, 3], [0, 6, 5, 4, 3]]\n\n    You can use this function to efficiently compute the k shortest/best\n    paths between two nodes.\n\n    >>> from itertools import islice\n    >>> def k_shortest_paths(G, source, target, k, weight=None):\n    ...     return list(\n    ...         islice(nx.shortest_simple_paths(G, source, target, weight=weight), k)\n    ...     )\n    >>> for path in k_shortest_paths(G, 0, 3, 2):\n    ...     print(path)\n    [0, 1, 2, 3]\n    [0, 6, 5, 4, 3]\n\n    Notes\n    -----\n    This procedure is based on algorithm by Jin Y. Yen [1]_.  Finding\n    the first $K$ paths requires $O(KN^3)$ operations.\n\n    See Also\n    --------\n    all_shortest_paths\n    shortest_path\n    all_simple_paths\n\n    References\n    ----------\n    .. [1] Jin Y. Yen, "Finding the K Shortest Loopless Paths in a\n       Network", Management Science, Vol. 17, No. 11, Theory Series\n       (Jul., 1971), pp. 712-716.\n\n    '
    if source not in G:
        raise nx.NodeNotFound(f'source node {source} not in graph')
    if target not in G:
        raise nx.NodeNotFound(f'target node {target} not in graph')
    if weight is None:
        length_func = len
        shortest_path_func = _bidirectional_shortest_path
    else:
        wt = _weight_function(G, weight)

        def length_func(path):
            if False:
                print('Hello World!')
            return sum((wt(u, v, G.get_edge_data(u, v)) for (u, v) in zip(path, path[1:])))
        shortest_path_func = _bidirectional_dijkstra
    listA = []
    listB = PathBuffer()
    prev_path = None
    while True:
        if not prev_path:
            (length, path) = shortest_path_func(G, source, target, weight=weight)
            listB.push(length, path)
        else:
            ignore_nodes = set()
            ignore_edges = set()
            for i in range(1, len(prev_path)):
                root = prev_path[:i]
                root_length = length_func(root)
                for path in listA:
                    if path[:i] == root:
                        ignore_edges.add((path[i - 1], path[i]))
                try:
                    (length, spur) = shortest_path_func(G, root[-1], target, ignore_nodes=ignore_nodes, ignore_edges=ignore_edges, weight=weight)
                    path = root[:-1] + spur
                    listB.push(root_length + length, path)
                except nx.NetworkXNoPath:
                    pass
                ignore_nodes.add(root[-1])
        if listB:
            path = listB.pop()
            yield path
            listA.append(path)
            prev_path = path
        else:
            break

class PathBuffer:

    def __init__(self):
        if False:
            print('Hello World!')
        self.paths = set()
        self.sortedpaths = []
        self.counter = count()

    def __len__(self):
        if False:
            while True:
                i = 10
        return len(self.sortedpaths)

    def push(self, cost, path):
        if False:
            return 10
        hashable_path = tuple(path)
        if hashable_path not in self.paths:
            heappush(self.sortedpaths, (cost, next(self.counter), path))
            self.paths.add(hashable_path)

    def pop(self):
        if False:
            for i in range(10):
                print('nop')
        (cost, num, path) = heappop(self.sortedpaths)
        hashable_path = tuple(path)
        self.paths.remove(hashable_path)
        return path

def _bidirectional_shortest_path(G, source, target, ignore_nodes=None, ignore_edges=None, weight=None):
    if False:
        i = 10
        return i + 15
    'Returns the shortest path between source and target ignoring\n       nodes and edges in the containers ignore_nodes and ignore_edges.\n\n    This is a custom modification of the standard bidirectional shortest\n    path implementation at networkx.algorithms.unweighted\n\n    Parameters\n    ----------\n    G : NetworkX graph\n\n    source : node\n       starting node for path\n\n    target : node\n       ending node for path\n\n    ignore_nodes : container of nodes\n       nodes to ignore, optional\n\n    ignore_edges : container of edges\n       edges to ignore, optional\n\n    weight : None\n       This function accepts a weight argument for convenience of\n       shortest_simple_paths function. It will be ignored.\n\n    Returns\n    -------\n    path: list\n       List of nodes in a path from source to target.\n\n    Raises\n    ------\n    NetworkXNoPath\n       If no path exists between source and target.\n\n    See Also\n    --------\n    shortest_path\n\n    '
    results = _bidirectional_pred_succ(G, source, target, ignore_nodes, ignore_edges)
    (pred, succ, w) = results
    path = []
    while w is not None:
        path.append(w)
        w = succ[w]
    w = pred[path[0]]
    while w is not None:
        path.insert(0, w)
        w = pred[w]
    return (len(path), path)

def _bidirectional_pred_succ(G, source, target, ignore_nodes=None, ignore_edges=None):
    if False:
        print('Hello World!')
    'Bidirectional shortest path helper.\n    Returns (pred,succ,w) where\n    pred is a dictionary of predecessors from w to the source, and\n    succ is a dictionary of successors from w to the target.\n    '
    if ignore_nodes and (source in ignore_nodes or target in ignore_nodes):
        raise nx.NetworkXNoPath(f'No path between {source} and {target}.')
    if target == source:
        return ({target: None}, {source: None}, source)
    if G.is_directed():
        Gpred = G.predecessors
        Gsucc = G.successors
    else:
        Gpred = G.neighbors
        Gsucc = G.neighbors
    if ignore_nodes:

        def filter_iter(nodes):
            if False:
                while True:
                    i = 10

            def iterate(v):
                if False:
                    for i in range(10):
                        print('nop')
                for w in nodes(v):
                    if w not in ignore_nodes:
                        yield w
            return iterate
        Gpred = filter_iter(Gpred)
        Gsucc = filter_iter(Gsucc)
    if ignore_edges:
        if G.is_directed():

            def filter_pred_iter(pred_iter):
                if False:
                    for i in range(10):
                        print('nop')

                def iterate(v):
                    if False:
                        while True:
                            i = 10
                    for w in pred_iter(v):
                        if (w, v) not in ignore_edges:
                            yield w
                return iterate

            def filter_succ_iter(succ_iter):
                if False:
                    while True:
                        i = 10

                def iterate(v):
                    if False:
                        for i in range(10):
                            print('nop')
                    for w in succ_iter(v):
                        if (v, w) not in ignore_edges:
                            yield w
                return iterate
            Gpred = filter_pred_iter(Gpred)
            Gsucc = filter_succ_iter(Gsucc)
        else:

            def filter_iter(nodes):
                if False:
                    print('Hello World!')

                def iterate(v):
                    if False:
                        i = 10
                        return i + 15
                    for w in nodes(v):
                        if (v, w) not in ignore_edges and (w, v) not in ignore_edges:
                            yield w
                return iterate
            Gpred = filter_iter(Gpred)
            Gsucc = filter_iter(Gsucc)
    pred = {source: None}
    succ = {target: None}
    forward_fringe = [source]
    reverse_fringe = [target]
    while forward_fringe and reverse_fringe:
        if len(forward_fringe) <= len(reverse_fringe):
            this_level = forward_fringe
            forward_fringe = []
            for v in this_level:
                for w in Gsucc(v):
                    if w not in pred:
                        forward_fringe.append(w)
                        pred[w] = v
                    if w in succ:
                        return (pred, succ, w)
        else:
            this_level = reverse_fringe
            reverse_fringe = []
            for v in this_level:
                for w in Gpred(v):
                    if w not in succ:
                        succ[w] = v
                        reverse_fringe.append(w)
                    if w in pred:
                        return (pred, succ, w)
    raise nx.NetworkXNoPath(f'No path between {source} and {target}.')

def _bidirectional_dijkstra(G, source, target, weight='weight', ignore_nodes=None, ignore_edges=None):
    if False:
        print('Hello World!')
    "Dijkstra's algorithm for shortest paths using bidirectional search.\n\n    This function returns the shortest path between source and target\n    ignoring nodes and edges in the containers ignore_nodes and\n    ignore_edges.\n\n    This is a custom modification of the standard Dijkstra bidirectional\n    shortest path implementation at networkx.algorithms.weighted\n\n    Parameters\n    ----------\n    G : NetworkX graph\n\n    source : node\n       Starting node.\n\n    target : node\n       Ending node.\n\n    weight: string, function, optional (default='weight')\n       Edge data key or weight function corresponding to the edge weight\n\n    ignore_nodes : container of nodes\n       nodes to ignore, optional\n\n    ignore_edges : container of edges\n       edges to ignore, optional\n\n    Returns\n    -------\n    length : number\n        Shortest path length.\n\n    Returns a tuple of two dictionaries keyed by node.\n    The first dictionary stores distance from the source.\n    The second stores the path from the source to that node.\n\n    Raises\n    ------\n    NetworkXNoPath\n        If no path exists between source and target.\n\n    Notes\n    -----\n    Edge weight attributes must be numerical.\n    Distances are calculated as sums of weighted edges traversed.\n\n    In practice  bidirectional Dijkstra is much more than twice as fast as\n    ordinary Dijkstra.\n\n    Ordinary Dijkstra expands nodes in a sphere-like manner from the\n    source. The radius of this sphere will eventually be the length\n    of the shortest path. Bidirectional Dijkstra will expand nodes\n    from both the source and the target, making two spheres of half\n    this radius. Volume of the first sphere is pi*r*r while the\n    others are 2*pi*r/2*r/2, making up half the volume.\n\n    This algorithm is not guaranteed to work if edge weights\n    are negative or are floating point numbers\n    (overflows and roundoff errors can cause problems).\n\n    See Also\n    --------\n    shortest_path\n    shortest_path_length\n    "
    if ignore_nodes and (source in ignore_nodes or target in ignore_nodes):
        raise nx.NetworkXNoPath(f'No path between {source} and {target}.')
    if source == target:
        if source not in G:
            raise nx.NodeNotFound(f'Node {source} not in graph')
        return (0, [source])
    if G.is_directed():
        Gpred = G.predecessors
        Gsucc = G.successors
    else:
        Gpred = G.neighbors
        Gsucc = G.neighbors
    if ignore_nodes:

        def filter_iter(nodes):
            if False:
                print('Hello World!')

            def iterate(v):
                if False:
                    i = 10
                    return i + 15
                for w in nodes(v):
                    if w not in ignore_nodes:
                        yield w
            return iterate
        Gpred = filter_iter(Gpred)
        Gsucc = filter_iter(Gsucc)
    if ignore_edges:
        if G.is_directed():

            def filter_pred_iter(pred_iter):
                if False:
                    print('Hello World!')

                def iterate(v):
                    if False:
                        return 10
                    for w in pred_iter(v):
                        if (w, v) not in ignore_edges:
                            yield w
                return iterate

            def filter_succ_iter(succ_iter):
                if False:
                    while True:
                        i = 10

                def iterate(v):
                    if False:
                        return 10
                    for w in succ_iter(v):
                        if (v, w) not in ignore_edges:
                            yield w
                return iterate
            Gpred = filter_pred_iter(Gpred)
            Gsucc = filter_succ_iter(Gsucc)
        else:

            def filter_iter(nodes):
                if False:
                    return 10

                def iterate(v):
                    if False:
                        return 10
                    for w in nodes(v):
                        if (v, w) not in ignore_edges and (w, v) not in ignore_edges:
                            yield w
                return iterate
            Gpred = filter_iter(Gpred)
            Gsucc = filter_iter(Gsucc)
    push = heappush
    pop = heappop
    dists = [{}, {}]
    paths = [{source: [source]}, {target: [target]}]
    fringe = [[], []]
    seen = [{source: 0}, {target: 0}]
    c = count()
    push(fringe[0], (0, next(c), source))
    push(fringe[1], (0, next(c), target))
    neighs = [Gsucc, Gpred]
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
        wt = _weight_function(G, weight)
        for w in neighs[dir](v):
            if dir == 0:
                minweight = wt(v, w, G.get_edge_data(v, w))
                vwLength = dists[dir][v] + minweight
            else:
                minweight = wt(w, v, G.get_edge_data(w, v))
                vwLength = dists[dir][v] + minweight
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