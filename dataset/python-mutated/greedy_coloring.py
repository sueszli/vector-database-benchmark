"""
Greedy graph coloring using various strategies.
"""
import itertools
from collections import defaultdict, deque
import networkx as nx
from networkx.utils import arbitrary_element, py_random_state
__all__ = ['greedy_color', 'strategy_connected_sequential', 'strategy_connected_sequential_bfs', 'strategy_connected_sequential_dfs', 'strategy_independent_set', 'strategy_largest_first', 'strategy_random_sequential', 'strategy_saturation_largest_first', 'strategy_smallest_last']

@nx._dispatch
def strategy_largest_first(G, colors):
    if False:
        print('Hello World!')
    'Returns a list of the nodes of ``G`` in decreasing order by\n    degree.\n\n    ``G`` is a NetworkX graph. ``colors`` is ignored.\n\n    '
    return sorted(G, key=G.degree, reverse=True)

@py_random_state(2)
@nx._dispatch
def strategy_random_sequential(G, colors, seed=None):
    if False:
        for i in range(10):
            print('nop')
    'Returns a random permutation of the nodes of ``G`` as a list.\n\n    ``G`` is a NetworkX graph. ``colors`` is ignored.\n\n    seed : integer, random_state, or None (default)\n        Indicator of random number generation state.\n        See :ref:`Randomness<randomness>`.\n    '
    nodes = list(G)
    seed.shuffle(nodes)
    return nodes

@nx._dispatch
def strategy_smallest_last(G, colors):
    if False:
        i = 10
        return i + 15
    'Returns a deque of the nodes of ``G``, "smallest" last.\n\n    Specifically, the degrees of each node are tracked in a bucket queue.\n    From this, the node of minimum degree is repeatedly popped from the\n    graph, updating its neighbors\' degrees.\n\n    ``G`` is a NetworkX graph. ``colors`` is ignored.\n\n    This implementation of the strategy runs in $O(n + m)$ time\n    (ignoring polylogarithmic factors), where $n$ is the number of nodes\n    and $m$ is the number of edges.\n\n    This strategy is related to :func:`strategy_independent_set`: if we\n    interpret each node removed as an independent set of size one, then\n    this strategy chooses an independent set of size one instead of a\n    maximal independent set.\n\n    '
    H = G.copy()
    result = deque()
    degrees = defaultdict(set)
    lbound = float('inf')
    for (node, d) in H.degree():
        degrees[d].add(node)
        lbound = min(lbound, d)

    def find_min_degree():
        if False:
            while True:
                i = 10
        return next((d for d in itertools.count(lbound) if d in degrees))
    for _ in G:
        min_degree = find_min_degree()
        u = degrees[min_degree].pop()
        if not degrees[min_degree]:
            del degrees[min_degree]
        result.appendleft(u)
        for v in H[u]:
            degree = H.degree(v)
            degrees[degree].remove(v)
            if not degrees[degree]:
                del degrees[degree]
            degrees[degree - 1].add(v)
        H.remove_node(u)
        lbound = min_degree - 1
    return result

def _maximal_independent_set(G):
    if False:
        i = 10
        return i + 15
    'Returns a maximal independent set of nodes in ``G`` by repeatedly\n    choosing an independent node of minimum degree (with respect to the\n    subgraph of unchosen nodes).\n\n    '
    result = set()
    remaining = set(G)
    while remaining:
        G = G.subgraph(remaining)
        v = min(remaining, key=G.degree)
        result.add(v)
        remaining -= set(G[v]) | {v}
    return result

@nx._dispatch
def strategy_independent_set(G, colors):
    if False:
        while True:
            i = 10
    'Uses a greedy independent set removal strategy to determine the\n    colors.\n\n    This function updates ``colors`` **in-place** and return ``None``,\n    unlike the other strategy functions in this module.\n\n    This algorithm repeatedly finds and removes a maximal independent\n    set, assigning each node in the set an unused color.\n\n    ``G`` is a NetworkX graph.\n\n    This strategy is related to :func:`strategy_smallest_last`: in that\n    strategy, an independent set of size one is chosen at each step\n    instead of a maximal independent set.\n\n    '
    remaining_nodes = set(G)
    while len(remaining_nodes) > 0:
        nodes = _maximal_independent_set(G.subgraph(remaining_nodes))
        remaining_nodes -= nodes
        yield from nodes

@nx._dispatch
def strategy_connected_sequential_bfs(G, colors):
    if False:
        while True:
            i = 10
    'Returns an iterable over nodes in ``G`` in the order given by a\n    breadth-first traversal.\n\n    The generated sequence has the property that for each node except\n    the first, at least one neighbor appeared earlier in the sequence.\n\n    ``G`` is a NetworkX graph. ``colors`` is ignored.\n\n    '
    return strategy_connected_sequential(G, colors, 'bfs')

@nx._dispatch
def strategy_connected_sequential_dfs(G, colors):
    if False:
        print('Hello World!')
    'Returns an iterable over nodes in ``G`` in the order given by a\n    depth-first traversal.\n\n    The generated sequence has the property that for each node except\n    the first, at least one neighbor appeared earlier in the sequence.\n\n    ``G`` is a NetworkX graph. ``colors`` is ignored.\n\n    '
    return strategy_connected_sequential(G, colors, 'dfs')

@nx._dispatch
def strategy_connected_sequential(G, colors, traversal='bfs'):
    if False:
        return 10
    "Returns an iterable over nodes in ``G`` in the order given by a\n    breadth-first or depth-first traversal.\n\n    ``traversal`` must be one of the strings ``'dfs'`` or ``'bfs'``,\n    representing depth-first traversal or breadth-first traversal,\n    respectively.\n\n    The generated sequence has the property that for each node except\n    the first, at least one neighbor appeared earlier in the sequence.\n\n    ``G`` is a NetworkX graph. ``colors`` is ignored.\n\n    "
    if traversal == 'bfs':
        traverse = nx.bfs_edges
    elif traversal == 'dfs':
        traverse = nx.dfs_edges
    else:
        raise nx.NetworkXError("Please specify one of the strings 'bfs' or 'dfs' for connected sequential ordering")
    for component in nx.connected_components(G):
        source = arbitrary_element(component)
        yield source
        for (_, end) in traverse(G.subgraph(component), source):
            yield end

@nx._dispatch
def strategy_saturation_largest_first(G, colors):
    if False:
        while True:
            i = 10
    'Iterates over all the nodes of ``G`` in "saturation order" (also\n    known as "DSATUR").\n\n    ``G`` is a NetworkX graph. ``colors`` is a dictionary mapping nodes of\n    ``G`` to colors, for those nodes that have already been colored.\n\n    '
    distinct_colors = {v: set() for v in G}
    for (node, color) in colors.items():
        for neighbor in G[node]:
            distinct_colors[neighbor].add(color)
    if len(colors) >= 2:
        for (node, color) in colors.items():
            if color in distinct_colors[node]:
                raise nx.NetworkXError('Neighboring nodes must have different colors')
    if not colors:
        node = max(G, key=G.degree)
        yield node
        for v in G[node]:
            distinct_colors[v].add(0)
    while len(G) != len(colors):
        for (node, color) in colors.items():
            for neighbor in G[node]:
                distinct_colors[neighbor].add(color)
        saturation = {v: len(c) for (v, c) in distinct_colors.items() if v not in colors}
        node = max(saturation, key=lambda v: (saturation[v], G.degree(v)))
        yield node
STRATEGIES = {'largest_first': strategy_largest_first, 'random_sequential': strategy_random_sequential, 'smallest_last': strategy_smallest_last, 'independent_set': strategy_independent_set, 'connected_sequential_bfs': strategy_connected_sequential_bfs, 'connected_sequential_dfs': strategy_connected_sequential_dfs, 'connected_sequential': strategy_connected_sequential, 'saturation_largest_first': strategy_saturation_largest_first, 'DSATUR': strategy_saturation_largest_first}

@nx._dispatch
def greedy_color(G, strategy='largest_first', interchange=False):
    if False:
        while True:
            i = 10
    'Color a graph using various strategies of greedy graph coloring.\n\n    Attempts to color a graph using as few colors as possible, where no\n    neighbours of a node can have same color as the node itself. The\n    given strategy determines the order in which nodes are colored.\n\n    The strategies are described in [1]_, and smallest-last is based on\n    [2]_.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n\n    strategy : string or function(G, colors)\n       A function (or a string representing a function) that provides\n       the coloring strategy, by returning nodes in the ordering they\n       should be colored. ``G`` is the graph, and ``colors`` is a\n       dictionary of the currently assigned colors, keyed by nodes. The\n       function must return an iterable over all the nodes in ``G``.\n\n       If the strategy function is an iterator generator (that is, a\n       function with ``yield`` statements), keep in mind that the\n       ``colors`` dictionary will be updated after each ``yield``, since\n       this function chooses colors greedily.\n\n       If ``strategy`` is a string, it must be one of the following,\n       each of which represents one of the built-in strategy functions.\n\n       * ``\'largest_first\'``\n       * ``\'random_sequential\'``\n       * ``\'smallest_last\'``\n       * ``\'independent_set\'``\n       * ``\'connected_sequential_bfs\'``\n       * ``\'connected_sequential_dfs\'``\n       * ``\'connected_sequential\'`` (alias for the previous strategy)\n       * ``\'saturation_largest_first\'``\n       * ``\'DSATUR\'`` (alias for the previous strategy)\n\n    interchange: bool\n       Will use the color interchange algorithm described by [3]_ if set\n       to ``True``.\n\n       Note that ``saturation_largest_first`` and ``independent_set``\n       do not work with interchange. Furthermore, if you use\n       interchange with your own strategy function, you cannot rely\n       on the values in the ``colors`` argument.\n\n    Returns\n    -------\n    A dictionary with keys representing nodes and values representing\n    corresponding coloring.\n\n    Examples\n    --------\n    >>> G = nx.cycle_graph(4)\n    >>> d = nx.coloring.greedy_color(G, strategy="largest_first")\n    >>> d in [{0: 0, 1: 1, 2: 0, 3: 1}, {0: 1, 1: 0, 2: 1, 3: 0}]\n    True\n\n    Raises\n    ------\n    NetworkXPointlessConcept\n        If ``strategy`` is ``saturation_largest_first`` or\n        ``independent_set`` and ``interchange`` is ``True``.\n\n    References\n    ----------\n    .. [1] Adrian Kosowski, and Krzysztof Manuszewski,\n       Classical Coloring of Graphs, Graph Colorings, 2-19, 2004.\n       ISBN 0-8218-3458-4.\n    .. [2] David W. Matula, and Leland L. Beck, "Smallest-last\n       ordering and clustering and graph coloring algorithms." *J. ACM* 30,\n       3 (July 1983), 417–427. <https://doi.org/10.1145/2402.322385>\n    .. [3] Maciej M. Sysło, Narsingh Deo, Janusz S. Kowalik,\n       Discrete Optimization Algorithms with Pascal Programs, 415-424, 1983.\n       ISBN 0-486-45353-7.\n\n    '
    if len(G) == 0:
        return {}
    strategy = STRATEGIES.get(strategy, strategy)
    if not callable(strategy):
        raise nx.NetworkXError(f'strategy must be callable or a valid string. {strategy} not valid.')
    if interchange:
        if strategy is strategy_independent_set:
            msg = 'interchange cannot be used with independent_set'
            raise nx.NetworkXPointlessConcept(msg)
        if strategy is strategy_saturation_largest_first:
            msg = 'interchange cannot be used with saturation_largest_first'
            raise nx.NetworkXPointlessConcept(msg)
    colors = {}
    nodes = strategy(G, colors)
    if interchange:
        return _greedy_coloring_with_interchange(G, nodes)
    for u in nodes:
        neighbour_colors = {colors[v] for v in G[u] if v in colors}
        for color in itertools.count():
            if color not in neighbour_colors:
                break
        colors[u] = color
    return colors

class _Node:
    __slots__ = ['node_id', 'color', 'adj_list', 'adj_color']

    def __init__(self, node_id, n):
        if False:
            print('Hello World!')
        self.node_id = node_id
        self.color = -1
        self.adj_list = None
        self.adj_color = [None for _ in range(n)]

    def __repr__(self):
        if False:
            print('Hello World!')
        return f'Node_id: {self.node_id}, Color: {self.color}, Adj_list: ({self.adj_list}), adj_color: ({self.adj_color})'

    def assign_color(self, adj_entry, color):
        if False:
            for i in range(10):
                print('nop')
        adj_entry.col_prev = None
        adj_entry.col_next = self.adj_color[color]
        self.adj_color[color] = adj_entry
        if adj_entry.col_next is not None:
            adj_entry.col_next.col_prev = adj_entry

    def clear_color(self, adj_entry, color):
        if False:
            while True:
                i = 10
        if adj_entry.col_prev is None:
            self.adj_color[color] = adj_entry.col_next
        else:
            adj_entry.col_prev.col_next = adj_entry.col_next
        if adj_entry.col_next is not None:
            adj_entry.col_next.col_prev = adj_entry.col_prev

    def iter_neighbors(self):
        if False:
            for i in range(10):
                print('nop')
        adj_node = self.adj_list
        while adj_node is not None:
            yield adj_node
            adj_node = adj_node.next

    def iter_neighbors_color(self, color):
        if False:
            while True:
                i = 10
        adj_color_node = self.adj_color[color]
        while adj_color_node is not None:
            yield adj_color_node.node_id
            adj_color_node = adj_color_node.col_next

class _AdjEntry:
    __slots__ = ['node_id', 'next', 'mate', 'col_next', 'col_prev']

    def __init__(self, node_id):
        if False:
            i = 10
            return i + 15
        self.node_id = node_id
        self.next = None
        self.mate = None
        self.col_next = None
        self.col_prev = None

    def __repr__(self):
        if False:
            return 10
        col_next = None if self.col_next is None else self.col_next.node_id
        col_prev = None if self.col_prev is None else self.col_prev.node_id
        return f'Node_id: {self.node_id}, Next: ({self.next}), Mate: ({self.mate.node_id}), col_next: ({col_next}), col_prev: ({col_prev})'

def _greedy_coloring_with_interchange(G, nodes):
    if False:
        print('Hello World!')
    'Return a coloring for `original_graph` using interchange approach\n\n    This procedure is an adaption of the algorithm described by [1]_,\n    and is an implementation of coloring with interchange. Please be\n    advised, that the datastructures used are rather complex because\n    they are optimized to minimize the time spent identifying\n    subcomponents of the graph, which are possible candidates for color\n    interchange.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n        The graph to be colored\n\n    nodes : list\n        nodes ordered using the strategy of choice\n\n    Returns\n    -------\n    dict :\n        A dictionary keyed by node to a color value\n\n    References\n    ----------\n    .. [1] Maciej M. Syslo, Narsingh Deo, Janusz S. Kowalik,\n       Discrete Optimization Algorithms with Pascal Programs, 415-424, 1983.\n       ISBN 0-486-45353-7.\n    '
    n = len(G)
    graph = {node: _Node(node, n) for node in G}
    for (node1, node2) in G.edges():
        adj_entry1 = _AdjEntry(node2)
        adj_entry2 = _AdjEntry(node1)
        adj_entry1.mate = adj_entry2
        adj_entry2.mate = adj_entry1
        node1_head = graph[node1].adj_list
        adj_entry1.next = node1_head
        graph[node1].adj_list = adj_entry1
        node2_head = graph[node2].adj_list
        adj_entry2.next = node2_head
        graph[node2].adj_list = adj_entry2
    k = 0
    for node in nodes:
        neighbors = graph[node].iter_neighbors()
        col_used = {graph[adj_node.node_id].color for adj_node in neighbors}
        col_used.discard(-1)
        k1 = next(itertools.dropwhile(lambda x: x in col_used, itertools.count()))
        if k1 > k:
            connected = True
            visited = set()
            col1 = -1
            col2 = -1
            while connected and col1 < k:
                col1 += 1
                neighbor_cols = graph[node].iter_neighbors_color(col1)
                col1_adj = list(neighbor_cols)
                col2 = col1
                while connected and col2 < k:
                    col2 += 1
                    visited = set(col1_adj)
                    frontier = list(col1_adj)
                    i = 0
                    while i < len(frontier):
                        search_node = frontier[i]
                        i += 1
                        col_opp = col2 if graph[search_node].color == col1 else col1
                        neighbor_cols = graph[search_node].iter_neighbors_color(col_opp)
                        for neighbor in neighbor_cols:
                            if neighbor not in visited:
                                visited.add(neighbor)
                                frontier.append(neighbor)
                    connected = len(visited.intersection(graph[node].iter_neighbors_color(col2))) > 0
            if not connected:
                for search_node in visited:
                    graph[search_node].color = col2 if graph[search_node].color == col1 else col1
                    col2_adj = graph[search_node].adj_color[col2]
                    graph[search_node].adj_color[col2] = graph[search_node].adj_color[col1]
                    graph[search_node].adj_color[col1] = col2_adj
                for search_node in visited:
                    col = graph[search_node].color
                    col_opp = col1 if col == col2 else col2
                    for adj_node in graph[search_node].iter_neighbors():
                        if graph[adj_node.node_id].color != col_opp:
                            adj_mate = adj_node.mate
                            graph[adj_node.node_id].clear_color(adj_mate, col_opp)
                            graph[adj_node.node_id].assign_color(adj_mate, col)
                k1 = col1
        graph[node].color = k1
        k = max(k1, k)
        for adj_node in graph[node].iter_neighbors():
            adj_mate = adj_node.mate
            graph[adj_node.node_id].assign_color(adj_mate, k1)
    return {node.node_id: node.color for node in graph.values()}