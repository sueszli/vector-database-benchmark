"""
Minimum cost flow algorithms on directed connected graphs.
"""
__all__ = ['network_simplex']
from itertools import chain, islice, repeat
from math import ceil, sqrt
import networkx as nx
from networkx.utils import not_implemented_for

class _DataEssentialsAndFunctions:

    def __init__(self, G, multigraph, demand='demand', capacity='capacity', weight='weight'):
        if False:
            print('Hello World!')
        self.node_list = list(G)
        self.node_indices = {u: i for (i, u) in enumerate(self.node_list)}
        self.node_demands = [G.nodes[u].get(demand, 0) for u in self.node_list]
        self.edge_sources = []
        self.edge_targets = []
        if multigraph:
            self.edge_keys = []
        self.edge_indices = {}
        self.edge_capacities = []
        self.edge_weights = []
        if not multigraph:
            edges = G.edges(data=True)
        else:
            edges = G.edges(data=True, keys=True)
        inf = float('inf')
        edges = (e for e in edges if e[0] != e[1] and e[-1].get(capacity, inf) != 0)
        for (i, e) in enumerate(edges):
            self.edge_sources.append(self.node_indices[e[0]])
            self.edge_targets.append(self.node_indices[e[1]])
            if multigraph:
                self.edge_keys.append(e[2])
            self.edge_indices[e[:-1]] = i
            self.edge_capacities.append(e[-1].get(capacity, inf))
            self.edge_weights.append(e[-1].get(weight, 0))
        self.edge_count = None
        self.edge_flow = None
        self.node_potentials = None
        self.parent = None
        self.parent_edge = None
        self.subtree_size = None
        self.next_node_dft = None
        self.prev_node_dft = None
        self.last_descendent_dft = None
        self._spanning_tree_initialized = False

    def initialize_spanning_tree(self, n, faux_inf):
        if False:
            for i in range(10):
                print('nop')
        self.edge_count = len(self.edge_indices)
        self.edge_flow = list(chain(repeat(0, self.edge_count), (abs(d) for d in self.node_demands)))
        self.node_potentials = [faux_inf if d <= 0 else -faux_inf for d in self.node_demands]
        self.parent = list(chain(repeat(-1, n), [None]))
        self.parent_edge = list(range(self.edge_count, self.edge_count + n))
        self.subtree_size = list(chain(repeat(1, n), [n + 1]))
        self.next_node_dft = list(chain(range(1, n), [-1, 0]))
        self.prev_node_dft = list(range(-1, n))
        self.last_descendent_dft = list(chain(range(n), [n - 1]))
        self._spanning_tree_initialized = True

    def find_apex(self, p, q):
        if False:
            i = 10
            return i + 15
        '\n        Find the lowest common ancestor of nodes p and q in the spanning tree.\n        '
        size_p = self.subtree_size[p]
        size_q = self.subtree_size[q]
        while True:
            while size_p < size_q:
                p = self.parent[p]
                size_p = self.subtree_size[p]
            while size_p > size_q:
                q = self.parent[q]
                size_q = self.subtree_size[q]
            if size_p == size_q:
                if p != q:
                    p = self.parent[p]
                    size_p = self.subtree_size[p]
                    q = self.parent[q]
                    size_q = self.subtree_size[q]
                else:
                    return p

    def trace_path(self, p, w):
        if False:
            print('Hello World!')
        '\n        Returns the nodes and edges on the path from node p to its ancestor w.\n        '
        Wn = [p]
        We = []
        while p != w:
            We.append(self.parent_edge[p])
            p = self.parent[p]
            Wn.append(p)
        return (Wn, We)

    def find_cycle(self, i, p, q):
        if False:
            print('Hello World!')
        '\n        Returns the nodes and edges on the cycle containing edge i == (p, q)\n        when the latter is added to the spanning tree.\n\n        The cycle is oriented in the direction from p to q.\n        '
        w = self.find_apex(p, q)
        (Wn, We) = self.trace_path(p, w)
        Wn.reverse()
        We.reverse()
        if We != [i]:
            We.append(i)
        (WnR, WeR) = self.trace_path(q, w)
        del WnR[-1]
        Wn += WnR
        We += WeR
        return (Wn, We)

    def augment_flow(self, Wn, We, f):
        if False:
            return 10
        '\n        Augment f units of flow along a cycle represented by Wn and We.\n        '
        for (i, p) in zip(We, Wn):
            if self.edge_sources[i] == p:
                self.edge_flow[i] += f
            else:
                self.edge_flow[i] -= f

    def trace_subtree(self, p):
        if False:
            while True:
                i = 10
        '\n        Yield the nodes in the subtree rooted at a node p.\n        '
        yield p
        l = self.last_descendent_dft[p]
        while p != l:
            p = self.next_node_dft[p]
            yield p

    def remove_edge(self, s, t):
        if False:
            print('Hello World!')
        '\n        Remove an edge (s, t) where parent[t] == s from the spanning tree.\n        '
        size_t = self.subtree_size[t]
        prev_t = self.prev_node_dft[t]
        last_t = self.last_descendent_dft[t]
        next_last_t = self.next_node_dft[last_t]
        self.parent[t] = None
        self.parent_edge[t] = None
        self.next_node_dft[prev_t] = next_last_t
        self.prev_node_dft[next_last_t] = prev_t
        self.next_node_dft[last_t] = t
        self.prev_node_dft[t] = last_t
        while s is not None:
            self.subtree_size[s] -= size_t
            if self.last_descendent_dft[s] == last_t:
                self.last_descendent_dft[s] = prev_t
            s = self.parent[s]

    def make_root(self, q):
        if False:
            for i in range(10):
                print('nop')
        '\n        Make a node q the root of its containing subtree.\n        '
        ancestors = []
        while q is not None:
            ancestors.append(q)
            q = self.parent[q]
        ancestors.reverse()
        for (p, q) in zip(ancestors, islice(ancestors, 1, None)):
            size_p = self.subtree_size[p]
            last_p = self.last_descendent_dft[p]
            prev_q = self.prev_node_dft[q]
            last_q = self.last_descendent_dft[q]
            next_last_q = self.next_node_dft[last_q]
            self.parent[p] = q
            self.parent[q] = None
            self.parent_edge[p] = self.parent_edge[q]
            self.parent_edge[q] = None
            self.subtree_size[p] = size_p - self.subtree_size[q]
            self.subtree_size[q] = size_p
            self.next_node_dft[prev_q] = next_last_q
            self.prev_node_dft[next_last_q] = prev_q
            self.next_node_dft[last_q] = q
            self.prev_node_dft[q] = last_q
            if last_p == last_q:
                self.last_descendent_dft[p] = prev_q
                last_p = prev_q
            self.prev_node_dft[p] = last_q
            self.next_node_dft[last_q] = p
            self.next_node_dft[last_p] = q
            self.prev_node_dft[q] = last_p
            self.last_descendent_dft[q] = last_p

    def add_edge(self, i, p, q):
        if False:
            for i in range(10):
                print('nop')
        '\n        Add an edge (p, q) to the spanning tree where q is the root of a subtree.\n        '
        last_p = self.last_descendent_dft[p]
        next_last_p = self.next_node_dft[last_p]
        size_q = self.subtree_size[q]
        last_q = self.last_descendent_dft[q]
        self.parent[q] = p
        self.parent_edge[q] = i
        self.next_node_dft[last_p] = q
        self.prev_node_dft[q] = last_p
        self.prev_node_dft[next_last_p] = last_q
        self.next_node_dft[last_q] = next_last_p
        while p is not None:
            self.subtree_size[p] += size_q
            if self.last_descendent_dft[p] == last_p:
                self.last_descendent_dft[p] = last_q
            p = self.parent[p]

    def update_potentials(self, i, p, q):
        if False:
            print('Hello World!')
        '\n        Update the potentials of the nodes in the subtree rooted at a node\n        q connected to its parent p by an edge i.\n        '
        if q == self.edge_targets[i]:
            d = self.node_potentials[p] - self.edge_weights[i] - self.node_potentials[q]
        else:
            d = self.node_potentials[p] + self.edge_weights[i] - self.node_potentials[q]
        for q in self.trace_subtree(q):
            self.node_potentials[q] += d

    def reduced_cost(self, i):
        if False:
            return 10
        'Returns the reduced cost of an edge i.'
        c = self.edge_weights[i] - self.node_potentials[self.edge_sources[i]] + self.node_potentials[self.edge_targets[i]]
        return c if self.edge_flow[i] == 0 else -c

    def find_entering_edges(self):
        if False:
            return 10
        'Yield entering edges until none can be found.'
        if self.edge_count == 0:
            return
        B = int(ceil(sqrt(self.edge_count)))
        M = (self.edge_count + B - 1) // B
        m = 0
        f = 0
        while m < M:
            l = f + B
            if l <= self.edge_count:
                edges = range(f, l)
            else:
                l -= self.edge_count
                edges = chain(range(f, self.edge_count), range(l))
            f = l
            i = min(edges, key=self.reduced_cost)
            c = self.reduced_cost(i)
            if c >= 0:
                m += 1
            else:
                if self.edge_flow[i] == 0:
                    p = self.edge_sources[i]
                    q = self.edge_targets[i]
                else:
                    p = self.edge_targets[i]
                    q = self.edge_sources[i]
                yield (i, p, q)
                m = 0

    def residual_capacity(self, i, p):
        if False:
            print('Hello World!')
        'Returns the residual capacity of an edge i in the direction away\n        from its endpoint p.\n        '
        return self.edge_capacities[i] - self.edge_flow[i] if self.edge_sources[i] == p else self.edge_flow[i]

    def find_leaving_edge(self, Wn, We):
        if False:
            while True:
                i = 10
        'Returns the leaving edge in a cycle represented by Wn and We.'
        (j, s) = min(zip(reversed(We), reversed(Wn)), key=lambda i_p: self.residual_capacity(*i_p))
        t = self.edge_targets[j] if self.edge_sources[j] == s else self.edge_sources[j]
        return (j, s, t)

@not_implemented_for('undirected')
@nx._dispatch(node_attrs='demand', edge_attrs={'capacity': float('inf'), 'weight': 0})
def network_simplex(G, demand='demand', capacity='capacity', weight='weight'):
    if False:
        print('Hello World!')
    'Find a minimum cost flow satisfying all demands in digraph G.\n\n    This is a primal network simplex algorithm that uses the leaving\n    arc rule to prevent cycling.\n\n    G is a digraph with edge costs and capacities and in which nodes\n    have demand, i.e., they want to send or receive some amount of\n    flow. A negative demand means that the node wants to send flow, a\n    positive demand means that the node want to receive flow. A flow on\n    the digraph G satisfies all demand if the net flow into each node\n    is equal to the demand of that node.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n        DiGraph on which a minimum cost flow satisfying all demands is\n        to be found.\n\n    demand : string\n        Nodes of the graph G are expected to have an attribute demand\n        that indicates how much flow a node wants to send (negative\n        demand) or receive (positive demand). Note that the sum of the\n        demands should be 0 otherwise the problem in not feasible. If\n        this attribute is not present, a node is considered to have 0\n        demand. Default value: \'demand\'.\n\n    capacity : string\n        Edges of the graph G are expected to have an attribute capacity\n        that indicates how much flow the edge can support. If this\n        attribute is not present, the edge is considered to have\n        infinite capacity. Default value: \'capacity\'.\n\n    weight : string\n        Edges of the graph G are expected to have an attribute weight\n        that indicates the cost incurred by sending one unit of flow on\n        that edge. If not present, the weight is considered to be 0.\n        Default value: \'weight\'.\n\n    Returns\n    -------\n    flowCost : integer, float\n        Cost of a minimum cost flow satisfying all demands.\n\n    flowDict : dictionary\n        Dictionary of dictionaries keyed by nodes such that\n        flowDict[u][v] is the flow edge (u, v).\n\n    Raises\n    ------\n    NetworkXError\n        This exception is raised if the input graph is not directed or\n        not connected.\n\n    NetworkXUnfeasible\n        This exception is raised in the following situations:\n\n            * The sum of the demands is not zero. Then, there is no\n              flow satisfying all demands.\n            * There is no flow satisfying all demand.\n\n    NetworkXUnbounded\n        This exception is raised if the digraph G has a cycle of\n        negative cost and infinite capacity. Then, the cost of a flow\n        satisfying all demands is unbounded below.\n\n    Notes\n    -----\n    This algorithm is not guaranteed to work if edge weights or demands\n    are floating point numbers (overflows and roundoff errors can\n    cause problems). As a workaround you can use integer numbers by\n    multiplying the relevant edge attributes by a convenient\n    constant factor (eg 100).\n\n    See also\n    --------\n    cost_of_flow, max_flow_min_cost, min_cost_flow, min_cost_flow_cost\n\n    Examples\n    --------\n    A simple example of a min cost flow problem.\n\n    >>> G = nx.DiGraph()\n    >>> G.add_node("a", demand=-5)\n    >>> G.add_node("d", demand=5)\n    >>> G.add_edge("a", "b", weight=3, capacity=4)\n    >>> G.add_edge("a", "c", weight=6, capacity=10)\n    >>> G.add_edge("b", "d", weight=1, capacity=9)\n    >>> G.add_edge("c", "d", weight=2, capacity=5)\n    >>> flowCost, flowDict = nx.network_simplex(G)\n    >>> flowCost\n    24\n    >>> flowDict\n    {\'a\': {\'b\': 4, \'c\': 1}, \'d\': {}, \'b\': {\'d\': 4}, \'c\': {\'d\': 1}}\n\n    The mincost flow algorithm can also be used to solve shortest path\n    problems. To find the shortest path between two nodes u and v,\n    give all edges an infinite capacity, give node u a demand of -1 and\n    node v a demand a 1. Then run the network simplex. The value of a\n    min cost flow will be the distance between u and v and edges\n    carrying positive flow will indicate the path.\n\n    >>> G = nx.DiGraph()\n    >>> G.add_weighted_edges_from(\n    ...     [\n    ...         ("s", "u", 10),\n    ...         ("s", "x", 5),\n    ...         ("u", "v", 1),\n    ...         ("u", "x", 2),\n    ...         ("v", "y", 1),\n    ...         ("x", "u", 3),\n    ...         ("x", "v", 5),\n    ...         ("x", "y", 2),\n    ...         ("y", "s", 7),\n    ...         ("y", "v", 6),\n    ...     ]\n    ... )\n    >>> G.add_node("s", demand=-1)\n    >>> G.add_node("v", demand=1)\n    >>> flowCost, flowDict = nx.network_simplex(G)\n    >>> flowCost == nx.shortest_path_length(G, "s", "v", weight="weight")\n    True\n    >>> sorted([(u, v) for u in flowDict for v in flowDict[u] if flowDict[u][v] > 0])\n    [(\'s\', \'x\'), (\'u\', \'v\'), (\'x\', \'u\')]\n    >>> nx.shortest_path(G, "s", "v", weight="weight")\n    [\'s\', \'x\', \'u\', \'v\']\n\n    It is possible to change the name of the attributes used for the\n    algorithm.\n\n    >>> G = nx.DiGraph()\n    >>> G.add_node("p", spam=-4)\n    >>> G.add_node("q", spam=2)\n    >>> G.add_node("a", spam=-2)\n    >>> G.add_node("d", spam=-1)\n    >>> G.add_node("t", spam=2)\n    >>> G.add_node("w", spam=3)\n    >>> G.add_edge("p", "q", cost=7, vacancies=5)\n    >>> G.add_edge("p", "a", cost=1, vacancies=4)\n    >>> G.add_edge("q", "d", cost=2, vacancies=3)\n    >>> G.add_edge("t", "q", cost=1, vacancies=2)\n    >>> G.add_edge("a", "t", cost=2, vacancies=4)\n    >>> G.add_edge("d", "w", cost=3, vacancies=4)\n    >>> G.add_edge("t", "w", cost=4, vacancies=1)\n    >>> flowCost, flowDict = nx.network_simplex(\n    ...     G, demand="spam", capacity="vacancies", weight="cost"\n    ... )\n    >>> flowCost\n    37\n    >>> flowDict\n    {\'p\': {\'q\': 2, \'a\': 2}, \'q\': {\'d\': 1}, \'a\': {\'t\': 4}, \'d\': {\'w\': 2}, \'t\': {\'q\': 1, \'w\': 1}, \'w\': {}}\n\n    References\n    ----------\n    .. [1] Z. Kiraly, P. Kovacs.\n           Efficient implementation of minimum-cost flow algorithms.\n           Acta Universitatis Sapientiae, Informatica 4(1):67--118. 2012.\n    .. [2] R. Barr, F. Glover, D. Klingman.\n           Enhancement of spanning tree labeling procedures for network\n           optimization.\n           INFOR 17(1):16--34. 1979.\n    '
    if len(G) == 0:
        raise nx.NetworkXError('graph has no nodes')
    multigraph = G.is_multigraph()
    DEAF = _DataEssentialsAndFunctions(G, multigraph, demand=demand, capacity=capacity, weight=weight)
    inf = float('inf')
    for (u, d) in zip(DEAF.node_list, DEAF.node_demands):
        if abs(d) == inf:
            raise nx.NetworkXError(f'node {u!r} has infinite demand')
    for (e, w) in zip(DEAF.edge_indices, DEAF.edge_weights):
        if abs(w) == inf:
            raise nx.NetworkXError(f'edge {e!r} has infinite weight')
    if not multigraph:
        edges = nx.selfloop_edges(G, data=True)
    else:
        edges = nx.selfloop_edges(G, data=True, keys=True)
    for e in edges:
        if abs(e[-1].get(weight, 0)) == inf:
            raise nx.NetworkXError(f'edge {e[:-1]!r} has infinite weight')
    if sum(DEAF.node_demands) != 0:
        raise nx.NetworkXUnfeasible('total node demand is not zero')
    for (e, c) in zip(DEAF.edge_indices, DEAF.edge_capacities):
        if c < 0:
            raise nx.NetworkXUnfeasible(f'edge {e!r} has negative capacity')
    if not multigraph:
        edges = nx.selfloop_edges(G, data=True)
    else:
        edges = nx.selfloop_edges(G, data=True, keys=True)
    for e in edges:
        if e[-1].get(capacity, inf) < 0:
            raise nx.NetworkXUnfeasible(f'edge {e[:-1]!r} has negative capacity')
    for (i, d) in enumerate(DEAF.node_demands):
        if d > 0:
            DEAF.edge_sources.append(-1)
            DEAF.edge_targets.append(i)
        else:
            DEAF.edge_sources.append(i)
            DEAF.edge_targets.append(-1)
    faux_inf = 3 * max(chain([sum((c for c in DEAF.edge_capacities if c < inf)), sum((abs(w) for w in DEAF.edge_weights))], (abs(d) for d in DEAF.node_demands))) or 1
    n = len(DEAF.node_list)
    DEAF.edge_weights.extend(repeat(faux_inf, n))
    DEAF.edge_capacities.extend(repeat(faux_inf, n))
    DEAF.initialize_spanning_tree(n, faux_inf)
    for (i, p, q) in DEAF.find_entering_edges():
        (Wn, We) = DEAF.find_cycle(i, p, q)
        (j, s, t) = DEAF.find_leaving_edge(Wn, We)
        DEAF.augment_flow(Wn, We, DEAF.residual_capacity(j, s))
        if i != j:
            if DEAF.parent[t] != s:
                (s, t) = (t, s)
            if We.index(i) > We.index(j):
                (p, q) = (q, p)
            DEAF.remove_edge(s, t)
            DEAF.make_root(q)
            DEAF.add_edge(i, p, q)
            DEAF.update_potentials(i, p, q)
    if any((DEAF.edge_flow[i] != 0 for i in range(-n, 0))):
        raise nx.NetworkXUnfeasible('no flow satisfies all node demands')
    if any((DEAF.edge_flow[i] * 2 >= faux_inf for i in range(DEAF.edge_count))) or any((e[-1].get(capacity, inf) == inf and e[-1].get(weight, 0) < 0 for e in nx.selfloop_edges(G, data=True))):
        raise nx.NetworkXUnbounded('negative cycle with infinite capacity found')
    del DEAF.edge_flow[DEAF.edge_count:]
    flow_cost = sum((w * x for (w, x) in zip(DEAF.edge_weights, DEAF.edge_flow)))
    flow_dict = {n: {} for n in DEAF.node_list}

    def add_entry(e):
        if False:
            while True:
                i = 10
        'Add a flow dict entry.'
        d = flow_dict[e[0]]
        for k in e[1:-2]:
            try:
                d = d[k]
            except KeyError:
                t = {}
                d[k] = t
                d = t
        d[e[-2]] = e[-1]
    DEAF.edge_sources = (DEAF.node_list[s] for s in DEAF.edge_sources)
    DEAF.edge_targets = (DEAF.node_list[t] for t in DEAF.edge_targets)
    if not multigraph:
        for e in zip(DEAF.edge_sources, DEAF.edge_targets, DEAF.edge_flow):
            add_entry(e)
        edges = G.edges(data=True)
    else:
        for e in zip(DEAF.edge_sources, DEAF.edge_targets, DEAF.edge_keys, DEAF.edge_flow):
            add_entry(e)
        edges = G.edges(data=True, keys=True)
    for e in edges:
        if e[0] != e[1]:
            if e[-1].get(capacity, inf) == 0:
                add_entry(e[:-1] + (0,))
        else:
            w = e[-1].get(weight, 0)
            if w >= 0:
                add_entry(e[:-1] + (0,))
            else:
                c = e[-1][capacity]
                flow_cost += w * c
                add_entry(e[:-1] + (c,))
    return (flow_cost, flow_dict)