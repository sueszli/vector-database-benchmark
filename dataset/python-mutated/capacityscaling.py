"""
Capacity scaling minimum cost flow algorithm.
"""
__all__ = ['capacity_scaling']
from itertools import chain
from math import log
import networkx as nx
from ...utils import BinaryHeap, arbitrary_element, not_implemented_for

def _detect_unboundedness(R):
    if False:
        return 10
    'Detect infinite-capacity negative cycles.'
    G = nx.DiGraph()
    G.add_nodes_from(R)
    inf = R.graph['inf']
    f_inf = float('inf')
    for u in R:
        for (v, e) in R[u].items():
            w = f_inf
            for (k, e) in e.items():
                if e['capacity'] == inf:
                    w = min(w, e['weight'])
            if w != f_inf:
                G.add_edge(u, v, weight=w)
    if nx.negative_edge_cycle(G):
        raise nx.NetworkXUnbounded('Negative cost cycle of infinite capacity found. Min cost flow may be unbounded below.')

@not_implemented_for('undirected')
def _build_residual_network(G, demand, capacity, weight):
    if False:
        while True:
            i = 10
    'Build a residual network and initialize a zero flow.'
    if sum((G.nodes[u].get(demand, 0) for u in G)) != 0:
        raise nx.NetworkXUnfeasible('Sum of the demands should be 0.')
    R = nx.MultiDiGraph()
    R.add_nodes_from(((u, {'excess': -G.nodes[u].get(demand, 0), 'potential': 0}) for u in G))
    inf = float('inf')
    for (u, v, e) in nx.selfloop_edges(G, data=True):
        if e.get(weight, 0) < 0 and e.get(capacity, inf) == inf:
            raise nx.NetworkXUnbounded('Negative cost cycle of infinite capacity found. Min cost flow may be unbounded below.')
    if G.is_multigraph():
        edge_list = [(u, v, k, e) for (u, v, k, e) in G.edges(data=True, keys=True) if u != v and e.get(capacity, inf) > 0]
    else:
        edge_list = [(u, v, 0, e) for (u, v, e) in G.edges(data=True) if u != v and e.get(capacity, inf) > 0]
    inf = max(sum((abs(R.nodes[u]['excess']) for u in R)), 2 * sum((e[capacity] for (u, v, k, e) in edge_list if capacity in e and e[capacity] != inf))) or 1
    for (u, v, k, e) in edge_list:
        r = min(e.get(capacity, inf), inf)
        w = e.get(weight, 0)
        R.add_edge(u, v, key=(k, True), capacity=r, weight=w, flow=0)
        R.add_edge(v, u, key=(k, False), capacity=0, weight=-w, flow=0)
    R.graph['inf'] = inf
    _detect_unboundedness(R)
    return R

def _build_flow_dict(G, R, capacity, weight):
    if False:
        print('Hello World!')
    'Build a flow dictionary from a residual network.'
    inf = float('inf')
    flow_dict = {}
    if G.is_multigraph():
        for u in G:
            flow_dict[u] = {}
            for (v, es) in G[u].items():
                flow_dict[u][v] = {k: 0 if u != v or e.get(capacity, inf) <= 0 or e.get(weight, 0) >= 0 else e[capacity] for (k, e) in es.items()}
            for (v, es) in R[u].items():
                if v in flow_dict[u]:
                    flow_dict[u][v].update(((k[0], e['flow']) for (k, e) in es.items() if e['flow'] > 0))
    else:
        for u in G:
            flow_dict[u] = {v: 0 if u != v or e.get(capacity, inf) <= 0 or e.get(weight, 0) >= 0 else e[capacity] for (v, e) in G[u].items()}
            flow_dict[u].update(((v, e['flow']) for (v, es) in R[u].items() for e in es.values() if e['flow'] > 0))
    return flow_dict

@nx._dispatch(node_attrs='demand', edge_attrs={'capacity': float('inf'), 'weight': 0})
def capacity_scaling(G, demand='demand', capacity='capacity', weight='weight', heap=BinaryHeap):
    if False:
        return 10
    'Find a minimum cost flow satisfying all demands in digraph G.\n\n    This is a capacity scaling successive shortest augmenting path algorithm.\n\n    G is a digraph with edge costs and capacities and in which nodes\n    have demand, i.e., they want to send or receive some amount of\n    flow. A negative demand means that the node wants to send flow, a\n    positive demand means that the node want to receive flow. A flow on\n    the digraph G satisfies all demand if the net flow into each node\n    is equal to the demand of that node.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n        DiGraph or MultiDiGraph on which a minimum cost flow satisfying all\n        demands is to be found.\n\n    demand : string\n        Nodes of the graph G are expected to have an attribute demand\n        that indicates how much flow a node wants to send (negative\n        demand) or receive (positive demand). Note that the sum of the\n        demands should be 0 otherwise the problem in not feasible. If\n        this attribute is not present, a node is considered to have 0\n        demand. Default value: \'demand\'.\n\n    capacity : string\n        Edges of the graph G are expected to have an attribute capacity\n        that indicates how much flow the edge can support. If this\n        attribute is not present, the edge is considered to have\n        infinite capacity. Default value: \'capacity\'.\n\n    weight : string\n        Edges of the graph G are expected to have an attribute weight\n        that indicates the cost incurred by sending one unit of flow on\n        that edge. If not present, the weight is considered to be 0.\n        Default value: \'weight\'.\n\n    heap : class\n        Type of heap to be used in the algorithm. It should be a subclass of\n        :class:`MinHeap` or implement a compatible interface.\n\n        If a stock heap implementation is to be used, :class:`BinaryHeap` is\n        recommended over :class:`PairingHeap` for Python implementations without\n        optimized attribute accesses (e.g., CPython) despite a slower\n        asymptotic running time. For Python implementations with optimized\n        attribute accesses (e.g., PyPy), :class:`PairingHeap` provides better\n        performance. Default value: :class:`BinaryHeap`.\n\n    Returns\n    -------\n    flowCost : integer\n        Cost of a minimum cost flow satisfying all demands.\n\n    flowDict : dictionary\n        If G is a digraph, a dict-of-dicts keyed by nodes such that\n        flowDict[u][v] is the flow on edge (u, v).\n        If G is a MultiDiGraph, a dict-of-dicts-of-dicts keyed by nodes\n        so that flowDict[u][v][key] is the flow on edge (u, v, key).\n\n    Raises\n    ------\n    NetworkXError\n        This exception is raised if the input graph is not directed,\n        not connected.\n\n    NetworkXUnfeasible\n        This exception is raised in the following situations:\n\n            * The sum of the demands is not zero. Then, there is no\n              flow satisfying all demands.\n            * There is no flow satisfying all demand.\n\n    NetworkXUnbounded\n        This exception is raised if the digraph G has a cycle of\n        negative cost and infinite capacity. Then, the cost of a flow\n        satisfying all demands is unbounded below.\n\n    Notes\n    -----\n    This algorithm does not work if edge weights are floating-point numbers.\n\n    See also\n    --------\n    :meth:`network_simplex`\n\n    Examples\n    --------\n    A simple example of a min cost flow problem.\n\n    >>> G = nx.DiGraph()\n    >>> G.add_node("a", demand=-5)\n    >>> G.add_node("d", demand=5)\n    >>> G.add_edge("a", "b", weight=3, capacity=4)\n    >>> G.add_edge("a", "c", weight=6, capacity=10)\n    >>> G.add_edge("b", "d", weight=1, capacity=9)\n    >>> G.add_edge("c", "d", weight=2, capacity=5)\n    >>> flowCost, flowDict = nx.capacity_scaling(G)\n    >>> flowCost\n    24\n    >>> flowDict\n    {\'a\': {\'b\': 4, \'c\': 1}, \'d\': {}, \'b\': {\'d\': 4}, \'c\': {\'d\': 1}}\n\n    It is possible to change the name of the attributes used for the\n    algorithm.\n\n    >>> G = nx.DiGraph()\n    >>> G.add_node("p", spam=-4)\n    >>> G.add_node("q", spam=2)\n    >>> G.add_node("a", spam=-2)\n    >>> G.add_node("d", spam=-1)\n    >>> G.add_node("t", spam=2)\n    >>> G.add_node("w", spam=3)\n    >>> G.add_edge("p", "q", cost=7, vacancies=5)\n    >>> G.add_edge("p", "a", cost=1, vacancies=4)\n    >>> G.add_edge("q", "d", cost=2, vacancies=3)\n    >>> G.add_edge("t", "q", cost=1, vacancies=2)\n    >>> G.add_edge("a", "t", cost=2, vacancies=4)\n    >>> G.add_edge("d", "w", cost=3, vacancies=4)\n    >>> G.add_edge("t", "w", cost=4, vacancies=1)\n    >>> flowCost, flowDict = nx.capacity_scaling(\n    ...     G, demand="spam", capacity="vacancies", weight="cost"\n    ... )\n    >>> flowCost\n    37\n    >>> flowDict\n    {\'p\': {\'q\': 2, \'a\': 2}, \'q\': {\'d\': 1}, \'a\': {\'t\': 4}, \'d\': {\'w\': 2}, \'t\': {\'q\': 1, \'w\': 1}, \'w\': {}}\n    '
    R = _build_residual_network(G, demand, capacity, weight)
    inf = float('inf')
    flow_cost = sum((0 if e.get(capacity, inf) <= 0 or e.get(weight, 0) >= 0 else e[capacity] * e[weight] for (u, v, e) in nx.selfloop_edges(G, data=True)))
    wmax = max(chain([-inf], (e['capacity'] for (u, v, e) in R.edges(data=True))))
    if wmax == -inf:
        return (flow_cost, _build_flow_dict(G, R, capacity, weight))
    R_nodes = R.nodes
    R_succ = R.succ
    delta = 2 ** int(log(wmax, 2))
    while delta >= 1:
        for u in R:
            p_u = R_nodes[u]['potential']
            for (v, es) in R_succ[u].items():
                for (k, e) in es.items():
                    flow = e['capacity'] - e['flow']
                    if e['weight'] - p_u + R_nodes[v]['potential'] < 0:
                        flow = e['capacity'] - e['flow']
                        if flow >= delta:
                            e['flow'] += flow
                            R_succ[v][u][k[0], not k[1]]['flow'] -= flow
                            R_nodes[u]['excess'] -= flow
                            R_nodes[v]['excess'] += flow
        S = set()
        T = set()
        S_add = S.add
        S_remove = S.remove
        T_add = T.add
        T_remove = T.remove
        for u in R:
            excess = R_nodes[u]['excess']
            if excess >= delta:
                S_add(u)
            elif excess <= -delta:
                T_add(u)
        while S and T:
            s = arbitrary_element(S)
            t = None
            d = {}
            pred = {s: None}
            h = heap()
            h_insert = h.insert
            h_get = h.get
            h_insert(s, 0)
            while h:
                (u, d_u) = h.pop()
                d[u] = d_u
                if u in T:
                    t = u
                    break
                p_u = R_nodes[u]['potential']
                for (v, es) in R_succ[u].items():
                    if v in d:
                        continue
                    wmin = inf
                    for (k, e) in es.items():
                        if e['capacity'] - e['flow'] >= delta:
                            w = e['weight']
                            if w < wmin:
                                wmin = w
                                kmin = k
                                emin = e
                    if wmin == inf:
                        continue
                    d_v = d_u + wmin - p_u + R_nodes[v]['potential']
                    if h_insert(v, d_v):
                        pred[v] = (u, kmin, emin)
            if t is not None:
                while u != s:
                    v = u
                    (u, k, e) = pred[v]
                    e['flow'] += delta
                    R_succ[v][u][k[0], not k[1]]['flow'] -= delta
                R_nodes[s]['excess'] -= delta
                R_nodes[t]['excess'] += delta
                if R_nodes[s]['excess'] < delta:
                    S_remove(s)
                if R_nodes[t]['excess'] > -delta:
                    T_remove(t)
                d_t = d[t]
                for (u, d_u) in d.items():
                    R_nodes[u]['potential'] -= d_u - d_t
            else:
                S_remove(s)
        delta //= 2
    if any((R.nodes[u]['excess'] != 0 for u in R)):
        raise nx.NetworkXUnfeasible('No flow satisfying all demands.')
    for u in R:
        for (v, es) in R_succ[u].items():
            for e in es.values():
                flow = e['flow']
                if flow > 0:
                    flow_cost += flow * e['weight']
    return (flow_cost, _build_flow_dict(G, R, capacity, weight))