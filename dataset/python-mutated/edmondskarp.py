"""
Edmonds-Karp algorithm for maximum flow problems.
"""
import networkx as nx
from networkx.algorithms.flow.utils import build_residual_network
__all__ = ['edmonds_karp']

@nx._dispatch(graphs='R', preserve_edge_attrs={'R': {'capacity': float('inf'), 'flow': 0}}, preserve_graph_attrs=True)
def edmonds_karp_core(R, s, t, cutoff):
    if False:
        while True:
            i = 10
    'Implementation of the Edmonds-Karp algorithm.'
    R_nodes = R.nodes
    R_pred = R.pred
    R_succ = R.succ
    inf = R.graph['inf']

    def augment(path):
        if False:
            for i in range(10):
                print('nop')
        'Augment flow along a path from s to t.'
        flow = inf
        it = iter(path)
        u = next(it)
        for v in it:
            attr = R_succ[u][v]
            flow = min(flow, attr['capacity'] - attr['flow'])
            u = v
        if flow * 2 > inf:
            raise nx.NetworkXUnbounded('Infinite capacity path, flow unbounded above.')
        it = iter(path)
        u = next(it)
        for v in it:
            R_succ[u][v]['flow'] += flow
            R_succ[v][u]['flow'] -= flow
            u = v
        return flow

    def bidirectional_bfs():
        if False:
            for i in range(10):
                print('nop')
        'Bidirectional breadth-first search for an augmenting path.'
        pred = {s: None}
        q_s = [s]
        succ = {t: None}
        q_t = [t]
        while True:
            q = []
            if len(q_s) <= len(q_t):
                for u in q_s:
                    for (v, attr) in R_succ[u].items():
                        if v not in pred and attr['flow'] < attr['capacity']:
                            pred[v] = u
                            if v in succ:
                                return (v, pred, succ)
                            q.append(v)
                if not q:
                    return (None, None, None)
                q_s = q
            else:
                for u in q_t:
                    for (v, attr) in R_pred[u].items():
                        if v not in succ and attr['flow'] < attr['capacity']:
                            succ[v] = u
                            if v in pred:
                                return (v, pred, succ)
                            q.append(v)
                if not q:
                    return (None, None, None)
                q_t = q
    flow_value = 0
    while flow_value < cutoff:
        (v, pred, succ) = bidirectional_bfs()
        if pred is None:
            break
        path = [v]
        u = v
        while u != s:
            u = pred[u]
            path.append(u)
        path.reverse()
        u = v
        while u != t:
            u = succ[u]
            path.append(u)
        flow_value += augment(path)
    return flow_value

def edmonds_karp_impl(G, s, t, capacity, residual, cutoff):
    if False:
        while True:
            i = 10
    'Implementation of the Edmonds-Karp algorithm.'
    if s not in G:
        raise nx.NetworkXError(f'node {str(s)} not in graph')
    if t not in G:
        raise nx.NetworkXError(f'node {str(t)} not in graph')
    if s == t:
        raise nx.NetworkXError('source and sink are the same node')
    if residual is None:
        R = build_residual_network(G, capacity)
    else:
        R = residual
    for u in R:
        for e in R[u].values():
            e['flow'] = 0
    if cutoff is None:
        cutoff = float('inf')
    R.graph['flow_value'] = edmonds_karp_core(R, s, t, cutoff)
    return R

@nx._dispatch(graphs={'G': 0, 'residual?': 4}, edge_attrs={'capacity': float('inf')}, preserve_edge_attrs={'residual': {'capacity': float('inf')}}, preserve_graph_attrs={'residual'})
def edmonds_karp(G, s, t, capacity='capacity', residual=None, value_only=False, cutoff=None):
    if False:
        print('Hello World!')
    'Find a maximum single-commodity flow using the Edmonds-Karp algorithm.\n\n    This function returns the residual network resulting after computing\n    the maximum flow. See below for details about the conventions\n    NetworkX uses for defining residual networks.\n\n    This algorithm has a running time of $O(n m^2)$ for $n$ nodes and $m$\n    edges.\n\n\n    Parameters\n    ----------\n    G : NetworkX graph\n        Edges of the graph are expected to have an attribute called\n        \'capacity\'. If this attribute is not present, the edge is\n        considered to have infinite capacity.\n\n    s : node\n        Source node for the flow.\n\n    t : node\n        Sink node for the flow.\n\n    capacity : string\n        Edges of the graph G are expected to have an attribute capacity\n        that indicates how much flow the edge can support. If this\n        attribute is not present, the edge is considered to have\n        infinite capacity. Default value: \'capacity\'.\n\n    residual : NetworkX graph\n        Residual network on which the algorithm is to be executed. If None, a\n        new residual network is created. Default value: None.\n\n    value_only : bool\n        If True compute only the value of the maximum flow. This parameter\n        will be ignored by this algorithm because it is not applicable.\n\n    cutoff : integer, float\n        If specified, the algorithm will terminate when the flow value reaches\n        or exceeds the cutoff. In this case, it may be unable to immediately\n        determine a minimum cut. Default value: None.\n\n    Returns\n    -------\n    R : NetworkX DiGraph\n        Residual network after computing the maximum flow.\n\n    Raises\n    ------\n    NetworkXError\n        The algorithm does not support MultiGraph and MultiDiGraph. If\n        the input graph is an instance of one of these two classes, a\n        NetworkXError is raised.\n\n    NetworkXUnbounded\n        If the graph has a path of infinite capacity, the value of a\n        feasible flow on the graph is unbounded above and the function\n        raises a NetworkXUnbounded.\n\n    See also\n    --------\n    :meth:`maximum_flow`\n    :meth:`minimum_cut`\n    :meth:`preflow_push`\n    :meth:`shortest_augmenting_path`\n\n    Notes\n    -----\n    The residual network :samp:`R` from an input graph :samp:`G` has the\n    same nodes as :samp:`G`. :samp:`R` is a DiGraph that contains a pair\n    of edges :samp:`(u, v)` and :samp:`(v, u)` iff :samp:`(u, v)` is not a\n    self-loop, and at least one of :samp:`(u, v)` and :samp:`(v, u)` exists\n    in :samp:`G`.\n\n    For each edge :samp:`(u, v)` in :samp:`R`, :samp:`R[u][v][\'capacity\']`\n    is equal to the capacity of :samp:`(u, v)` in :samp:`G` if it exists\n    in :samp:`G` or zero otherwise. If the capacity is infinite,\n    :samp:`R[u][v][\'capacity\']` will have a high arbitrary finite value\n    that does not affect the solution of the problem. This value is stored in\n    :samp:`R.graph[\'inf\']`. For each edge :samp:`(u, v)` in :samp:`R`,\n    :samp:`R[u][v][\'flow\']` represents the flow function of :samp:`(u, v)` and\n    satisfies :samp:`R[u][v][\'flow\'] == -R[v][u][\'flow\']`.\n\n    The flow value, defined as the total flow into :samp:`t`, the sink, is\n    stored in :samp:`R.graph[\'flow_value\']`. If :samp:`cutoff` is not\n    specified, reachability to :samp:`t` using only edges :samp:`(u, v)` such\n    that :samp:`R[u][v][\'flow\'] < R[u][v][\'capacity\']` induces a minimum\n    :samp:`s`-:samp:`t` cut.\n\n    Examples\n    --------\n    >>> from networkx.algorithms.flow import edmonds_karp\n\n    The functions that implement flow algorithms and output a residual\n    network, such as this one, are not imported to the base NetworkX\n    namespace, so you have to explicitly import them from the flow package.\n\n    >>> G = nx.DiGraph()\n    >>> G.add_edge("x", "a", capacity=3.0)\n    >>> G.add_edge("x", "b", capacity=1.0)\n    >>> G.add_edge("a", "c", capacity=3.0)\n    >>> G.add_edge("b", "c", capacity=5.0)\n    >>> G.add_edge("b", "d", capacity=4.0)\n    >>> G.add_edge("d", "e", capacity=2.0)\n    >>> G.add_edge("c", "y", capacity=2.0)\n    >>> G.add_edge("e", "y", capacity=3.0)\n    >>> R = edmonds_karp(G, "x", "y")\n    >>> flow_value = nx.maximum_flow_value(G, "x", "y")\n    >>> flow_value\n    3.0\n    >>> flow_value == R.graph["flow_value"]\n    True\n\n    '
    R = edmonds_karp_impl(G, s, t, capacity, residual, cutoff)
    R.graph['algorithm'] = 'edmonds_karp'
    return R