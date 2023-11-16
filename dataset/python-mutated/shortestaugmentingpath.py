"""
Shortest augmenting path algorithm for maximum flow problems.
"""
from collections import deque
import networkx as nx
from .edmondskarp import edmonds_karp_core
from .utils import CurrentEdge, build_residual_network
__all__ = ['shortest_augmenting_path']

def shortest_augmenting_path_impl(G, s, t, capacity, residual, two_phase, cutoff):
    if False:
        print('Hello World!')
    'Implementation of the shortest augmenting path algorithm.'
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
    R_nodes = R.nodes
    R_pred = R.pred
    R_succ = R.succ
    for u in R:
        for e in R_succ[u].values():
            e['flow'] = 0
    heights = {t: 0}
    q = deque([(t, 0)])
    while q:
        (u, height) = q.popleft()
        height += 1
        for (v, attr) in R_pred[u].items():
            if v not in heights and attr['flow'] < attr['capacity']:
                heights[v] = height
                q.append((v, height))
    if s not in heights:
        R.graph['flow_value'] = 0
        return R
    n = len(G)
    m = R.size() / 2
    for u in R:
        R_nodes[u]['height'] = heights[u] if u in heights else n
        R_nodes[u]['curr_edge'] = CurrentEdge(R_succ[u])
    counts = [0] * (2 * n - 1)
    for u in R:
        counts[R_nodes[u]['height']] += 1
    inf = R.graph['inf']

    def augment(path):
        if False:
            print('Hello World!')
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

    def relabel(u):
        if False:
            print('Hello World!')
        'Relabel a node to create an admissible edge.'
        height = n - 1
        for (v, attr) in R_succ[u].items():
            if attr['flow'] < attr['capacity']:
                height = min(height, R_nodes[v]['height'])
        return height + 1
    if cutoff is None:
        cutoff = float('inf')
    flow_value = 0
    path = [s]
    u = s
    d = n if not two_phase else int(min(m ** 0.5, 2 * n ** (2.0 / 3)))
    done = R_nodes[s]['height'] >= d
    while not done:
        height = R_nodes[u]['height']
        curr_edge = R_nodes[u]['curr_edge']
        while True:
            (v, attr) = curr_edge.get()
            if height == R_nodes[v]['height'] + 1 and attr['flow'] < attr['capacity']:
                path.append(v)
                u = v
                break
            try:
                curr_edge.move_to_next()
            except StopIteration:
                counts[height] -= 1
                if counts[height] == 0:
                    R.graph['flow_value'] = flow_value
                    return R
                height = relabel(u)
                if u == s and height >= d:
                    if not two_phase:
                        R.graph['flow_value'] = flow_value
                        return R
                    else:
                        done = True
                        break
                counts[height] += 1
                R_nodes[u]['height'] = height
                if u != s:
                    path.pop()
                    u = path[-1]
                    break
        if u == t:
            flow_value += augment(path)
            if flow_value >= cutoff:
                R.graph['flow_value'] = flow_value
                return R
            path = [s]
            u = s
    flow_value += edmonds_karp_core(R, s, t, cutoff - flow_value)
    R.graph['flow_value'] = flow_value
    return R

@nx._dispatch(graphs={'G': 0, 'residual?': 4}, edge_attrs={'capacity': float('inf')}, preserve_edge_attrs={'residual': {'capacity': float('inf')}}, preserve_graph_attrs={'residual'})
def shortest_augmenting_path(G, s, t, capacity='capacity', residual=None, value_only=False, two_phase=False, cutoff=None):
    if False:
        for i in range(10):
            print('nop')
    'Find a maximum single-commodity flow using the shortest augmenting path\n    algorithm.\n\n    This function returns the residual network resulting after computing\n    the maximum flow. See below for details about the conventions\n    NetworkX uses for defining residual networks.\n\n    This algorithm has a running time of $O(n^2 m)$ for $n$ nodes and $m$\n    edges.\n\n\n    Parameters\n    ----------\n    G : NetworkX graph\n        Edges of the graph are expected to have an attribute called\n        \'capacity\'. If this attribute is not present, the edge is\n        considered to have infinite capacity.\n\n    s : node\n        Source node for the flow.\n\n    t : node\n        Sink node for the flow.\n\n    capacity : string\n        Edges of the graph G are expected to have an attribute capacity\n        that indicates how much flow the edge can support. If this\n        attribute is not present, the edge is considered to have\n        infinite capacity. Default value: \'capacity\'.\n\n    residual : NetworkX graph\n        Residual network on which the algorithm is to be executed. If None, a\n        new residual network is created. Default value: None.\n\n    value_only : bool\n        If True compute only the value of the maximum flow. This parameter\n        will be ignored by this algorithm because it is not applicable.\n\n    two_phase : bool\n        If True, a two-phase variant is used. The two-phase variant improves\n        the running time on unit-capacity networks from $O(nm)$ to\n        $O(\\min(n^{2/3}, m^{1/2}) m)$. Default value: False.\n\n    cutoff : integer, float\n        If specified, the algorithm will terminate when the flow value reaches\n        or exceeds the cutoff. In this case, it may be unable to immediately\n        determine a minimum cut. Default value: None.\n\n    Returns\n    -------\n    R : NetworkX DiGraph\n        Residual network after computing the maximum flow.\n\n    Raises\n    ------\n    NetworkXError\n        The algorithm does not support MultiGraph and MultiDiGraph. If\n        the input graph is an instance of one of these two classes, a\n        NetworkXError is raised.\n\n    NetworkXUnbounded\n        If the graph has a path of infinite capacity, the value of a\n        feasible flow on the graph is unbounded above and the function\n        raises a NetworkXUnbounded.\n\n    See also\n    --------\n    :meth:`maximum_flow`\n    :meth:`minimum_cut`\n    :meth:`edmonds_karp`\n    :meth:`preflow_push`\n\n    Notes\n    -----\n    The residual network :samp:`R` from an input graph :samp:`G` has the\n    same nodes as :samp:`G`. :samp:`R` is a DiGraph that contains a pair\n    of edges :samp:`(u, v)` and :samp:`(v, u)` iff :samp:`(u, v)` is not a\n    self-loop, and at least one of :samp:`(u, v)` and :samp:`(v, u)` exists\n    in :samp:`G`.\n\n    For each edge :samp:`(u, v)` in :samp:`R`, :samp:`R[u][v][\'capacity\']`\n    is equal to the capacity of :samp:`(u, v)` in :samp:`G` if it exists\n    in :samp:`G` or zero otherwise. If the capacity is infinite,\n    :samp:`R[u][v][\'capacity\']` will have a high arbitrary finite value\n    that does not affect the solution of the problem. This value is stored in\n    :samp:`R.graph[\'inf\']`. For each edge :samp:`(u, v)` in :samp:`R`,\n    :samp:`R[u][v][\'flow\']` represents the flow function of :samp:`(u, v)` and\n    satisfies :samp:`R[u][v][\'flow\'] == -R[v][u][\'flow\']`.\n\n    The flow value, defined as the total flow into :samp:`t`, the sink, is\n    stored in :samp:`R.graph[\'flow_value\']`. If :samp:`cutoff` is not\n    specified, reachability to :samp:`t` using only edges :samp:`(u, v)` such\n    that :samp:`R[u][v][\'flow\'] < R[u][v][\'capacity\']` induces a minimum\n    :samp:`s`-:samp:`t` cut.\n\n    Examples\n    --------\n    >>> from networkx.algorithms.flow import shortest_augmenting_path\n\n    The functions that implement flow algorithms and output a residual\n    network, such as this one, are not imported to the base NetworkX\n    namespace, so you have to explicitly import them from the flow package.\n\n    >>> G = nx.DiGraph()\n    >>> G.add_edge("x", "a", capacity=3.0)\n    >>> G.add_edge("x", "b", capacity=1.0)\n    >>> G.add_edge("a", "c", capacity=3.0)\n    >>> G.add_edge("b", "c", capacity=5.0)\n    >>> G.add_edge("b", "d", capacity=4.0)\n    >>> G.add_edge("d", "e", capacity=2.0)\n    >>> G.add_edge("c", "y", capacity=2.0)\n    >>> G.add_edge("e", "y", capacity=3.0)\n    >>> R = shortest_augmenting_path(G, "x", "y")\n    >>> flow_value = nx.maximum_flow_value(G, "x", "y")\n    >>> flow_value\n    3.0\n    >>> flow_value == R.graph["flow_value"]\n    True\n\n    '
    R = shortest_augmenting_path_impl(G, s, t, capacity, residual, two_phase, cutoff)
    R.graph['algorithm'] = 'shortest_augmenting_path'
    return R