"""
Boykov-Kolmogorov algorithm for maximum flow problems.
"""
from collections import deque
from operator import itemgetter
import networkx as nx
from networkx.algorithms.flow.utils import build_residual_network
__all__ = ['boykov_kolmogorov']

@nx._dispatch(graphs={'G': 0, 'residual?': 4}, edge_attrs={'capacity': float('inf')}, preserve_edge_attrs={'residual': {'capacity': float('inf')}}, preserve_graph_attrs={'residual'})
def boykov_kolmogorov(G, s, t, capacity='capacity', residual=None, value_only=False, cutoff=None):
    if False:
        while True:
            i = 10
    'Find a maximum single-commodity flow using Boykov-Kolmogorov algorithm.\n\n    This function returns the residual network resulting after computing\n    the maximum flow. See below for details about the conventions\n    NetworkX uses for defining residual networks.\n\n    This algorithm has worse case complexity $O(n^2 m |C|)$ for $n$ nodes, $m$\n    edges, and $|C|$ the cost of the minimum cut [1]_. This implementation\n    uses the marking heuristic defined in [2]_ which improves its running\n    time in many practical problems.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n        Edges of the graph are expected to have an attribute called\n        \'capacity\'. If this attribute is not present, the edge is\n        considered to have infinite capacity.\n\n    s : node\n        Source node for the flow.\n\n    t : node\n        Sink node for the flow.\n\n    capacity : string\n        Edges of the graph G are expected to have an attribute capacity\n        that indicates how much flow the edge can support. If this\n        attribute is not present, the edge is considered to have\n        infinite capacity. Default value: \'capacity\'.\n\n    residual : NetworkX graph\n        Residual network on which the algorithm is to be executed. If None, a\n        new residual network is created. Default value: None.\n\n    value_only : bool\n        If True compute only the value of the maximum flow. This parameter\n        will be ignored by this algorithm because it is not applicable.\n\n    cutoff : integer, float\n        If specified, the algorithm will terminate when the flow value reaches\n        or exceeds the cutoff. In this case, it may be unable to immediately\n        determine a minimum cut. Default value: None.\n\n    Returns\n    -------\n    R : NetworkX DiGraph\n        Residual network after computing the maximum flow.\n\n    Raises\n    ------\n    NetworkXError\n        The algorithm does not support MultiGraph and MultiDiGraph. If\n        the input graph is an instance of one of these two classes, a\n        NetworkXError is raised.\n\n    NetworkXUnbounded\n        If the graph has a path of infinite capacity, the value of a\n        feasible flow on the graph is unbounded above and the function\n        raises a NetworkXUnbounded.\n\n    See also\n    --------\n    :meth:`maximum_flow`\n    :meth:`minimum_cut`\n    :meth:`preflow_push`\n    :meth:`shortest_augmenting_path`\n\n    Notes\n    -----\n    The residual network :samp:`R` from an input graph :samp:`G` has the\n    same nodes as :samp:`G`. :samp:`R` is a DiGraph that contains a pair\n    of edges :samp:`(u, v)` and :samp:`(v, u)` iff :samp:`(u, v)` is not a\n    self-loop, and at least one of :samp:`(u, v)` and :samp:`(v, u)` exists\n    in :samp:`G`.\n\n    For each edge :samp:`(u, v)` in :samp:`R`, :samp:`R[u][v][\'capacity\']`\n    is equal to the capacity of :samp:`(u, v)` in :samp:`G` if it exists\n    in :samp:`G` or zero otherwise. If the capacity is infinite,\n    :samp:`R[u][v][\'capacity\']` will have a high arbitrary finite value\n    that does not affect the solution of the problem. This value is stored in\n    :samp:`R.graph[\'inf\']`. For each edge :samp:`(u, v)` in :samp:`R`,\n    :samp:`R[u][v][\'flow\']` represents the flow function of :samp:`(u, v)` and\n    satisfies :samp:`R[u][v][\'flow\'] == -R[v][u][\'flow\']`.\n\n    The flow value, defined as the total flow into :samp:`t`, the sink, is\n    stored in :samp:`R.graph[\'flow_value\']`. If :samp:`cutoff` is not\n    specified, reachability to :samp:`t` using only edges :samp:`(u, v)` such\n    that :samp:`R[u][v][\'flow\'] < R[u][v][\'capacity\']` induces a minimum\n    :samp:`s`-:samp:`t` cut.\n\n    Examples\n    --------\n    >>> from networkx.algorithms.flow import boykov_kolmogorov\n\n    The functions that implement flow algorithms and output a residual\n    network, such as this one, are not imported to the base NetworkX\n    namespace, so you have to explicitly import them from the flow package.\n\n    >>> G = nx.DiGraph()\n    >>> G.add_edge("x", "a", capacity=3.0)\n    >>> G.add_edge("x", "b", capacity=1.0)\n    >>> G.add_edge("a", "c", capacity=3.0)\n    >>> G.add_edge("b", "c", capacity=5.0)\n    >>> G.add_edge("b", "d", capacity=4.0)\n    >>> G.add_edge("d", "e", capacity=2.0)\n    >>> G.add_edge("c", "y", capacity=2.0)\n    >>> G.add_edge("e", "y", capacity=3.0)\n    >>> R = boykov_kolmogorov(G, "x", "y")\n    >>> flow_value = nx.maximum_flow_value(G, "x", "y")\n    >>> flow_value\n    3.0\n    >>> flow_value == R.graph["flow_value"]\n    True\n\n    A nice feature of the Boykov-Kolmogorov algorithm is that a partition\n    of the nodes that defines a minimum cut can be easily computed based\n    on the search trees used during the algorithm. These trees are stored\n    in the graph attribute `trees` of the residual network.\n\n    >>> source_tree, target_tree = R.graph["trees"]\n    >>> partition = (set(source_tree), set(G) - set(source_tree))\n\n    Or equivalently:\n\n    >>> partition = (set(G) - set(target_tree), set(target_tree))\n\n    References\n    ----------\n    .. [1] Boykov, Y., & Kolmogorov, V. (2004). An experimental comparison\n           of min-cut/max-flow algorithms for energy minimization in vision.\n           Pattern Analysis and Machine Intelligence, IEEE Transactions on,\n           26(9), 1124-1137.\n           https://doi.org/10.1109/TPAMI.2004.60\n\n    .. [2] Vladimir Kolmogorov. Graph-based Algorithms for Multi-camera\n           Reconstruction Problem. PhD thesis, Cornell University, CS Department,\n           2003. pp. 109-114.\n           https://web.archive.org/web/20170809091249/https://pub.ist.ac.at/~vnk/papers/thesis.pdf\n\n    '
    R = boykov_kolmogorov_impl(G, s, t, capacity, residual, cutoff)
    R.graph['algorithm'] = 'boykov_kolmogorov'
    return R

def boykov_kolmogorov_impl(G, s, t, capacity, residual, cutoff):
    if False:
        i = 10
        return i + 15
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
    INF = R.graph['inf']
    if cutoff is None:
        cutoff = INF
    R_succ = R.succ
    R_pred = R.pred

    def grow():
        if False:
            for i in range(10):
                print('nop')
        'Bidirectional breadth-first search for the growth stage.\n\n        Returns a connecting edge, that is and edge that connects\n        a node from the source search tree with a node from the\n        target search tree.\n        The first node in the connecting edge is always from the\n        source tree and the last node from the target tree.\n        '
        while active:
            u = active[0]
            if u in source_tree:
                this_tree = source_tree
                other_tree = target_tree
                neighbors = R_succ
            else:
                this_tree = target_tree
                other_tree = source_tree
                neighbors = R_pred
            for (v, attr) in neighbors[u].items():
                if attr['capacity'] - attr['flow'] > 0:
                    if v not in this_tree:
                        if v in other_tree:
                            return (u, v) if this_tree is source_tree else (v, u)
                        this_tree[v] = u
                        dist[v] = dist[u] + 1
                        timestamp[v] = timestamp[u]
                        active.append(v)
                    elif v in this_tree and _is_closer(u, v):
                        this_tree[v] = u
                        dist[v] = dist[u] + 1
                        timestamp[v] = timestamp[u]
            _ = active.popleft()
        return (None, None)

    def augment(u, v):
        if False:
            for i in range(10):
                print('nop')
        'Augmentation stage.\n\n        Reconstruct path and determine its residual capacity.\n        We start from a connecting edge, which links a node\n        from the source tree to a node from the target tree.\n        The connecting edge is the output of the grow function\n        and the input of this function.\n        '
        attr = R_succ[u][v]
        flow = min(INF, attr['capacity'] - attr['flow'])
        path = [u]
        w = u
        while w != s:
            n = w
            w = source_tree[n]
            attr = R_pred[n][w]
            flow = min(flow, attr['capacity'] - attr['flow'])
            path.append(w)
        path.reverse()
        path.append(v)
        w = v
        while w != t:
            n = w
            w = target_tree[n]
            attr = R_succ[n][w]
            flow = min(flow, attr['capacity'] - attr['flow'])
            path.append(w)
        it = iter(path)
        u = next(it)
        these_orphans = []
        for v in it:
            R_succ[u][v]['flow'] += flow
            R_succ[v][u]['flow'] -= flow
            if R_succ[u][v]['flow'] == R_succ[u][v]['capacity']:
                if v in source_tree:
                    source_tree[v] = None
                    these_orphans.append(v)
                if u in target_tree:
                    target_tree[u] = None
                    these_orphans.append(u)
            u = v
        orphans.extend(sorted(these_orphans, key=dist.get))
        return flow

    def adopt():
        if False:
            for i in range(10):
                print('nop')
        'Adoption stage.\n\n        Reconstruct search trees by adopting or discarding orphans.\n        During augmentation stage some edges got saturated and thus\n        the source and target search trees broke down to forests, with\n        orphans as roots of some of its trees. We have to reconstruct\n        the search trees rooted to source and target before we can grow\n        them again.\n        '
        while orphans:
            u = orphans.popleft()
            if u in source_tree:
                tree = source_tree
                neighbors = R_pred
            else:
                tree = target_tree
                neighbors = R_succ
            nbrs = ((n, attr, dist[n]) for (n, attr) in neighbors[u].items() if n in tree)
            for (v, attr, d) in sorted(nbrs, key=itemgetter(2)):
                if attr['capacity'] - attr['flow'] > 0:
                    if _has_valid_root(v, tree):
                        tree[u] = v
                        dist[u] = dist[v] + 1
                        timestamp[u] = time
                        break
            else:
                nbrs = ((n, attr, dist[n]) for (n, attr) in neighbors[u].items() if n in tree)
                for (v, attr, d) in sorted(nbrs, key=itemgetter(2)):
                    if attr['capacity'] - attr['flow'] > 0:
                        if v not in active:
                            active.append(v)
                    if tree[v] == u:
                        tree[v] = None
                        orphans.appendleft(v)
                if u in active:
                    active.remove(u)
                del tree[u]

    def _has_valid_root(n, tree):
        if False:
            return 10
        path = []
        v = n
        while v is not None:
            path.append(v)
            if v in (s, t):
                base_dist = 0
                break
            elif timestamp[v] == time:
                base_dist = dist[v]
                break
            v = tree[v]
        else:
            return False
        length = len(path)
        for (i, u) in enumerate(path, 1):
            dist[u] = base_dist + length - i
            timestamp[u] = time
        return True

    def _is_closer(u, v):
        if False:
            while True:
                i = 10
        return timestamp[v] <= timestamp[u] and dist[v] > dist[u] + 1
    source_tree = {s: None}
    target_tree = {t: None}
    active = deque([s, t])
    orphans = deque()
    flow_value = 0
    time = 1
    timestamp = {s: time, t: time}
    dist = {s: 0, t: 0}
    while flow_value < cutoff:
        (u, v) = grow()
        if u is None:
            break
        time += 1
        flow_value += augment(u, v)
        adopt()
    if flow_value * 2 > INF:
        raise nx.NetworkXUnbounded('Infinite capacity path, flow unbounded above.')
    R.graph['trees'] = (source_tree, target_tree)
    R.graph['flow_value'] = flow_value
    return R