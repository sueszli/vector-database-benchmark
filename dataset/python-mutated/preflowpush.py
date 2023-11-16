"""
Highest-label preflow-push algorithm for maximum flow problems.
"""
from collections import deque
from itertools import islice
import networkx as nx
from ...utils import arbitrary_element
from .utils import CurrentEdge, GlobalRelabelThreshold, Level, build_residual_network, detect_unboundedness
__all__ = ['preflow_push']

def preflow_push_impl(G, s, t, capacity, residual, global_relabel_freq, value_only):
    if False:
        while True:
            i = 10
    'Implementation of the highest-label preflow-push algorithm.'
    if s not in G:
        raise nx.NetworkXError(f'node {str(s)} not in graph')
    if t not in G:
        raise nx.NetworkXError(f'node {str(t)} not in graph')
    if s == t:
        raise nx.NetworkXError('source and sink are the same node')
    if global_relabel_freq is None:
        global_relabel_freq = 0
    if global_relabel_freq < 0:
        raise nx.NetworkXError('global_relabel_freq must be nonnegative.')
    if residual is None:
        R = build_residual_network(G, capacity)
    else:
        R = residual
    detect_unboundedness(R, s, t)
    R_nodes = R.nodes
    R_pred = R.pred
    R_succ = R.succ
    for u in R:
        R_nodes[u]['excess'] = 0
        for e in R_succ[u].values():
            e['flow'] = 0

    def reverse_bfs(src):
        if False:
            i = 10
            return i + 15
        'Perform a reverse breadth-first search from src in the residual\n        network.\n        '
        heights = {src: 0}
        q = deque([(src, 0)])
        while q:
            (u, height) = q.popleft()
            height += 1
            for (v, attr) in R_pred[u].items():
                if v not in heights and attr['flow'] < attr['capacity']:
                    heights[v] = height
                    q.append((v, height))
        return heights
    heights = reverse_bfs(t)
    if s not in heights:
        R.graph['flow_value'] = 0
        return R
    n = len(R)
    max_height = max((heights[u] for u in heights if u != s))
    heights[s] = n
    grt = GlobalRelabelThreshold(n, R.size(), global_relabel_freq)
    for u in R:
        R_nodes[u]['height'] = heights[u] if u in heights else n + 1
        R_nodes[u]['curr_edge'] = CurrentEdge(R_succ[u])

    def push(u, v, flow):
        if False:
            for i in range(10):
                print('nop')
        'Push flow units of flow from u to v.'
        R_succ[u][v]['flow'] += flow
        R_succ[v][u]['flow'] -= flow
        R_nodes[u]['excess'] -= flow
        R_nodes[v]['excess'] += flow
    for (u, attr) in R_succ[s].items():
        flow = attr['capacity']
        if flow > 0:
            push(s, u, flow)
    levels = [Level() for i in range(2 * n)]
    for u in R:
        if u != s and u != t:
            level = levels[R_nodes[u]['height']]
            if R_nodes[u]['excess'] > 0:
                level.active.add(u)
            else:
                level.inactive.add(u)

    def activate(v):
        if False:
            return 10
        'Move a node from the inactive set to the active set of its level.'
        if v != s and v != t:
            level = levels[R_nodes[v]['height']]
            if v in level.inactive:
                level.inactive.remove(v)
                level.active.add(v)

    def relabel(u):
        if False:
            print('Hello World!')
        'Relabel a node to create an admissible edge.'
        grt.add_work(len(R_succ[u]))
        return min((R_nodes[v]['height'] for (v, attr) in R_succ[u].items() if attr['flow'] < attr['capacity'])) + 1

    def discharge(u, is_phase1):
        if False:
            while True:
                i = 10
        'Discharge a node until it becomes inactive or, during phase 1 (see\n        below), its height reaches at least n. The node is known to have the\n        largest height among active nodes.\n        '
        height = R_nodes[u]['height']
        curr_edge = R_nodes[u]['curr_edge']
        next_height = height
        levels[height].active.remove(u)
        while True:
            (v, attr) = curr_edge.get()
            if height == R_nodes[v]['height'] + 1 and attr['flow'] < attr['capacity']:
                flow = min(R_nodes[u]['excess'], attr['capacity'] - attr['flow'])
                push(u, v, flow)
                activate(v)
                if R_nodes[u]['excess'] == 0:
                    levels[height].inactive.add(u)
                    break
            try:
                curr_edge.move_to_next()
            except StopIteration:
                height = relabel(u)
                if is_phase1 and height >= n - 1:
                    levels[height].active.add(u)
                    break
                next_height = height
        R_nodes[u]['height'] = height
        return next_height

    def gap_heuristic(height):
        if False:
            while True:
                i = 10
        'Apply the gap heuristic.'
        for level in islice(levels, height + 1, max_height + 1):
            for u in level.active:
                R_nodes[u]['height'] = n + 1
            for u in level.inactive:
                R_nodes[u]['height'] = n + 1
            levels[n + 1].active.update(level.active)
            level.active.clear()
            levels[n + 1].inactive.update(level.inactive)
            level.inactive.clear()

    def global_relabel(from_sink):
        if False:
            while True:
                i = 10
        'Apply the global relabeling heuristic.'
        src = t if from_sink else s
        heights = reverse_bfs(src)
        if not from_sink:
            del heights[t]
        max_height = max(heights.values())
        if from_sink:
            for u in R:
                if u not in heights and R_nodes[u]['height'] < n:
                    heights[u] = n + 1
        else:
            for u in heights:
                heights[u] += n
            max_height += n
        del heights[src]
        for (u, new_height) in heights.items():
            old_height = R_nodes[u]['height']
            if new_height != old_height:
                if u in levels[old_height].active:
                    levels[old_height].active.remove(u)
                    levels[new_height].active.add(u)
                else:
                    levels[old_height].inactive.remove(u)
                    levels[new_height].inactive.add(u)
                R_nodes[u]['height'] = new_height
        return max_height
    height = max_height
    while height > 0:
        while True:
            level = levels[height]
            if not level.active:
                height -= 1
                break
            old_height = height
            old_level = level
            u = arbitrary_element(level.active)
            height = discharge(u, True)
            if grt.is_reached():
                height = global_relabel(True)
                max_height = height
                grt.clear_work()
            elif not old_level.active and (not old_level.inactive):
                gap_heuristic(old_height)
                height = old_height - 1
                max_height = height
            else:
                max_height = max(max_height, height)
    if value_only:
        R.graph['flow_value'] = R_nodes[t]['excess']
        return R
    height = global_relabel(False)
    grt.clear_work()
    while height > n:
        while True:
            level = levels[height]
            if not level.active:
                height -= 1
                break
            u = arbitrary_element(level.active)
            height = discharge(u, False)
            if grt.is_reached():
                height = global_relabel(False)
                grt.clear_work()
    R.graph['flow_value'] = R_nodes[t]['excess']
    return R

@nx._dispatch(graphs={'G': 0, 'residual?': 4}, edge_attrs={'capacity': float('inf')}, preserve_edge_attrs={'residual': {'capacity': float('inf')}}, preserve_graph_attrs={'residual'})
def preflow_push(G, s, t, capacity='capacity', residual=None, global_relabel_freq=1, value_only=False):
    if False:
        i = 10
        return i + 15
    'Find a maximum single-commodity flow using the highest-label\n    preflow-push algorithm.\n\n    This function returns the residual network resulting after computing\n    the maximum flow. See below for details about the conventions\n    NetworkX uses for defining residual networks.\n\n    This algorithm has a running time of $O(n^2 \\sqrt{m})$ for $n$ nodes and\n    $m$ edges.\n\n\n    Parameters\n    ----------\n    G : NetworkX graph\n        Edges of the graph are expected to have an attribute called\n        \'capacity\'. If this attribute is not present, the edge is\n        considered to have infinite capacity.\n\n    s : node\n        Source node for the flow.\n\n    t : node\n        Sink node for the flow.\n\n    capacity : string\n        Edges of the graph G are expected to have an attribute capacity\n        that indicates how much flow the edge can support. If this\n        attribute is not present, the edge is considered to have\n        infinite capacity. Default value: \'capacity\'.\n\n    residual : NetworkX graph\n        Residual network on which the algorithm is to be executed. If None, a\n        new residual network is created. Default value: None.\n\n    global_relabel_freq : integer, float\n        Relative frequency of applying the global relabeling heuristic to speed\n        up the algorithm. If it is None, the heuristic is disabled. Default\n        value: 1.\n\n    value_only : bool\n        If False, compute a maximum flow; otherwise, compute a maximum preflow\n        which is enough for computing the maximum flow value. Default value:\n        False.\n\n    Returns\n    -------\n    R : NetworkX DiGraph\n        Residual network after computing the maximum flow.\n\n    Raises\n    ------\n    NetworkXError\n        The algorithm does not support MultiGraph and MultiDiGraph. If\n        the input graph is an instance of one of these two classes, a\n        NetworkXError is raised.\n\n    NetworkXUnbounded\n        If the graph has a path of infinite capacity, the value of a\n        feasible flow on the graph is unbounded above and the function\n        raises a NetworkXUnbounded.\n\n    See also\n    --------\n    :meth:`maximum_flow`\n    :meth:`minimum_cut`\n    :meth:`edmonds_karp`\n    :meth:`shortest_augmenting_path`\n\n    Notes\n    -----\n    The residual network :samp:`R` from an input graph :samp:`G` has the\n    same nodes as :samp:`G`. :samp:`R` is a DiGraph that contains a pair\n    of edges :samp:`(u, v)` and :samp:`(v, u)` iff :samp:`(u, v)` is not a\n    self-loop, and at least one of :samp:`(u, v)` and :samp:`(v, u)` exists\n    in :samp:`G`. For each node :samp:`u` in :samp:`R`,\n    :samp:`R.nodes[u][\'excess\']` represents the difference between flow into\n    :samp:`u` and flow out of :samp:`u`.\n\n    For each edge :samp:`(u, v)` in :samp:`R`, :samp:`R[u][v][\'capacity\']`\n    is equal to the capacity of :samp:`(u, v)` in :samp:`G` if it exists\n    in :samp:`G` or zero otherwise. If the capacity is infinite,\n    :samp:`R[u][v][\'capacity\']` will have a high arbitrary finite value\n    that does not affect the solution of the problem. This value is stored in\n    :samp:`R.graph[\'inf\']`. For each edge :samp:`(u, v)` in :samp:`R`,\n    :samp:`R[u][v][\'flow\']` represents the flow function of :samp:`(u, v)` and\n    satisfies :samp:`R[u][v][\'flow\'] == -R[v][u][\'flow\']`.\n\n    The flow value, defined as the total flow into :samp:`t`, the sink, is\n    stored in :samp:`R.graph[\'flow_value\']`. Reachability to :samp:`t` using\n    only edges :samp:`(u, v)` such that\n    :samp:`R[u][v][\'flow\'] < R[u][v][\'capacity\']` induces a minimum\n    :samp:`s`-:samp:`t` cut.\n\n    Examples\n    --------\n    >>> from networkx.algorithms.flow import preflow_push\n\n    The functions that implement flow algorithms and output a residual\n    network, such as this one, are not imported to the base NetworkX\n    namespace, so you have to explicitly import them from the flow package.\n\n    >>> G = nx.DiGraph()\n    >>> G.add_edge("x", "a", capacity=3.0)\n    >>> G.add_edge("x", "b", capacity=1.0)\n    >>> G.add_edge("a", "c", capacity=3.0)\n    >>> G.add_edge("b", "c", capacity=5.0)\n    >>> G.add_edge("b", "d", capacity=4.0)\n    >>> G.add_edge("d", "e", capacity=2.0)\n    >>> G.add_edge("c", "y", capacity=2.0)\n    >>> G.add_edge("e", "y", capacity=3.0)\n    >>> R = preflow_push(G, "x", "y")\n    >>> flow_value = nx.maximum_flow_value(G, "x", "y")\n    >>> flow_value == R.graph["flow_value"]\n    True\n    >>> # preflow_push also stores the maximum flow value\n    >>> # in the excess attribute of the sink node t\n    >>> flow_value == R.nodes["y"]["excess"]\n    True\n    >>> # For some problems, you might only want to compute a\n    >>> # maximum preflow.\n    >>> R = preflow_push(G, "x", "y", value_only=True)\n    >>> flow_value == R.graph["flow_value"]\n    True\n    >>> flow_value == R.nodes["y"]["excess"]\n    True\n\n    '
    R = preflow_push_impl(G, s, t, capacity, residual, global_relabel_freq, value_only)
    R.graph['algorithm'] = 'preflow_push'
    return R