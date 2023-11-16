"""Flow based node and edge disjoint paths."""
import networkx as nx
from networkx.algorithms.flow import edmonds_karp, preflow_push, shortest_augmenting_path
from networkx.exception import NetworkXNoPath
default_flow_func = edmonds_karp
from itertools import filterfalse as _filterfalse
from .utils import build_auxiliary_edge_connectivity, build_auxiliary_node_connectivity
__all__ = ['edge_disjoint_paths', 'node_disjoint_paths']

@nx._dispatch(graphs={'G': 0, 'auxiliary?': 5, 'residual?': 6}, preserve_edge_attrs={'auxiliary': {'capacity': float('inf')}, 'residual': {'capacity': float('inf')}}, preserve_graph_attrs={'residual'})
def edge_disjoint_paths(G, s, t, flow_func=None, cutoff=None, auxiliary=None, residual=None):
    if False:
        while True:
            i = 10
    'Returns the edges disjoint paths between source and target.\n\n    Edge disjoint paths are paths that do not share any edge. The\n    number of edge disjoint paths between source and target is equal\n    to their edge connectivity.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n\n    s : node\n        Source node for the flow.\n\n    t : node\n        Sink node for the flow.\n\n    flow_func : function\n        A function for computing the maximum flow among a pair of nodes.\n        The function has to accept at least three parameters: a Digraph,\n        a source node, and a target node. And return a residual network\n        that follows NetworkX conventions (see :meth:`maximum_flow` for\n        details). If flow_func is None, the default maximum flow function\n        (:meth:`edmonds_karp`) is used. The choice of the default function\n        may change from version to version and should not be relied on.\n        Default value: None.\n\n    cutoff : integer or None (default: None)\n        Maximum number of paths to yield. If specified, the maximum flow\n        algorithm will terminate when the flow value reaches or exceeds the\n        cutoff. This only works for flows that support the cutoff parameter\n        (most do) and is ignored otherwise.\n\n    auxiliary : NetworkX DiGraph\n        Auxiliary digraph to compute flow based edge connectivity. It has\n        to have a graph attribute called mapping with a dictionary mapping\n        node names in G and in the auxiliary digraph. If provided\n        it will be reused instead of recreated. Default value: None.\n\n    residual : NetworkX DiGraph\n        Residual network to compute maximum flow. If provided it will be\n        reused instead of recreated. Default value: None.\n\n    Returns\n    -------\n    paths : generator\n        A generator of edge independent paths.\n\n    Raises\n    ------\n    NetworkXNoPath\n        If there is no path between source and target.\n\n    NetworkXError\n        If source or target are not in the graph G.\n\n    See also\n    --------\n    :meth:`node_disjoint_paths`\n    :meth:`edge_connectivity`\n    :meth:`maximum_flow`\n    :meth:`edmonds_karp`\n    :meth:`preflow_push`\n    :meth:`shortest_augmenting_path`\n\n    Examples\n    --------\n    We use in this example the platonic icosahedral graph, which has node\n    edge connectivity 5, thus there are 5 edge disjoint paths between any\n    pair of nodes.\n\n    >>> G = nx.icosahedral_graph()\n    >>> len(list(nx.edge_disjoint_paths(G, 0, 6)))\n    5\n\n\n    If you need to compute edge disjoint paths on several pairs of\n    nodes in the same graph, it is recommended that you reuse the\n    data structures that NetworkX uses in the computation: the\n    auxiliary digraph for edge connectivity, and the residual\n    network for the underlying maximum flow computation.\n\n    Example of how to compute edge disjoint paths among all pairs of\n    nodes of the platonic icosahedral graph reusing the data\n    structures.\n\n    >>> import itertools\n    >>> # You also have to explicitly import the function for\n    >>> # building the auxiliary digraph from the connectivity package\n    >>> from networkx.algorithms.connectivity import build_auxiliary_edge_connectivity\n    >>> H = build_auxiliary_edge_connectivity(G)\n    >>> # And the function for building the residual network from the\n    >>> # flow package\n    >>> from networkx.algorithms.flow import build_residual_network\n    >>> # Note that the auxiliary digraph has an edge attribute named capacity\n    >>> R = build_residual_network(H, "capacity")\n    >>> result = {n: {} for n in G}\n    >>> # Reuse the auxiliary digraph and the residual network by passing them\n    >>> # as arguments\n    >>> for u, v in itertools.combinations(G, 2):\n    ...     k = len(list(nx.edge_disjoint_paths(G, u, v, auxiliary=H, residual=R)))\n    ...     result[u][v] = k\n    >>> all(result[u][v] == 5 for u, v in itertools.combinations(G, 2))\n    True\n\n    You can also use alternative flow algorithms for computing edge disjoint\n    paths. For instance, in dense networks the algorithm\n    :meth:`shortest_augmenting_path` will usually perform better than\n    the default :meth:`edmonds_karp` which is faster for sparse\n    networks with highly skewed degree distributions. Alternative flow\n    functions have to be explicitly imported from the flow package.\n\n    >>> from networkx.algorithms.flow import shortest_augmenting_path\n    >>> len(list(nx.edge_disjoint_paths(G, 0, 6, flow_func=shortest_augmenting_path)))\n    5\n\n    Notes\n    -----\n    This is a flow based implementation of edge disjoint paths. We compute\n    the maximum flow between source and target on an auxiliary directed\n    network. The saturated edges in the residual network after running the\n    maximum flow algorithm correspond to edge disjoint paths between source\n    and target in the original network. This function handles both directed\n    and undirected graphs, and can use all flow algorithms from NetworkX flow\n    package.\n\n    '
    if s not in G:
        raise nx.NetworkXError(f'node {s} not in graph')
    if t not in G:
        raise nx.NetworkXError(f'node {t} not in graph')
    if flow_func is None:
        flow_func = default_flow_func
    if auxiliary is None:
        H = build_auxiliary_edge_connectivity(G)
    else:
        H = auxiliary
    possible = min(H.out_degree(s), H.in_degree(t))
    if not possible:
        raise NetworkXNoPath
    if cutoff is None:
        cutoff = possible
    else:
        cutoff = min(cutoff, possible)
    kwargs = {'capacity': 'capacity', 'residual': residual, 'cutoff': cutoff, 'value_only': True}
    if flow_func is preflow_push:
        del kwargs['cutoff']
    if flow_func is shortest_augmenting_path:
        kwargs['two_phase'] = True
    R = flow_func(H, s, t, **kwargs)
    if R.graph['flow_value'] == 0:
        raise NetworkXNoPath
    cutset = [(u, v) for (u, v, d) in R.edges(data=True) if d['capacity'] == d['flow'] and d['flow'] > 0]
    flow_dict = {n: {} for edge in cutset for n in edge}
    for (u, v) in cutset:
        flow_dict[u][v] = 1
    paths_found = 0
    for v in list(flow_dict[s]):
        if paths_found >= cutoff:
            break
        path = [s]
        if v == t:
            path.append(v)
            yield path
            continue
        u = v
        while u != t:
            path.append(u)
            try:
                (u, _) = flow_dict[u].popitem()
            except KeyError:
                break
        else:
            path.append(t)
            yield path
            paths_found += 1

@nx._dispatch(graphs={'G': 0, 'auxiliary?': 5, 'residual?': 6}, preserve_edge_attrs={'residual': {'capacity': float('inf')}}, preserve_node_attrs={'auxiliary': {'id': None}}, preserve_graph_attrs={'auxiliary', 'residual'})
def node_disjoint_paths(G, s, t, flow_func=None, cutoff=None, auxiliary=None, residual=None):
    if False:
        i = 10
        return i + 15
    'Computes node disjoint paths between source and target.\n\n    Node disjoint paths are paths that only share their first and last\n    nodes. The number of node independent paths between two nodes is\n    equal to their local node connectivity.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n\n    s : node\n        Source node.\n\n    t : node\n        Target node.\n\n    flow_func : function\n        A function for computing the maximum flow among a pair of nodes.\n        The function has to accept at least three parameters: a Digraph,\n        a source node, and a target node. And return a residual network\n        that follows NetworkX conventions (see :meth:`maximum_flow` for\n        details). If flow_func is None, the default maximum flow function\n        (:meth:`edmonds_karp`) is used. See below for details. The choice\n        of the default function may change from version to version and\n        should not be relied on. Default value: None.\n\n    cutoff : integer or None (default: None)\n        Maximum number of paths to yield. If specified, the maximum flow\n        algorithm will terminate when the flow value reaches or exceeds the\n        cutoff. This only works for flows that support the cutoff parameter\n        (most do) and is ignored otherwise.\n\n    auxiliary : NetworkX DiGraph\n        Auxiliary digraph to compute flow based node connectivity. It has\n        to have a graph attribute called mapping with a dictionary mapping\n        node names in G and in the auxiliary digraph. If provided\n        it will be reused instead of recreated. Default value: None.\n\n    residual : NetworkX DiGraph\n        Residual network to compute maximum flow. If provided it will be\n        reused instead of recreated. Default value: None.\n\n    Returns\n    -------\n    paths : generator\n        Generator of node disjoint paths.\n\n    Raises\n    ------\n    NetworkXNoPath\n        If there is no path between source and target.\n\n    NetworkXError\n        If source or target are not in the graph G.\n\n    Examples\n    --------\n    We use in this example the platonic icosahedral graph, which has node\n    connectivity 5, thus there are 5 node disjoint paths between any pair\n    of non neighbor nodes.\n\n    >>> G = nx.icosahedral_graph()\n    >>> len(list(nx.node_disjoint_paths(G, 0, 6)))\n    5\n\n    If you need to compute node disjoint paths between several pairs of\n    nodes in the same graph, it is recommended that you reuse the\n    data structures that NetworkX uses in the computation: the\n    auxiliary digraph for node connectivity and node cuts, and the\n    residual network for the underlying maximum flow computation.\n\n    Example of how to compute node disjoint paths reusing the data\n    structures:\n\n    >>> # You also have to explicitly import the function for\n    >>> # building the auxiliary digraph from the connectivity package\n    >>> from networkx.algorithms.connectivity import build_auxiliary_node_connectivity\n    >>> H = build_auxiliary_node_connectivity(G)\n    >>> # And the function for building the residual network from the\n    >>> # flow package\n    >>> from networkx.algorithms.flow import build_residual_network\n    >>> # Note that the auxiliary digraph has an edge attribute named capacity\n    >>> R = build_residual_network(H, "capacity")\n    >>> # Reuse the auxiliary digraph and the residual network by passing them\n    >>> # as arguments\n    >>> len(list(nx.node_disjoint_paths(G, 0, 6, auxiliary=H, residual=R)))\n    5\n\n    You can also use alternative flow algorithms for computing node disjoint\n    paths. For instance, in dense networks the algorithm\n    :meth:`shortest_augmenting_path` will usually perform better than\n    the default :meth:`edmonds_karp` which is faster for sparse\n    networks with highly skewed degree distributions. Alternative flow\n    functions have to be explicitly imported from the flow package.\n\n    >>> from networkx.algorithms.flow import shortest_augmenting_path\n    >>> len(list(nx.node_disjoint_paths(G, 0, 6, flow_func=shortest_augmenting_path)))\n    5\n\n    Notes\n    -----\n    This is a flow based implementation of node disjoint paths. We compute\n    the maximum flow between source and target on an auxiliary directed\n    network. The saturated edges in the residual network after running the\n    maximum flow algorithm correspond to node disjoint paths between source\n    and target in the original network. This function handles both directed\n    and undirected graphs, and can use all flow algorithms from NetworkX flow\n    package.\n\n    See also\n    --------\n    :meth:`edge_disjoint_paths`\n    :meth:`node_connectivity`\n    :meth:`maximum_flow`\n    :meth:`edmonds_karp`\n    :meth:`preflow_push`\n    :meth:`shortest_augmenting_path`\n\n    '
    if s not in G:
        raise nx.NetworkXError(f'node {s} not in graph')
    if t not in G:
        raise nx.NetworkXError(f'node {t} not in graph')
    if auxiliary is None:
        H = build_auxiliary_node_connectivity(G)
    else:
        H = auxiliary
    mapping = H.graph.get('mapping', None)
    if mapping is None:
        raise nx.NetworkXError('Invalid auxiliary digraph.')
    possible = min(H.out_degree(f'{mapping[s]}B'), H.in_degree(f'{mapping[t]}A'))
    if not possible:
        raise NetworkXNoPath
    if cutoff is None:
        cutoff = possible
    else:
        cutoff = min(cutoff, possible)
    kwargs = {'flow_func': flow_func, 'residual': residual, 'auxiliary': H, 'cutoff': cutoff}
    paths_edges = edge_disjoint_paths(H, f'{mapping[s]}B', f'{mapping[t]}A', **kwargs)
    for path in paths_edges:
        yield list(_unique_everseen((H.nodes[node]['id'] for node in path)))

def _unique_everseen(iterable):
    if False:
        return 10
    'List unique elements, preserving order. Remember all elements ever seen.'
    seen = set()
    seen_add = seen.add
    for element in _filterfalse(seen.__contains__, iterable):
        seen_add(element)
        yield element