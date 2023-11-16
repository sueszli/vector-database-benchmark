import networkx as nx
__all__ = ['average_neighbor_degree']

@nx._dispatch(edge_attrs='weight')
def average_neighbor_degree(G, source='out', target='out', nodes=None, weight=None):
    if False:
        for i in range(10):
            print('nop')
    'Returns the average degree of the neighborhood of each node.\n\n    In an undirected graph, the neighborhood `N(i)` of node `i` contains the\n    nodes that are connected to `i` by an edge.\n\n    For directed graphs, `N(i)` is defined according to the parameter `source`:\n\n        - if source is \'in\', then `N(i)` consists of predecessors of node `i`.\n        - if source is \'out\', then `N(i)` consists of successors of node `i`.\n        - if source is \'in+out\', then `N(i)` is both predecessors and successors.\n\n    The average neighborhood degree of a node `i` is\n\n    .. math::\n\n        k_{nn,i} = \\frac{1}{|N(i)|} \\sum_{j \\in N(i)} k_j\n\n    where `N(i)` are the neighbors of node `i` and `k_j` is\n    the degree of node `j` which belongs to `N(i)`. For weighted\n    graphs, an analogous measure can be defined [1]_,\n\n    .. math::\n\n        k_{nn,i}^{w} = \\frac{1}{s_i} \\sum_{j \\in N(i)} w_{ij} k_j\n\n    where `s_i` is the weighted degree of node `i`, `w_{ij}`\n    is the weight of the edge that links `i` and `j` and\n    `N(i)` are the neighbors of node `i`.\n\n\n    Parameters\n    ----------\n    G : NetworkX graph\n\n    source : string ("in"|"out"|"in+out"), optional (default="out")\n       Directed graphs only.\n       Use "in"- or "out"-neighbors of source node.\n\n    target : string ("in"|"out"|"in+out"), optional (default="out")\n       Directed graphs only.\n       Use "in"- or "out"-degree for target node.\n\n    nodes : list or iterable, optional (default=G.nodes)\n        Compute neighbor degree only for specified nodes.\n\n    weight : string or None, optional (default=None)\n       The edge attribute that holds the numerical value used as a weight.\n       If None, then each edge has weight 1.\n\n    Returns\n    -------\n    d: dict\n       A dictionary keyed by node to the average degree of its neighbors.\n\n    Raises\n    ------\n    NetworkXError\n        If either `source` or `target` are not one of \'in\', \'out\', or \'in+out\'.\n        If either `source` or `target` is passed for an undirected graph.\n\n    Examples\n    --------\n    >>> G = nx.path_graph(4)\n    >>> G.edges[0, 1]["weight"] = 5\n    >>> G.edges[2, 3]["weight"] = 3\n\n    >>> nx.average_neighbor_degree(G)\n    {0: 2.0, 1: 1.5, 2: 1.5, 3: 2.0}\n    >>> nx.average_neighbor_degree(G, weight="weight")\n    {0: 2.0, 1: 1.1666666666666667, 2: 1.25, 3: 2.0}\n\n    >>> G = nx.DiGraph()\n    >>> nx.add_path(G, [0, 1, 2, 3])\n    >>> nx.average_neighbor_degree(G, source="in", target="in")\n    {0: 0.0, 1: 0.0, 2: 1.0, 3: 1.0}\n\n    >>> nx.average_neighbor_degree(G, source="out", target="out")\n    {0: 1.0, 1: 1.0, 2: 0.0, 3: 0.0}\n\n    See Also\n    --------\n    average_degree_connectivity\n\n    References\n    ----------\n    .. [1] A. Barrat, M. Barthélemy, R. Pastor-Satorras, and A. Vespignani,\n       "The architecture of complex weighted networks".\n       PNAS 101 (11): 3747–3752 (2004).\n    '
    if G.is_directed():
        if source == 'in':
            source_degree = G.in_degree
        elif source == 'out':
            source_degree = G.out_degree
        elif source == 'in+out':
            source_degree = G.degree
        else:
            raise nx.NetworkXError(f"source argument {source} must be 'in', 'out' or 'in+out'")
        if target == 'in':
            target_degree = G.in_degree
        elif target == 'out':
            target_degree = G.out_degree
        elif target == 'in+out':
            target_degree = G.degree
        else:
            raise nx.NetworkXError(f"target argument {target} must be 'in', 'out' or 'in+out'")
    else:
        if source != 'out' or target != 'out':
            raise nx.NetworkXError(f'source and target arguments are only supported for directed graphs')
        source_degree = target_degree = G.degree
    t_deg = dict(target_degree())
    G_P = G_S = {n: {} for n in G}
    if G.is_directed():
        if 'in' in source:
            G_P = G.pred
        if 'out' in source:
            G_S = G.succ
    else:
        G_S = G.adj
    avg = {}
    for (n, deg) in source_degree(nodes, weight=weight):
        if deg == 0:
            avg[n] = 0.0
            continue
        if weight is None:
            avg[n] = (sum((t_deg[nbr] for nbr in G_S[n])) + sum((t_deg[nbr] for nbr in G_P[n]))) / deg
        else:
            avg[n] = (sum((dd.get(weight, 1) * t_deg[nbr] for (nbr, dd) in G_S[n].items())) + sum((dd.get(weight, 1) * t_deg[nbr] for (nbr, dd) in G_P[n].items()))) / deg
    return avg