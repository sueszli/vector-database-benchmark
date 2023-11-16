from itertools import chain
import networkx as nx
from networkx.utils import not_implemented_for, pairwise
__all__ = ['metric_closure', 'steiner_tree']

@not_implemented_for('directed')
@nx._dispatch(edge_attrs='weight')
def metric_closure(G, weight='weight'):
    if False:
        for i in range(10):
            print('nop')
    'Return the metric closure of a graph.\n\n    The metric closure of a graph *G* is the complete graph in which each edge\n    is weighted by the shortest path distance between the nodes in *G* .\n\n    Parameters\n    ----------\n    G : NetworkX graph\n\n    Returns\n    -------\n    NetworkX graph\n        Metric closure of the graph `G`.\n\n    '
    M = nx.Graph()
    Gnodes = set(G)
    all_paths_iter = nx.all_pairs_dijkstra(G, weight=weight)
    (u, (distance, path)) = next(all_paths_iter)
    if Gnodes - set(distance):
        msg = 'G is not a connected graph. metric_closure is not defined.'
        raise nx.NetworkXError(msg)
    Gnodes.remove(u)
    for v in Gnodes:
        M.add_edge(u, v, distance=distance[v], path=path[v])
    for (u, (distance, path)) in all_paths_iter:
        Gnodes.remove(u)
        for v in Gnodes:
            M.add_edge(u, v, distance=distance[v], path=path[v])
    return M

def _mehlhorn_steiner_tree(G, terminal_nodes, weight):
    if False:
        while True:
            i = 10
    paths = nx.multi_source_dijkstra_path(G, terminal_nodes)
    d_1 = {}
    s = {}
    for v in G.nodes():
        s[v] = paths[v][0]
        d_1[v, s[v]] = len(paths[v]) - 1
    G_1_prime = nx.Graph()
    for (u, v, data) in G.edges(data=True):
        (su, sv) = (s[u], s[v])
        weight_here = d_1[u, su] + data.get(weight, 1) + d_1[v, sv]
        if not G_1_prime.has_edge(su, sv):
            G_1_prime.add_edge(su, sv, weight=weight_here)
        else:
            new_weight = min(weight_here, G_1_prime[su][sv][weight])
            G_1_prime.add_edge(su, sv, weight=new_weight)
    G_2 = nx.minimum_spanning_edges(G_1_prime, data=True)
    G_3 = nx.Graph()
    for (u, v, d) in G_2:
        path = nx.shortest_path(G, u, v, weight)
        for (n1, n2) in pairwise(path):
            G_3.add_edge(n1, n2)
    G_3_mst = list(nx.minimum_spanning_edges(G_3, data=False))
    if G.is_multigraph():
        G_3_mst = ((u, v, min(G[u][v], key=lambda k: G[u][v][k][weight])) for (u, v) in G_3_mst)
    G_4 = G.edge_subgraph(G_3_mst).copy()
    _remove_nonterminal_leaves(G_4, terminal_nodes)
    return G_4.edges()

def _kou_steiner_tree(G, terminal_nodes, weight):
    if False:
        i = 10
        return i + 15
    M = metric_closure(G, weight=weight)
    H = M.subgraph(terminal_nodes)
    mst_edges = nx.minimum_spanning_edges(H, weight='distance', data=True)
    mst_all_edges = chain.from_iterable((pairwise(d['path']) for (u, v, d) in mst_edges))
    if G.is_multigraph():
        mst_all_edges = ((u, v, min(G[u][v], key=lambda k: G[u][v][k][weight])) for (u, v) in mst_all_edges)
    G_S = G.edge_subgraph(mst_all_edges)
    T_S = nx.minimum_spanning_edges(G_S, weight='weight', data=False)
    T_H = G.edge_subgraph(T_S).copy()
    _remove_nonterminal_leaves(T_H, terminal_nodes)
    return T_H.edges()

def _remove_nonterminal_leaves(G, terminals):
    if False:
        i = 10
        return i + 15
    terminals_set = set(terminals)
    for n in list(G.nodes):
        if n not in terminals_set and G.degree(n) == 1:
            G.remove_node(n)
ALGORITHMS = {'kou': _kou_steiner_tree, 'mehlhorn': _mehlhorn_steiner_tree}

@not_implemented_for('directed')
@nx._dispatch(edge_attrs='weight')
def steiner_tree(G, terminal_nodes, weight='weight', method=None):
    if False:
        return 10
    'Return an approximation to the minimum Steiner tree of a graph.\n\n    The minimum Steiner tree of `G` w.r.t a set of `terminal_nodes` (also *S*)\n    is a tree within `G` that spans those nodes and has minimum size (sum of\n    edge weights) among all such trees.\n\n    The approximation algorithm is specified with the `method` keyword\n    argument. All three available algorithms produce a tree whose weight is\n    within a ``(2 - (2 / l))`` factor of the weight of the optimal Steiner tree,\n    where ``l`` is the minimum number of leaf nodes across all possible Steiner\n    trees.\n\n    * ``"kou"`` [2]_ (runtime $O(|S| |V|^2)$) computes the minimum spanning tree of\n      the subgraph of the metric closure of *G* induced by the terminal nodes,\n      where the metric closure of *G* is the complete graph in which each edge is\n      weighted by the shortest path distance between the nodes in *G*.\n\n    * ``"mehlhorn"`` [3]_ (runtime $O(|E|+|V|\\log|V|)$) modifies Kou et al.\'s\n      algorithm, beginning by finding the closest terminal node for each\n      non-terminal. This data is used to create a complete graph containing only\n      the terminal nodes, in which edge is weighted with the shortest path\n      distance between them. The algorithm then proceeds in the same way as Kou\n      et al..\n\n    Parameters\n    ----------\n    G : NetworkX graph\n\n    terminal_nodes : list\n         A list of terminal nodes for which minimum steiner tree is\n         to be found.\n\n    weight : string (default = \'weight\')\n        Use the edge attribute specified by this string as the edge weight.\n        Any edge attribute not present defaults to 1.\n\n    method : string, optional (default = \'kou\')\n        The algorithm to use to approximate the Steiner tree.\n        Supported options: \'kou\', \'mehlhorn\'.\n        Other inputs produce a ValueError.\n\n    Returns\n    -------\n    NetworkX graph\n        Approximation to the minimum steiner tree of `G` induced by\n        `terminal_nodes` .\n\n    Notes\n    -----\n    For multigraphs, the edge between two nodes with minimum weight is the\n    edge put into the Steiner tree.\n\n\n    References\n    ----------\n    .. [1] Steiner_tree_problem on Wikipedia.\n           https://en.wikipedia.org/wiki/Steiner_tree_problem\n    .. [2] Kou, L., G. Markowsky, and L. Berman. 1981.\n           ‘A Fast Algorithm for Steiner Trees’.\n           Acta Informatica 15 (2): 141–45.\n           https://doi.org/10.1007/BF00288961.\n    .. [3] Mehlhorn, Kurt. 1988.\n           ‘A Faster Approximation Algorithm for the Steiner Problem in Graphs’.\n           Information Processing Letters 27 (3): 125–28.\n           https://doi.org/10.1016/0020-0190(88)90066-X.\n    '
    if method is None:
        import warnings
        msg = "steiner_tree will change default method from 'kou' to 'mehlhorn' in version 3.2.\nSet the `method` kwarg to remove this warning."
        warnings.warn(msg, FutureWarning, stacklevel=4)
        method = 'kou'
    try:
        algo = ALGORITHMS[method]
    except KeyError as e:
        msg = f'{method} is not a valid choice for an algorithm.'
        raise ValueError(msg) from e
    edges = algo(G, terminal_nodes, weight)
    if G.is_multigraph():
        edges = ((u, v, min(G[u][v], key=lambda k: G[u][v][k][weight])) for (u, v) in edges)
    T = G.edge_subgraph(edges)
    return T