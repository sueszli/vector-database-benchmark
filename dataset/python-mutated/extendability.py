""" Provides a function for computing the extendability of a graph which is
undirected, simple, connected and bipartite and contains at least one perfect matching."""
import networkx as nx
from networkx.utils import not_implemented_for
__all__ = ['maximal_extendability']

@not_implemented_for('directed')
@not_implemented_for('multigraph')
def maximal_extendability(G):
    if False:
        i = 10
        return i + 15
    'Computes the extendability of a graph.\n\n    The extendability of a graph is defined as the maximum $k$ for which `G`\n    is $k$-extendable. Graph `G` is $k$-extendable if and only if `G` has a\n    perfect matching and every set of $k$ independent edges can be extended\n    to a perfect matching in `G`.\n\n    Parameters\n    ----------\n    G : NetworkX Graph\n        A fully-connected bipartite graph without self-loops\n\n    Returns\n    -------\n    extendability : int\n\n    Raises\n    ------\n    NetworkXError\n       If the graph `G` is disconnected.\n       If the graph `G` is not bipartite.\n       If the graph `G` does not contain a perfect matching.\n       If the residual graph of `G` is not strongly connected.\n\n    Notes\n    -----\n    Definition:\n    Let `G` be a simple, connected, undirected and bipartite graph with a perfect\n    matching M and bipartition (U,V). The residual graph of `G`, denoted by $G_M$,\n    is the graph obtained from G by directing the edges of M from V to U and the\n    edges that do not belong to M from U to V.\n\n    Lemma [1]_ :\n    Let M be a perfect matching of `G`. `G` is $k$-extendable if and only if its residual\n    graph $G_M$ is strongly connected and there are $k$ vertex-disjoint directed\n    paths between every vertex of U and every vertex of V.\n\n    Assuming that input graph `G` is undirected, simple, connected, bipartite and contains\n    a perfect matching M, this function constructs the residual graph $G_M$ of G and\n    returns the minimum value among the maximum vertex-disjoint directed paths between\n    every vertex of U and every vertex of V in $G_M$. By combining the definitions\n    and the lemma, this value represents the extendability of the graph `G`.\n\n    Time complexity O($n^3$ $m^2$)) where $n$ is the number of vertices\n    and $m$ is the number of edges.\n\n    References\n    ----------\n    .. [1] "A polynomial algorithm for the extendability problem in bipartite graphs",\n          J. Lakhal, L. Litzler, Information Processing Letters, 1998.\n    .. [2] "On n-extendible graphs", M. D. Plummer, Discrete Mathematics, 31:201–210, 1980\n          https://doi.org/10.1016/0012-365X(80)90037-0\n\n    '
    if not nx.is_connected(G):
        raise nx.NetworkXError('Graph G is not connected')
    if not nx.bipartite.is_bipartite(G):
        raise nx.NetworkXError('Graph G is not bipartite')
    (U, V) = nx.bipartite.sets(G)
    maximum_matching = nx.bipartite.hopcroft_karp_matching(G)
    if not nx.is_perfect_matching(G, maximum_matching):
        raise nx.NetworkXError('Graph G does not contain a perfect matching')
    pm = [(node, maximum_matching[node]) for node in V & maximum_matching.keys()]
    directed_edges = [(x, y) if x in V and (x, y) in pm or (x in U and (y, x) not in pm) else (y, x) for (x, y) in G.edges]
    residual_G = nx.DiGraph()
    residual_G.add_nodes_from(G)
    residual_G.add_edges_from(directed_edges)
    if not nx.is_strongly_connected(residual_G):
        raise nx.NetworkXError('The residual graph of G is not strongly connected')
    k = float('Inf')
    for u in U:
        for v in V:
            num_paths = sum((1 for _ in nx.node_disjoint_paths(residual_G, u, v)))
            k = k if k < num_paths else num_paths
    return k