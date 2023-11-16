from collections import defaultdict
import networkx as nx
__all__ = ['k_clique_communities']

@nx._dispatch
def k_clique_communities(G, k, cliques=None):
    if False:
        print('Hello World!')
    'Find k-clique communities in graph using the percolation method.\n\n    A k-clique community is the union of all cliques of size k that\n    can be reached through adjacent (sharing k-1 nodes) k-cliques.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n\n    k : int\n       Size of smallest clique\n\n    cliques: list or generator\n       Precomputed cliques (use networkx.find_cliques(G))\n\n    Returns\n    -------\n    Yields sets of nodes, one for each k-clique community.\n\n    Examples\n    --------\n    >>> G = nx.complete_graph(5)\n    >>> K5 = nx.convert_node_labels_to_integers(G, first_label=2)\n    >>> G.add_edges_from(K5.edges())\n    >>> c = list(nx.community.k_clique_communities(G, 4))\n    >>> sorted(list(c[0]))\n    [0, 1, 2, 3, 4, 5, 6]\n    >>> list(nx.community.k_clique_communities(G, 6))\n    []\n\n    References\n    ----------\n    .. [1] Gergely Palla, Imre Derényi, Illés Farkas1, and Tamás Vicsek,\n       Uncovering the overlapping community structure of complex networks\n       in nature and society Nature 435, 814-818, 2005,\n       doi:10.1038/nature03607\n    '
    if k < 2:
        raise nx.NetworkXError(f'k={k}, k must be greater than 1.')
    if cliques is None:
        cliques = nx.find_cliques(G)
    cliques = [frozenset(c) for c in cliques if len(c) >= k]
    membership_dict = defaultdict(list)
    for clique in cliques:
        for node in clique:
            membership_dict[node].append(clique)
    perc_graph = nx.Graph()
    perc_graph.add_nodes_from(cliques)
    for clique in cliques:
        for adj_clique in _get_adjacent_cliques(clique, membership_dict):
            if len(clique.intersection(adj_clique)) >= k - 1:
                perc_graph.add_edge(clique, adj_clique)
    for component in nx.connected_components(perc_graph):
        yield frozenset.union(*component)

def _get_adjacent_cliques(clique, membership_dict):
    if False:
        i = 10
        return i + 15
    adjacent_cliques = set()
    for n in clique:
        for adj_clique in membership_dict[n]:
            if clique != adj_clique:
                adjacent_cliques.add(adj_clique)
    return adjacent_cliques