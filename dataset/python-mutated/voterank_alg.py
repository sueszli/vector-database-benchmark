"""Algorithm to select influential nodes in a graph using VoteRank."""
import networkx as nx
__all__ = ['voterank']

@nx._dispatch
def voterank(G, number_of_nodes=None):
    if False:
        for i in range(10):
            print('nop')
    'Select a list of influential nodes in a graph using VoteRank algorithm\n\n    VoteRank [1]_ computes a ranking of the nodes in a graph G based on a\n    voting scheme. With VoteRank, all nodes vote for each of its in-neighbours\n    and the node with the highest votes is elected iteratively. The voting\n    ability of out-neighbors of elected nodes is decreased in subsequent turns.\n\n    Parameters\n    ----------\n    G : graph\n        A NetworkX graph.\n\n    number_of_nodes : integer, optional\n        Number of ranked nodes to extract (default all nodes).\n\n    Returns\n    -------\n    voterank : list\n        Ordered list of computed seeds.\n        Only nodes with positive number of votes are returned.\n\n    Examples\n    --------\n    >>> G = nx.Graph([(0, 1), (0, 2), (0, 3), (1, 4)])\n    >>> nx.voterank(G)\n    [0, 1]\n\n    The algorithm can be used both for undirected and directed graphs.\n    However, the directed version is different in two ways:\n    (i) nodes only vote for their in-neighbors and\n    (ii) only the voting ability of elected node and its out-neighbors are updated:\n\n    >>> G = nx.DiGraph([(0, 1), (2, 1), (2, 3), (3, 4)])\n    >>> nx.voterank(G)\n    [2, 3]\n\n    Notes\n    -----\n    Each edge is treated independently in case of multigraphs.\n\n    References\n    ----------\n    .. [1] Zhang, J.-X. et al. (2016).\n        Identifying a set of influential spreaders in complex networks.\n        Sci. Rep. 6, 27823; doi: 10.1038/srep27823.\n    '
    influential_nodes = []
    vote_rank = {}
    if len(G) == 0:
        return influential_nodes
    if number_of_nodes is None or number_of_nodes > len(G):
        number_of_nodes = len(G)
    if G.is_directed():
        avgDegree = sum((deg for (_, deg) in G.out_degree())) / len(G)
    else:
        avgDegree = sum((deg for (_, deg) in G.degree())) / len(G)
    for n in G.nodes():
        vote_rank[n] = [0, 1]
    for _ in range(number_of_nodes):
        for n in G.nodes():
            vote_rank[n][0] = 0
        for (n, nbr) in G.edges():
            vote_rank[n][0] += vote_rank[nbr][1]
            if not G.is_directed():
                vote_rank[nbr][0] += vote_rank[n][1]
        for n in influential_nodes:
            vote_rank[n][0] = 0
        n = max(G.nodes, key=lambda x: vote_rank[x][0])
        if vote_rank[n][0] == 0:
            return influential_nodes
        influential_nodes.append(n)
        vote_rank[n] = [0, 0]
        for (_, nbr) in G.edges(n):
            vote_rank[nbr][1] -= 1 / avgDegree
            vote_rank[nbr][1] = max(vote_rank[nbr][1], 0)
    return influential_nodes