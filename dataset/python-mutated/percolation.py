"""Percolation centrality measures."""
import networkx as nx
from networkx.algorithms.centrality.betweenness import _single_source_dijkstra_path_basic as dijkstra
from networkx.algorithms.centrality.betweenness import _single_source_shortest_path_basic as shortest_path
__all__ = ['percolation_centrality']

@nx._dispatch(node_attrs='attribute', edge_attrs='weight')
def percolation_centrality(G, attribute='percolation', states=None, weight=None):
    if False:
        for i in range(10):
            print('nop')
    "Compute the percolation centrality for nodes.\n\n    Percolation centrality of a node $v$, at a given time, is defined\n    as the proportion of ‘percolated paths’ that go through that node.\n\n    This measure quantifies relative impact of nodes based on their\n    topological connectivity, as well as their percolation states.\n\n    Percolation states of nodes are used to depict network percolation\n    scenarios (such as during infection transmission in a social network\n    of individuals, spreading of computer viruses on computer networks, or\n    transmission of disease over a network of towns) over time. In this\n    measure usually the percolation state is expressed as a decimal\n    between 0.0 and 1.0.\n\n    When all nodes are in the same percolated state this measure is\n    equivalent to betweenness centrality.\n\n    Parameters\n    ----------\n    G : graph\n      A NetworkX graph.\n\n    attribute : None or string, optional (default='percolation')\n      Name of the node attribute to use for percolation state, used\n      if `states` is None. If a node does not set the attribute the\n      state of that node will be set to the default value of 1.\n      If all nodes do not have the attribute all nodes will be set to\n      1 and the centrality measure will be equivalent to betweenness centrality.\n\n    states : None or dict, optional (default=None)\n      Specify percolation states for the nodes, nodes as keys states\n      as values.\n\n    weight : None or string, optional (default=None)\n      If None, all edge weights are considered equal.\n      Otherwise holds the name of the edge attribute used as weight.\n      The weight of an edge is treated as the length or distance between the two sides.\n\n\n    Returns\n    -------\n    nodes : dictionary\n       Dictionary of nodes with percolation centrality as the value.\n\n    See Also\n    --------\n    betweenness_centrality\n\n    Notes\n    -----\n    The algorithm is from Mahendra Piraveenan, Mikhail Prokopenko, and\n    Liaquat Hossain [1]_\n    Pair dependencies are calculated and accumulated using [2]_\n\n    For weighted graphs the edge weights must be greater than zero.\n    Zero edge weights can produce an infinite number of equal length\n    paths between pairs of nodes.\n\n    References\n    ----------\n    .. [1] Mahendra Piraveenan, Mikhail Prokopenko, Liaquat Hossain\n       Percolation Centrality: Quantifying Graph-Theoretic Impact of Nodes\n       during Percolation in Networks\n       http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0053095\n    .. [2] Ulrik Brandes:\n       A Faster Algorithm for Betweenness Centrality.\n       Journal of Mathematical Sociology 25(2):163-177, 2001.\n       https://doi.org/10.1080/0022250X.2001.9990249\n    "
    percolation = dict.fromkeys(G, 0.0)
    nodes = G
    if states is None:
        states = nx.get_node_attributes(nodes, attribute, default=1)
    p_sigma_x_t = 0.0
    for v in states.values():
        p_sigma_x_t += v
    for s in nodes:
        if weight is None:
            (S, P, sigma, _) = shortest_path(G, s)
        else:
            (S, P, sigma, _) = dijkstra(G, s, weight)
        percolation = _accumulate_percolation(percolation, S, P, sigma, s, states, p_sigma_x_t)
    n = len(G)
    for v in percolation:
        percolation[v] *= 1 / (n - 2)
    return percolation

def _accumulate_percolation(percolation, S, P, sigma, s, states, p_sigma_x_t):
    if False:
        print('Hello World!')
    delta = dict.fromkeys(S, 0)
    while S:
        w = S.pop()
        coeff = (1 + delta[w]) / sigma[w]
        for v in P[w]:
            delta[v] += sigma[v] * coeff
        if w != s:
            pw_s_w = states[s] / (p_sigma_x_t - states[w])
            percolation[w] += delta[w] * pw_s_w
    return percolation