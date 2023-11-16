"""
Link prediction algorithms.
"""
from math import log
import networkx as nx
from networkx.utils import not_implemented_for
__all__ = ['resource_allocation_index', 'jaccard_coefficient', 'adamic_adar_index', 'preferential_attachment', 'cn_soundarajan_hopcroft', 'ra_index_soundarajan_hopcroft', 'within_inter_cluster', 'common_neighbor_centrality']

def _apply_prediction(G, func, ebunch=None):
    if False:
        i = 10
        return i + 15
    'Applies the given function to each edge in the specified iterable\n    of edges.\n\n    `G` is an instance of :class:`networkx.Graph`.\n\n    `func` is a function on two inputs, each of which is a node in the\n    graph. The function can return anything, but it should return a\n    value representing a prediction of the likelihood of a "link"\n    joining the two nodes.\n\n    `ebunch` is an iterable of pairs of nodes. If not specified, all\n    non-edges in the graph `G` will be used.\n\n    '
    if ebunch is None:
        ebunch = nx.non_edges(G)
    return ((u, v, func(u, v)) for (u, v) in ebunch)

@not_implemented_for('directed')
@not_implemented_for('multigraph')
@nx._dispatch
def resource_allocation_index(G, ebunch=None):
    if False:
        for i in range(10):
            print('nop')
    'Compute the resource allocation index of all node pairs in ebunch.\n\n    Resource allocation index of `u` and `v` is defined as\n\n    .. math::\n\n        \\sum_{w \\in \\Gamma(u) \\cap \\Gamma(v)} \\frac{1}{|\\Gamma(w)|}\n\n    where $\\Gamma(u)$ denotes the set of neighbors of $u$.\n\n    Parameters\n    ----------\n    G : graph\n        A NetworkX undirected graph.\n\n    ebunch : iterable of node pairs, optional (default = None)\n        Resource allocation index will be computed for each pair of\n        nodes given in the iterable. The pairs must be given as\n        2-tuples (u, v) where u and v are nodes in the graph. If ebunch\n        is None then all nonexistent edges in the graph will be used.\n        Default value: None.\n\n    Returns\n    -------\n    piter : iterator\n        An iterator of 3-tuples in the form (u, v, p) where (u, v) is a\n        pair of nodes and p is their resource allocation index.\n\n    Examples\n    --------\n    >>> G = nx.complete_graph(5)\n    >>> preds = nx.resource_allocation_index(G, [(0, 1), (2, 3)])\n    >>> for u, v, p in preds:\n    ...     print(f"({u}, {v}) -> {p:.8f}")\n    (0, 1) -> 0.75000000\n    (2, 3) -> 0.75000000\n\n    References\n    ----------\n    .. [1] T. Zhou, L. Lu, Y.-C. Zhang.\n       Predicting missing links via local information.\n       Eur. Phys. J. B 71 (2009) 623.\n       https://arxiv.org/pdf/0901.0553.pdf\n    '

    def predict(u, v):
        if False:
            i = 10
            return i + 15
        return sum((1 / G.degree(w) for w in nx.common_neighbors(G, u, v)))
    return _apply_prediction(G, predict, ebunch)

@not_implemented_for('directed')
@not_implemented_for('multigraph')
@nx._dispatch
def jaccard_coefficient(G, ebunch=None):
    if False:
        while True:
            i = 10
    'Compute the Jaccard coefficient of all node pairs in ebunch.\n\n    Jaccard coefficient of nodes `u` and `v` is defined as\n\n    .. math::\n\n        \\frac{|\\Gamma(u) \\cap \\Gamma(v)|}{|\\Gamma(u) \\cup \\Gamma(v)|}\n\n    where $\\Gamma(u)$ denotes the set of neighbors of $u$.\n\n    Parameters\n    ----------\n    G : graph\n        A NetworkX undirected graph.\n\n    ebunch : iterable of node pairs, optional (default = None)\n        Jaccard coefficient will be computed for each pair of nodes\n        given in the iterable. The pairs must be given as 2-tuples\n        (u, v) where u and v are nodes in the graph. If ebunch is None\n        then all nonexistent edges in the graph will be used.\n        Default value: None.\n\n    Returns\n    -------\n    piter : iterator\n        An iterator of 3-tuples in the form (u, v, p) where (u, v) is a\n        pair of nodes and p is their Jaccard coefficient.\n\n    Examples\n    --------\n    >>> G = nx.complete_graph(5)\n    >>> preds = nx.jaccard_coefficient(G, [(0, 1), (2, 3)])\n    >>> for u, v, p in preds:\n    ...     print(f"({u}, {v}) -> {p:.8f}")\n    (0, 1) -> 0.60000000\n    (2, 3) -> 0.60000000\n\n    References\n    ----------\n    .. [1] D. Liben-Nowell, J. Kleinberg.\n           The Link Prediction Problem for Social Networks (2004).\n           http://www.cs.cornell.edu/home/kleinber/link-pred.pdf\n    '

    def predict(u, v):
        if False:
            i = 10
            return i + 15
        union_size = len(set(G[u]) | set(G[v]))
        if union_size == 0:
            return 0
        return len(list(nx.common_neighbors(G, u, v))) / union_size
    return _apply_prediction(G, predict, ebunch)

@not_implemented_for('directed')
@not_implemented_for('multigraph')
@nx._dispatch
def adamic_adar_index(G, ebunch=None):
    if False:
        for i in range(10):
            print('nop')
    'Compute the Adamic-Adar index of all node pairs in ebunch.\n\n    Adamic-Adar index of `u` and `v` is defined as\n\n    .. math::\n\n        \\sum_{w \\in \\Gamma(u) \\cap \\Gamma(v)} \\frac{1}{\\log |\\Gamma(w)|}\n\n    where $\\Gamma(u)$ denotes the set of neighbors of $u$.\n    This index leads to zero-division for nodes only connected via self-loops.\n    It is intended to be used when no self-loops are present.\n\n    Parameters\n    ----------\n    G : graph\n        NetworkX undirected graph.\n\n    ebunch : iterable of node pairs, optional (default = None)\n        Adamic-Adar index will be computed for each pair of nodes given\n        in the iterable. The pairs must be given as 2-tuples (u, v)\n        where u and v are nodes in the graph. If ebunch is None then all\n        nonexistent edges in the graph will be used.\n        Default value: None.\n\n    Returns\n    -------\n    piter : iterator\n        An iterator of 3-tuples in the form (u, v, p) where (u, v) is a\n        pair of nodes and p is their Adamic-Adar index.\n\n    Examples\n    --------\n    >>> G = nx.complete_graph(5)\n    >>> preds = nx.adamic_adar_index(G, [(0, 1), (2, 3)])\n    >>> for u, v, p in preds:\n    ...     print(f"({u}, {v}) -> {p:.8f}")\n    (0, 1) -> 2.16404256\n    (2, 3) -> 2.16404256\n\n    References\n    ----------\n    .. [1] D. Liben-Nowell, J. Kleinberg.\n           The Link Prediction Problem for Social Networks (2004).\n           http://www.cs.cornell.edu/home/kleinber/link-pred.pdf\n    '

    def predict(u, v):
        if False:
            for i in range(10):
                print('nop')
        return sum((1 / log(G.degree(w)) for w in nx.common_neighbors(G, u, v)))
    return _apply_prediction(G, predict, ebunch)

@not_implemented_for('directed')
@not_implemented_for('multigraph')
@nx._dispatch
def common_neighbor_centrality(G, ebunch=None, alpha=0.8):
    if False:
        while True:
            i = 10
    'Return the CCPA score for each pair of nodes.\n\n    Compute the Common Neighbor and Centrality based Parameterized Algorithm(CCPA)\n    score of all node pairs in ebunch.\n\n    CCPA score of `u` and `v` is defined as\n\n    .. math::\n\n        \\alpha \\cdot (|\\Gamma (u){\\cap }^{}\\Gamma (v)|)+(1-\\alpha )\\cdot \\frac{N}{{d}_{uv}}\n\n    where $\\Gamma(u)$ denotes the set of neighbors of $u$, $\\Gamma(v)$ denotes the\n    set of neighbors of $v$, $\\alpha$ is  parameter varies between [0,1], $N$ denotes\n    total number of nodes in the Graph and ${d}_{uv}$ denotes shortest distance\n    between $u$ and $v$.\n\n    This algorithm is based on two vital properties of nodes, namely the number\n    of common neighbors and their centrality. Common neighbor refers to the common\n    nodes between two nodes. Centrality refers to the prestige that a node enjoys\n    in a network.\n\n    .. seealso::\n\n        :func:`common_neighbors`\n\n    Parameters\n    ----------\n    G : graph\n        NetworkX undirected graph.\n\n    ebunch : iterable of node pairs, optional (default = None)\n        Preferential attachment score will be computed for each pair of\n        nodes given in the iterable. The pairs must be given as\n        2-tuples (u, v) where u and v are nodes in the graph. If ebunch\n        is None then all nonexistent edges in the graph will be used.\n        Default value: None.\n\n    alpha : Parameter defined for participation of Common Neighbor\n            and Centrality Algorithm share. Values for alpha should\n            normally be between 0 and 1. Default value set to 0.8\n            because author found better performance at 0.8 for all the\n            dataset.\n            Default value: 0.8\n\n\n    Returns\n    -------\n    piter : iterator\n        An iterator of 3-tuples in the form (u, v, p) where (u, v) is a\n        pair of nodes and p is their Common Neighbor and Centrality based\n        Parameterized Algorithm(CCPA) score.\n\n    Examples\n    --------\n    >>> G = nx.complete_graph(5)\n    >>> preds = nx.common_neighbor_centrality(G, [(0, 1), (2, 3)])\n    >>> for u, v, p in preds:\n    ...     print(f"({u}, {v}) -> {p}")\n    (0, 1) -> 3.4000000000000004\n    (2, 3) -> 3.4000000000000004\n\n    References\n    ----------\n    .. [1] Ahmad, I., Akhtar, M.U., Noor, S. et al.\n           Missing Link Prediction using Common Neighbor and Centrality based Parameterized Algorithm.\n           Sci Rep 10, 364 (2020).\n           https://doi.org/10.1038/s41598-019-57304-y\n    '
    if alpha == 1:

        def predict(u, v):
            if False:
                print('Hello World!')
            if u == v:
                raise nx.NetworkXAlgorithmError('Self links are not supported')
            return sum((1 for _ in nx.common_neighbors(G, u, v)))
    else:
        spl = dict(nx.shortest_path_length(G))
        inf = float('inf')

        def predict(u, v):
            if False:
                while True:
                    i = 10
            if u == v:
                raise nx.NetworkXAlgorithmError('Self links are not supported')
            path_len = spl[u].get(v, inf)
            return alpha * sum((1 for _ in nx.common_neighbors(G, u, v))) + (1 - alpha) * (G.number_of_nodes() / path_len)
    return _apply_prediction(G, predict, ebunch)

@not_implemented_for('directed')
@not_implemented_for('multigraph')
@nx._dispatch
def preferential_attachment(G, ebunch=None):
    if False:
        while True:
            i = 10
    'Compute the preferential attachment score of all node pairs in ebunch.\n\n    Preferential attachment score of `u` and `v` is defined as\n\n    .. math::\n\n        |\\Gamma(u)| |\\Gamma(v)|\n\n    where $\\Gamma(u)$ denotes the set of neighbors of $u$.\n\n    Parameters\n    ----------\n    G : graph\n        NetworkX undirected graph.\n\n    ebunch : iterable of node pairs, optional (default = None)\n        Preferential attachment score will be computed for each pair of\n        nodes given in the iterable. The pairs must be given as\n        2-tuples (u, v) where u and v are nodes in the graph. If ebunch\n        is None then all nonexistent edges in the graph will be used.\n        Default value: None.\n\n    Returns\n    -------\n    piter : iterator\n        An iterator of 3-tuples in the form (u, v, p) where (u, v) is a\n        pair of nodes and p is their preferential attachment score.\n\n    Examples\n    --------\n    >>> G = nx.complete_graph(5)\n    >>> preds = nx.preferential_attachment(G, [(0, 1), (2, 3)])\n    >>> for u, v, p in preds:\n    ...     print(f"({u}, {v}) -> {p}")\n    (0, 1) -> 16\n    (2, 3) -> 16\n\n    References\n    ----------\n    .. [1] D. Liben-Nowell, J. Kleinberg.\n           The Link Prediction Problem for Social Networks (2004).\n           http://www.cs.cornell.edu/home/kleinber/link-pred.pdf\n    '

    def predict(u, v):
        if False:
            return 10
        return G.degree(u) * G.degree(v)
    return _apply_prediction(G, predict, ebunch)

@not_implemented_for('directed')
@not_implemented_for('multigraph')
@nx._dispatch(node_attrs='community')
def cn_soundarajan_hopcroft(G, ebunch=None, community='community'):
    if False:
        return 10
    'Count the number of common neighbors of all node pairs in ebunch\n        using community information.\n\n    For two nodes $u$ and $v$, this function computes the number of\n    common neighbors and bonus one for each common neighbor belonging to\n    the same community as $u$ and $v$. Mathematically,\n\n    .. math::\n\n        |\\Gamma(u) \\cap \\Gamma(v)| + \\sum_{w \\in \\Gamma(u) \\cap \\Gamma(v)} f(w)\n\n    where $f(w)$ equals 1 if $w$ belongs to the same community as $u$\n    and $v$ or 0 otherwise and $\\Gamma(u)$ denotes the set of\n    neighbors of $u$.\n\n    Parameters\n    ----------\n    G : graph\n        A NetworkX undirected graph.\n\n    ebunch : iterable of node pairs, optional (default = None)\n        The score will be computed for each pair of nodes given in the\n        iterable. The pairs must be given as 2-tuples (u, v) where u\n        and v are nodes in the graph. If ebunch is None then all\n        nonexistent edges in the graph will be used.\n        Default value: None.\n\n    community : string, optional (default = \'community\')\n        Nodes attribute name containing the community information.\n        G[u][community] identifies which community u belongs to. Each\n        node belongs to at most one community. Default value: \'community\'.\n\n    Returns\n    -------\n    piter : iterator\n        An iterator of 3-tuples in the form (u, v, p) where (u, v) is a\n        pair of nodes and p is their score.\n\n    Examples\n    --------\n    >>> G = nx.path_graph(3)\n    >>> G.nodes[0]["community"] = 0\n    >>> G.nodes[1]["community"] = 0\n    >>> G.nodes[2]["community"] = 0\n    >>> preds = nx.cn_soundarajan_hopcroft(G, [(0, 2)])\n    >>> for u, v, p in preds:\n    ...     print(f"({u}, {v}) -> {p}")\n    (0, 2) -> 2\n\n    References\n    ----------\n    .. [1] Sucheta Soundarajan and John Hopcroft.\n       Using community information to improve the precision of link\n       prediction methods.\n       In Proceedings of the 21st international conference companion on\n       World Wide Web (WWW \'12 Companion). ACM, New York, NY, USA, 607-608.\n       http://doi.acm.org/10.1145/2187980.2188150\n    '

    def predict(u, v):
        if False:
            while True:
                i = 10
        Cu = _community(G, u, community)
        Cv = _community(G, v, community)
        cnbors = list(nx.common_neighbors(G, u, v))
        neighbors = sum((_community(G, w, community) == Cu for w in cnbors)) if Cu == Cv else 0
        return len(cnbors) + neighbors
    return _apply_prediction(G, predict, ebunch)

@not_implemented_for('directed')
@not_implemented_for('multigraph')
@nx._dispatch(node_attrs='community')
def ra_index_soundarajan_hopcroft(G, ebunch=None, community='community'):
    if False:
        return 10
    'Compute the resource allocation index of all node pairs in\n    ebunch using community information.\n\n    For two nodes $u$ and $v$, this function computes the resource\n    allocation index considering only common neighbors belonging to the\n    same community as $u$ and $v$. Mathematically,\n\n    .. math::\n\n        \\sum_{w \\in \\Gamma(u) \\cap \\Gamma(v)} \\frac{f(w)}{|\\Gamma(w)|}\n\n    where $f(w)$ equals 1 if $w$ belongs to the same community as $u$\n    and $v$ or 0 otherwise and $\\Gamma(u)$ denotes the set of\n    neighbors of $u$.\n\n    Parameters\n    ----------\n    G : graph\n        A NetworkX undirected graph.\n\n    ebunch : iterable of node pairs, optional (default = None)\n        The score will be computed for each pair of nodes given in the\n        iterable. The pairs must be given as 2-tuples (u, v) where u\n        and v are nodes in the graph. If ebunch is None then all\n        nonexistent edges in the graph will be used.\n        Default value: None.\n\n    community : string, optional (default = \'community\')\n        Nodes attribute name containing the community information.\n        G[u][community] identifies which community u belongs to. Each\n        node belongs to at most one community. Default value: \'community\'.\n\n    Returns\n    -------\n    piter : iterator\n        An iterator of 3-tuples in the form (u, v, p) where (u, v) is a\n        pair of nodes and p is their score.\n\n    Examples\n    --------\n    >>> G = nx.Graph()\n    >>> G.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3)])\n    >>> G.nodes[0]["community"] = 0\n    >>> G.nodes[1]["community"] = 0\n    >>> G.nodes[2]["community"] = 1\n    >>> G.nodes[3]["community"] = 0\n    >>> preds = nx.ra_index_soundarajan_hopcroft(G, [(0, 3)])\n    >>> for u, v, p in preds:\n    ...     print(f"({u}, {v}) -> {p:.8f}")\n    (0, 3) -> 0.50000000\n\n    References\n    ----------\n    .. [1] Sucheta Soundarajan and John Hopcroft.\n       Using community information to improve the precision of link\n       prediction methods.\n       In Proceedings of the 21st international conference companion on\n       World Wide Web (WWW \'12 Companion). ACM, New York, NY, USA, 607-608.\n       http://doi.acm.org/10.1145/2187980.2188150\n    '

    def predict(u, v):
        if False:
            for i in range(10):
                print('nop')
        Cu = _community(G, u, community)
        Cv = _community(G, v, community)
        if Cu != Cv:
            return 0
        cnbors = nx.common_neighbors(G, u, v)
        return sum((1 / G.degree(w) for w in cnbors if _community(G, w, community) == Cu))
    return _apply_prediction(G, predict, ebunch)

@not_implemented_for('directed')
@not_implemented_for('multigraph')
@nx._dispatch(node_attrs='community')
def within_inter_cluster(G, ebunch=None, delta=0.001, community='community'):
    if False:
        for i in range(10):
            print('nop')
    'Compute the ratio of within- and inter-cluster common neighbors\n    of all node pairs in ebunch.\n\n    For two nodes `u` and `v`, if a common neighbor `w` belongs to the\n    same community as them, `w` is considered as within-cluster common\n    neighbor of `u` and `v`. Otherwise, it is considered as\n    inter-cluster common neighbor of `u` and `v`. The ratio between the\n    size of the set of within- and inter-cluster common neighbors is\n    defined as the WIC measure. [1]_\n\n    Parameters\n    ----------\n    G : graph\n        A NetworkX undirected graph.\n\n    ebunch : iterable of node pairs, optional (default = None)\n        The WIC measure will be computed for each pair of nodes given in\n        the iterable. The pairs must be given as 2-tuples (u, v) where\n        u and v are nodes in the graph. If ebunch is None then all\n        nonexistent edges in the graph will be used.\n        Default value: None.\n\n    delta : float, optional (default = 0.001)\n        Value to prevent division by zero in case there is no\n        inter-cluster common neighbor between two nodes. See [1]_ for\n        details. Default value: 0.001.\n\n    community : string, optional (default = \'community\')\n        Nodes attribute name containing the community information.\n        G[u][community] identifies which community u belongs to. Each\n        node belongs to at most one community. Default value: \'community\'.\n\n    Returns\n    -------\n    piter : iterator\n        An iterator of 3-tuples in the form (u, v, p) where (u, v) is a\n        pair of nodes and p is their WIC measure.\n\n    Examples\n    --------\n    >>> G = nx.Graph()\n    >>> G.add_edges_from([(0, 1), (0, 2), (0, 3), (1, 4), (2, 4), (3, 4)])\n    >>> G.nodes[0]["community"] = 0\n    >>> G.nodes[1]["community"] = 1\n    >>> G.nodes[2]["community"] = 0\n    >>> G.nodes[3]["community"] = 0\n    >>> G.nodes[4]["community"] = 0\n    >>> preds = nx.within_inter_cluster(G, [(0, 4)])\n    >>> for u, v, p in preds:\n    ...     print(f"({u}, {v}) -> {p:.8f}")\n    (0, 4) -> 1.99800200\n    >>> preds = nx.within_inter_cluster(G, [(0, 4)], delta=0.5)\n    >>> for u, v, p in preds:\n    ...     print(f"({u}, {v}) -> {p:.8f}")\n    (0, 4) -> 1.33333333\n\n    References\n    ----------\n    .. [1] Jorge Carlos Valverde-Rebaza and Alneu de Andrade Lopes.\n       Link prediction in complex networks based on cluster information.\n       In Proceedings of the 21st Brazilian conference on Advances in\n       Artificial Intelligence (SBIA\'12)\n       https://doi.org/10.1007/978-3-642-34459-6_10\n    '
    if delta <= 0:
        raise nx.NetworkXAlgorithmError('Delta must be greater than zero')

    def predict(u, v):
        if False:
            return 10
        Cu = _community(G, u, community)
        Cv = _community(G, v, community)
        if Cu != Cv:
            return 0
        cnbors = set(nx.common_neighbors(G, u, v))
        within = {w for w in cnbors if _community(G, w, community) == Cu}
        inter = cnbors - within
        return len(within) / (len(inter) + delta)
    return _apply_prediction(G, predict, ebunch)

def _community(G, u, community):
    if False:
        i = 10
        return i + 15
    'Get the community of the given node.'
    node_u = G.nodes[u]
    try:
        return node_u[community]
    except KeyError as err:
        raise nx.NetworkXAlgorithmError('No community information') from err