"""Functions for detecting communities based on modularity."""
from collections import defaultdict
import networkx as nx
from networkx.algorithms.community.quality import modularity
from networkx.utils import not_implemented_for
from networkx.utils.mapped_queue import MappedQueue
__all__ = ['greedy_modularity_communities', 'naive_greedy_modularity_communities']

def _greedy_modularity_communities_generator(G, weight=None, resolution=1):
    if False:
        while True:
            i = 10
    'Yield community partitions of G and the modularity change at each step.\n\n    This function performs Clauset-Newman-Moore greedy modularity maximization [2]_\n    At each step of the process it yields the change in modularity that will occur in\n    the next step followed by yielding the new community partition after that step.\n\n    Greedy modularity maximization begins with each node in its own community\n    and repeatedly joins the pair of communities that lead to the largest\n    modularity until one community contains all nodes (the partition has one set).\n\n    This function maximizes the generalized modularity, where `resolution`\n    is the resolution parameter, often expressed as $\\gamma$.\n    See :func:`~networkx.algorithms.community.quality.modularity`.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n\n    weight : string or None, optional (default=None)\n        The name of an edge attribute that holds the numerical value used\n        as a weight.  If None, then each edge has weight 1.\n        The degree is the sum of the edge weights adjacent to the node.\n\n    resolution : float (default=1)\n        If resolution is less than 1, modularity favors larger communities.\n        Greater than 1 favors smaller communities.\n\n    Yields\n    ------\n    Alternating yield statements produce the following two objects:\n\n    communities: dict_values\n        A dict_values of frozensets of nodes, one for each community.\n        This represents a partition of the nodes of the graph into communities.\n        The first yield is the partition with each node in its own community.\n\n    dq: float\n        The change in modularity when merging the next two communities\n        that leads to the largest modularity.\n\n    See Also\n    --------\n    modularity\n\n    References\n    ----------\n    .. [1] Newman, M. E. J. "Networks: An Introduction", page 224\n       Oxford University Press 2011.\n    .. [2] Clauset, A., Newman, M. E., & Moore, C.\n       "Finding community structure in very large networks."\n       Physical Review E 70(6), 2004.\n    .. [3] Reichardt and Bornholdt "Statistical Mechanics of Community\n       Detection" Phys. Rev. E74, 2006.\n    .. [4] Newman, M. E. J."Analysis of weighted networks"\n       Physical Review E 70(5 Pt 2):056131, 2004.\n    '
    directed = G.is_directed()
    N = G.number_of_nodes()
    m = G.size(weight)
    q0 = 1 / m
    if directed:
        a = {node: deg_out * q0 for (node, deg_out) in G.out_degree(weight=weight)}
        b = {node: deg_in * q0 for (node, deg_in) in G.in_degree(weight=weight)}
    else:
        a = b = {node: deg * q0 * 0.5 for (node, deg) in G.degree(weight=weight)}
    dq_dict = defaultdict(lambda : defaultdict(float))
    for (u, v, wt) in G.edges(data=weight, default=1):
        if u == v:
            continue
        dq_dict[u][v] += wt
        dq_dict[v][u] += wt
    for (u, nbrdict) in dq_dict.items():
        for (v, wt) in nbrdict.items():
            dq_dict[u][v] = q0 * wt - resolution * (a[u] * b[v] + b[u] * a[v])
    dq_heap = {u: MappedQueue({(u, v): -dq for (v, dq) in dq_dict[u].items()}) for u in G}
    H = MappedQueue([dq_heap[n].heap[0] for n in G if len(dq_heap[n]) > 0])
    communities = {n: frozenset([n]) for n in G}
    yield communities.values()
    while len(H) > 1:
        try:
            (negdq, u, v) = H.pop()
        except IndexError:
            break
        dq = -negdq
        yield dq
        dq_heap[u].pop()
        if len(dq_heap[u]) > 0:
            H.push(dq_heap[u].heap[0])
        if dq_heap[v].heap[0] == (v, u):
            H.remove((v, u))
            dq_heap[v].remove((v, u))
            if len(dq_heap[v]) > 0:
                H.push(dq_heap[v].heap[0])
        else:
            dq_heap[v].remove((v, u))
        communities[v] = frozenset(communities[u] | communities[v])
        del communities[u]
        u_nbrs = set(dq_dict[u])
        v_nbrs = set(dq_dict[v])
        all_nbrs = (u_nbrs | v_nbrs) - {u, v}
        both_nbrs = u_nbrs & v_nbrs
        for w in all_nbrs:
            if w in both_nbrs:
                dq_vw = dq_dict[v][w] + dq_dict[u][w]
            elif w in v_nbrs:
                dq_vw = dq_dict[v][w] - resolution * (a[u] * b[w] + a[w] * b[u])
            else:
                dq_vw = dq_dict[u][w] - resolution * (a[v] * b[w] + a[w] * b[v])
            for (row, col) in [(v, w), (w, v)]:
                dq_heap_row = dq_heap[row]
                dq_dict[row][col] = dq_vw
                if len(dq_heap_row) > 0:
                    d_oldmax = dq_heap_row.heap[0]
                else:
                    d_oldmax = None
                d = (row, col)
                d_negdq = -dq_vw
                if w in v_nbrs:
                    dq_heap_row.update(d, d, priority=d_negdq)
                else:
                    dq_heap_row.push(d, priority=d_negdq)
                if d_oldmax is None:
                    H.push(d, priority=d_negdq)
                else:
                    row_max = dq_heap_row.heap[0]
                    if d_oldmax != row_max or d_oldmax.priority != row_max.priority:
                        H.update(d_oldmax, row_max)
        for w in dq_dict[u]:
            dq_old = dq_dict[w][u]
            del dq_dict[w][u]
            if w != v:
                for (row, col) in [(w, u), (u, w)]:
                    dq_heap_row = dq_heap[row]
                    d_old = (row, col)
                    if dq_heap_row.heap[0] == d_old:
                        dq_heap_row.remove(d_old)
                        H.remove(d_old)
                        if len(dq_heap_row) > 0:
                            H.push(dq_heap_row.heap[0])
                    else:
                        dq_heap_row.remove(d_old)
        del dq_dict[u]
        dq_heap[u] = MappedQueue()
        a[v] += a[u]
        a[u] = 0
        if directed:
            b[v] += b[u]
            b[u] = 0
        yield communities.values()

@nx._dispatch(edge_attrs='weight')
def greedy_modularity_communities(G, weight=None, resolution=1, cutoff=1, best_n=None):
    if False:
        for i in range(10):
            print('nop')
    'Find communities in G using greedy modularity maximization.\n\n    This function uses Clauset-Newman-Moore greedy modularity maximization [2]_\n    to find the community partition with the largest modularity.\n\n    Greedy modularity maximization begins with each node in its own community\n    and repeatedly joins the pair of communities that lead to the largest\n    modularity until no further increase in modularity is possible (a maximum).\n    Two keyword arguments adjust the stopping condition. `cutoff` is a lower\n    limit on the number of communities so you can stop the process before\n    reaching a maximum (used to save computation time). `best_n` is an upper\n    limit on the number of communities so you can make the process continue\n    until at most n communities remain even if the maximum modularity occurs\n    for more. To obtain exactly n communities, set both `cutoff` and `best_n` to n.\n\n    This function maximizes the generalized modularity, where `resolution`\n    is the resolution parameter, often expressed as $\\gamma$.\n    See :func:`~networkx.algorithms.community.quality.modularity`.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n\n    weight : string or None, optional (default=None)\n        The name of an edge attribute that holds the numerical value used\n        as a weight.  If None, then each edge has weight 1.\n        The degree is the sum of the edge weights adjacent to the node.\n\n    resolution : float, optional (default=1)\n        If resolution is less than 1, modularity favors larger communities.\n        Greater than 1 favors smaller communities.\n\n    cutoff : int, optional (default=1)\n        A minimum number of communities below which the merging process stops.\n        The process stops at this number of communities even if modularity\n        is not maximized. The goal is to let the user stop the process early.\n        The process stops before the cutoff if it finds a maximum of modularity.\n\n    best_n : int or None, optional (default=None)\n        A maximum number of communities above which the merging process will\n        not stop. This forces community merging to continue after modularity\n        starts to decrease until `best_n` communities remain.\n        If ``None``, don\'t force it to continue beyond a maximum.\n\n    Raises\n    ------\n    ValueError : If the `cutoff` or `best_n`  value is not in the range\n        ``[1, G.number_of_nodes()]``, or if `best_n` < `cutoff`.\n\n    Returns\n    -------\n    communities: list\n        A list of frozensets of nodes, one for each community.\n        Sorted by length with largest communities first.\n\n    Examples\n    --------\n    >>> G = nx.karate_club_graph()\n    >>> c = nx.community.greedy_modularity_communities(G)\n    >>> sorted(c[0])\n    [8, 14, 15, 18, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]\n\n    See Also\n    --------\n    modularity\n\n    References\n    ----------\n    .. [1] Newman, M. E. J. "Networks: An Introduction", page 224\n       Oxford University Press 2011.\n    .. [2] Clauset, A., Newman, M. E., & Moore, C.\n       "Finding community structure in very large networks."\n       Physical Review E 70(6), 2004.\n    .. [3] Reichardt and Bornholdt "Statistical Mechanics of Community\n       Detection" Phys. Rev. E74, 2006.\n    .. [4] Newman, M. E. J."Analysis of weighted networks"\n       Physical Review E 70(5 Pt 2):056131, 2004.\n    '
    if not G.size():
        return [{n} for n in G]
    if cutoff < 1 or cutoff > G.number_of_nodes():
        raise ValueError(f'cutoff must be between 1 and {len(G)}. Got {cutoff}.')
    if best_n is not None:
        if best_n < 1 or best_n > G.number_of_nodes():
            raise ValueError(f'best_n must be between 1 and {len(G)}. Got {best_n}.')
        if best_n < cutoff:
            raise ValueError(f'Must have best_n >= cutoff. Got {best_n} < {cutoff}')
        if best_n == 1:
            return [set(G)]
    else:
        best_n = G.number_of_nodes()
    community_gen = _greedy_modularity_communities_generator(G, weight=weight, resolution=resolution)
    communities = next(community_gen)
    while len(communities) > cutoff:
        try:
            dq = next(community_gen)
        except StopIteration:
            communities = sorted(communities, key=len, reverse=True)
            while len(communities) > best_n:
                (comm1, comm2, *rest) = communities
                communities = [comm1 ^ comm2]
                communities.extend(rest)
            return communities
        if dq < 0 and len(communities) <= best_n:
            break
        communities = next(community_gen)
    return sorted(communities, key=len, reverse=True)

@not_implemented_for('directed')
@not_implemented_for('multigraph')
@nx._dispatch(edge_attrs='weight')
def naive_greedy_modularity_communities(G, resolution=1, weight=None):
    if False:
        while True:
            i = 10
    'Find communities in G using greedy modularity maximization.\n\n    This implementation is O(n^4), much slower than alternatives, but it is\n    provided as an easy-to-understand reference implementation.\n\n    Greedy modularity maximization begins with each node in its own community\n    and joins the pair of communities that most increases modularity until no\n    such pair exists.\n\n    This function maximizes the generalized modularity, where `resolution`\n    is the resolution parameter, often expressed as $\\gamma$.\n    See :func:`~networkx.algorithms.community.quality.modularity`.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n        Graph must be simple and undirected.\n\n    resolution : float (default=1)\n        If resolution is less than 1, modularity favors larger communities.\n        Greater than 1 favors smaller communities.\n\n    weight : string or None, optional (default=None)\n        The name of an edge attribute that holds the numerical value used\n        as a weight.  If None, then each edge has weight 1.\n        The degree is the sum of the edge weights adjacent to the node.\n\n    Returns\n    -------\n    list\n        A list of sets of nodes, one for each community.\n        Sorted by length with largest communities first.\n\n    Examples\n    --------\n    >>> G = nx.karate_club_graph()\n    >>> c = nx.community.naive_greedy_modularity_communities(G)\n    >>> sorted(c[0])\n    [8, 14, 15, 18, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]\n\n    See Also\n    --------\n    greedy_modularity_communities\n    modularity\n    '
    communities = [frozenset([u]) for u in G.nodes()]
    merges = []
    old_modularity = None
    new_modularity = modularity(G, communities, resolution=resolution, weight=weight)
    while old_modularity is None or new_modularity > old_modularity:
        old_modularity = new_modularity
        trial_communities = list(communities)
        to_merge = None
        for (i, u) in enumerate(communities):
            for (j, v) in enumerate(communities):
                if j <= i or len(u) == 0 or len(v) == 0:
                    continue
                trial_communities[j] = u | v
                trial_communities[i] = frozenset([])
                trial_modularity = modularity(G, trial_communities, resolution=resolution, weight=weight)
                if trial_modularity >= new_modularity:
                    if trial_modularity > new_modularity:
                        new_modularity = trial_modularity
                        to_merge = (i, j, new_modularity - old_modularity)
                    elif to_merge and min(i, j) < min(to_merge[0], to_merge[1]):
                        new_modularity = trial_modularity
                        to_merge = (i, j, new_modularity - old_modularity)
                trial_communities[i] = u
                trial_communities[j] = v
        if to_merge is not None:
            merges.append(to_merge)
            (i, j, dq) = to_merge
            (u, v) = (communities[i], communities[j])
            communities[j] = u | v
            communities[i] = frozenset([])
    return sorted((c for c in communities if len(c) > 0), key=len, reverse=True)