"""
Algorithms for finding k-edge-augmentations

A k-edge-augmentation is a set of edges, that once added to a graph, ensures
that the graph is k-edge-connected; i.e. the graph cannot be disconnected
unless k or more edges are removed.  Typically, the goal is to find the
augmentation with minimum weight.  In general, it is not guaranteed that a
k-edge-augmentation exists.

See Also
--------
:mod:`edge_kcomponents` : algorithms for finding k-edge-connected components
:mod:`connectivity` : algorithms for determining edge connectivity.
"""
import itertools as it
import math
from collections import defaultdict, namedtuple
import networkx as nx
from networkx.utils import not_implemented_for, py_random_state
__all__ = ['k_edge_augmentation', 'is_k_edge_connected', 'is_locally_k_edge_connected']

@not_implemented_for('directed')
@not_implemented_for('multigraph')
@nx._dispatch
def is_k_edge_connected(G, k):
    if False:
        i = 10
        return i + 15
    'Tests to see if a graph is k-edge-connected.\n\n    Is it impossible to disconnect the graph by removing fewer than k edges?\n    If so, then G is k-edge-connected.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n       An undirected graph.\n\n    k : integer\n        edge connectivity to test for\n\n    Returns\n    -------\n    boolean\n        True if G is k-edge-connected.\n\n    See Also\n    --------\n    :func:`is_locally_k_edge_connected`\n\n    Examples\n    --------\n    >>> G = nx.barbell_graph(10, 0)\n    >>> nx.is_k_edge_connected(G, k=1)\n    True\n    >>> nx.is_k_edge_connected(G, k=2)\n    False\n    '
    if k < 1:
        raise ValueError(f'k must be positive, not {k}')
    if G.number_of_nodes() < k + 1:
        return False
    elif any((d < k for (n, d) in G.degree())):
        return False
    elif k == 1:
        return nx.is_connected(G)
    elif k == 2:
        return nx.is_connected(G) and (not nx.has_bridges(G))
    else:
        return nx.edge_connectivity(G, cutoff=k) >= k

@not_implemented_for('directed')
@not_implemented_for('multigraph')
@nx._dispatch
def is_locally_k_edge_connected(G, s, t, k):
    if False:
        print('Hello World!')
    'Tests to see if an edge in a graph is locally k-edge-connected.\n\n    Is it impossible to disconnect s and t by removing fewer than k edges?\n    If so, then s and t are locally k-edge-connected in G.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n       An undirected graph.\n\n    s : node\n        Source node\n\n    t : node\n        Target node\n\n    k : integer\n        local edge connectivity for nodes s and t\n\n    Returns\n    -------\n    boolean\n        True if s and t are locally k-edge-connected in G.\n\n    See Also\n    --------\n    :func:`is_k_edge_connected`\n\n    Examples\n    --------\n    >>> from networkx.algorithms.connectivity import is_locally_k_edge_connected\n    >>> G = nx.barbell_graph(10, 0)\n    >>> is_locally_k_edge_connected(G, 5, 15, k=1)\n    True\n    >>> is_locally_k_edge_connected(G, 5, 15, k=2)\n    False\n    >>> is_locally_k_edge_connected(G, 1, 5, k=2)\n    True\n    '
    if k < 1:
        raise ValueError(f'k must be positive, not {k}')
    if G.degree(s) < k or G.degree(t) < k:
        return False
    elif k == 1:
        return nx.has_path(G, s, t)
    else:
        localk = nx.connectivity.local_edge_connectivity(G, s, t, cutoff=k)
        return localk >= k

@not_implemented_for('directed')
@not_implemented_for('multigraph')
@nx._dispatch
def k_edge_augmentation(G, k, avail=None, weight=None, partial=False):
    if False:
        while True:
            i = 10
    'Finds set of edges to k-edge-connect G.\n\n    Adding edges from the augmentation to G make it impossible to disconnect G\n    unless k or more edges are removed. This function uses the most efficient\n    function available (depending on the value of k and if the problem is\n    weighted or unweighted) to search for a minimum weight subset of available\n    edges that k-edge-connects G. In general, finding a k-edge-augmentation is\n    NP-hard, so solutions are not guaranteed to be minimal. Furthermore, a\n    k-edge-augmentation may not exist.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n       An undirected graph.\n\n    k : integer\n        Desired edge connectivity\n\n    avail : dict or a set of 2 or 3 tuples\n        The available edges that can be used in the augmentation.\n\n        If unspecified, then all edges in the complement of G are available.\n        Otherwise, each item is an available edge (with an optional weight).\n\n        In the unweighted case, each item is an edge ``(u, v)``.\n\n        In the weighted case, each item is a 3-tuple ``(u, v, d)`` or a dict\n        with items ``(u, v): d``.  The third item, ``d``, can be a dictionary\n        or a real number.  If ``d`` is a dictionary ``d[weight]``\n        correspondings to the weight.\n\n    weight : string\n        key to use to find weights if ``avail`` is a set of 3-tuples where the\n        third item in each tuple is a dictionary.\n\n    partial : boolean\n        If partial is True and no feasible k-edge-augmentation exists, then all\n        a partial k-edge-augmentation is generated. Adding the edges in a\n        partial augmentation to G, minimizes the number of k-edge-connected\n        components and maximizes the edge connectivity between those\n        components. For details, see :func:`partial_k_edge_augmentation`.\n\n    Yields\n    ------\n    edge : tuple\n        Edges that, once added to G, would cause G to become k-edge-connected.\n        If partial is False, an error is raised if this is not possible.\n        Otherwise, generated edges form a partial augmentation, which\n        k-edge-connects any part of G where it is possible, and maximally\n        connects the remaining parts.\n\n    Raises\n    ------\n    NetworkXUnfeasible\n        If partial is False and no k-edge-augmentation exists.\n\n    NetworkXNotImplemented\n        If the input graph is directed or a multigraph.\n\n    ValueError:\n        If k is less than 1\n\n    Notes\n    -----\n    When k=1 this returns an optimal solution.\n\n    When k=2 and ``avail`` is None, this returns an optimal solution.\n    Otherwise when k=2, this returns a 2-approximation of the optimal solution.\n\n    For k>3, this problem is NP-hard and this uses a randomized algorithm that\n        produces a feasible solution, but provides no guarantees on the\n        solution weight.\n\n    Examples\n    --------\n    >>> # Unweighted cases\n    >>> G = nx.path_graph((1, 2, 3, 4))\n    >>> G.add_node(5)\n    >>> sorted(nx.k_edge_augmentation(G, k=1))\n    [(1, 5)]\n    >>> sorted(nx.k_edge_augmentation(G, k=2))\n    [(1, 5), (5, 4)]\n    >>> sorted(nx.k_edge_augmentation(G, k=3))\n    [(1, 4), (1, 5), (2, 5), (3, 5), (4, 5)]\n    >>> complement = list(nx.k_edge_augmentation(G, k=5, partial=True))\n    >>> G.add_edges_from(complement)\n    >>> nx.edge_connectivity(G)\n    4\n\n    >>> # Weighted cases\n    >>> G = nx.path_graph((1, 2, 3, 4))\n    >>> G.add_node(5)\n    >>> # avail can be a tuple with a dict\n    >>> avail = [(1, 5, {"weight": 11}), (2, 5, {"weight": 10})]\n    >>> sorted(nx.k_edge_augmentation(G, k=1, avail=avail, weight="weight"))\n    [(2, 5)]\n    >>> # or avail can be a 3-tuple with a real number\n    >>> avail = [(1, 5, 11), (2, 5, 10), (4, 3, 1), (4, 5, 51)]\n    >>> sorted(nx.k_edge_augmentation(G, k=2, avail=avail))\n    [(1, 5), (2, 5), (4, 5)]\n    >>> # or avail can be a dict\n    >>> avail = {(1, 5): 11, (2, 5): 10, (4, 3): 1, (4, 5): 51}\n    >>> sorted(nx.k_edge_augmentation(G, k=2, avail=avail))\n    [(1, 5), (2, 5), (4, 5)]\n    >>> # If augmentation is infeasible, then a partial solution can be found\n    >>> avail = {(1, 5): 11}\n    >>> sorted(nx.k_edge_augmentation(G, k=2, avail=avail, partial=True))\n    [(1, 5)]\n    '
    try:
        if k <= 0:
            raise ValueError(f'k must be a positive integer, not {k}')
        elif G.number_of_nodes() < k + 1:
            msg = f'impossible to {k} connect in graph with less than {k + 1} nodes'
            raise nx.NetworkXUnfeasible(msg)
        elif avail is not None and len(avail) == 0:
            if not nx.is_k_edge_connected(G, k):
                raise nx.NetworkXUnfeasible('no available edges')
            aug_edges = []
        elif k == 1:
            aug_edges = one_edge_augmentation(G, avail=avail, weight=weight, partial=partial)
        elif k == 2:
            aug_edges = bridge_augmentation(G, avail=avail, weight=weight)
        else:
            aug_edges = greedy_k_edge_augmentation(G, k=k, avail=avail, weight=weight, seed=0)
        yield from list(aug_edges)
    except nx.NetworkXUnfeasible:
        if partial:
            if avail is None:
                aug_edges = complement_edges(G)
            else:
                aug_edges = partial_k_edge_augmentation(G, k=k, avail=avail, weight=weight)
            yield from aug_edges
        else:
            raise

@nx._dispatch
def partial_k_edge_augmentation(G, k, avail, weight=None):
    if False:
        for i in range(10):
            print('nop')
    'Finds augmentation that k-edge-connects as much of the graph as possible.\n\n    When a k-edge-augmentation is not possible, we can still try to find a\n    small set of edges that partially k-edge-connects as much of the graph as\n    possible. All possible edges are generated between remaining parts.\n    This minimizes the number of k-edge-connected subgraphs in the resulting\n    graph and maximizes the edge connectivity between those subgraphs.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n       An undirected graph.\n\n    k : integer\n        Desired edge connectivity\n\n    avail : dict or a set of 2 or 3 tuples\n        For more details, see :func:`k_edge_augmentation`.\n\n    weight : string\n        key to use to find weights if ``avail`` is a set of 3-tuples.\n        For more details, see :func:`k_edge_augmentation`.\n\n    Yields\n    ------\n    edge : tuple\n        Edges in the partial augmentation of G. These edges k-edge-connect any\n        part of G where it is possible, and maximally connects the remaining\n        parts. In other words, all edges from avail are generated except for\n        those within subgraphs that have already become k-edge-connected.\n\n    Notes\n    -----\n    Construct H that augments G with all edges in avail.\n    Find the k-edge-subgraphs of H.\n    For each k-edge-subgraph, if the number of nodes is more than k, then find\n    the k-edge-augmentation of that graph and add it to the solution. Then add\n    all edges in avail between k-edge subgraphs to the solution.\n\n    See Also\n    --------\n    :func:`k_edge_augmentation`\n\n    Examples\n    --------\n    >>> G = nx.path_graph((1, 2, 3, 4, 5, 6, 7))\n    >>> G.add_node(8)\n    >>> avail = [(1, 3), (1, 4), (1, 5), (2, 4), (2, 5), (3, 5), (1, 8)]\n    >>> sorted(partial_k_edge_augmentation(G, k=2, avail=avail))\n    [(1, 5), (1, 8)]\n    '

    def _edges_between_disjoint(H, only1, only2):
        if False:
            print('Hello World!')
        'finds edges between disjoint nodes'
        only1_adj = {u: set(H.adj[u]) for u in only1}
        for (u, neighbs) in only1_adj.items():
            neighbs12 = neighbs.intersection(only2)
            for v in neighbs12:
                yield (u, v)
    (avail_uv, avail_w) = _unpack_available_edges(avail, weight=weight, G=G)
    H = G.copy()
    H.add_edges_from(((u, v, {'weight': w, 'generator': (u, v)}) for ((u, v), w) in zip(avail, avail_w)))
    k_edge_subgraphs = list(nx.k_edge_subgraphs(H, k=k))
    for nodes in k_edge_subgraphs:
        if len(nodes) > 1:
            C = H.subgraph(nodes).copy()
            sub_avail = {d['generator']: d['weight'] for (u, v, d) in C.edges(data=True) if 'generator' in d}
            C.remove_edges_from(sub_avail.keys())
            yield from nx.k_edge_augmentation(C, k=k, avail=sub_avail)
    for (cc1, cc2) in it.combinations(k_edge_subgraphs, 2):
        for (u, v) in _edges_between_disjoint(H, cc1, cc2):
            d = H.get_edge_data(u, v)
            edge = d.get('generator', None)
            if edge is not None:
                yield edge

@not_implemented_for('multigraph')
@not_implemented_for('directed')
@nx._dispatch
def one_edge_augmentation(G, avail=None, weight=None, partial=False):
    if False:
        for i in range(10):
            print('nop')
    'Finds minimum weight set of edges to connect G.\n\n    Equivalent to :func:`k_edge_augmentation` when k=1. Adding the resulting\n    edges to G will make it 1-edge-connected. The solution is optimal for both\n    weighted and non-weighted variants.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n       An undirected graph.\n\n    avail : dict or a set of 2 or 3 tuples\n        For more details, see :func:`k_edge_augmentation`.\n\n    weight : string\n        key to use to find weights if ``avail`` is a set of 3-tuples.\n        For more details, see :func:`k_edge_augmentation`.\n\n    partial : boolean\n        If partial is True and no feasible k-edge-augmentation exists, then the\n        augmenting edges minimize the number of connected components.\n\n    Yields\n    ------\n    edge : tuple\n        Edges in the one-augmentation of G\n\n    Raises\n    ------\n    NetworkXUnfeasible\n        If partial is False and no one-edge-augmentation exists.\n\n    Notes\n    -----\n    Uses either :func:`unconstrained_one_edge_augmentation` or\n    :func:`weighted_one_edge_augmentation` depending on whether ``avail`` is\n    specified. Both algorithms are based on finding a minimum spanning tree.\n    As such both algorithms find optimal solutions and run in linear time.\n\n    See Also\n    --------\n    :func:`k_edge_augmentation`\n    '
    if avail is None:
        return unconstrained_one_edge_augmentation(G)
    else:
        return weighted_one_edge_augmentation(G, avail=avail, weight=weight, partial=partial)

@not_implemented_for('multigraph')
@not_implemented_for('directed')
@nx._dispatch
def bridge_augmentation(G, avail=None, weight=None):
    if False:
        i = 10
        return i + 15
    'Finds the a set of edges that bridge connects G.\n\n    Equivalent to :func:`k_edge_augmentation` when k=2, and partial=False.\n    Adding the resulting edges to G will make it 2-edge-connected.  If no\n    constraints are specified the returned set of edges is minimum an optimal,\n    otherwise the solution is approximated.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n       An undirected graph.\n\n    avail : dict or a set of 2 or 3 tuples\n        For more details, see :func:`k_edge_augmentation`.\n\n    weight : string\n        key to use to find weights if ``avail`` is a set of 3-tuples.\n        For more details, see :func:`k_edge_augmentation`.\n\n    Yields\n    ------\n    edge : tuple\n        Edges in the bridge-augmentation of G\n\n    Raises\n    ------\n    NetworkXUnfeasible\n        If no bridge-augmentation exists.\n\n    Notes\n    -----\n    If there are no constraints the solution can be computed in linear time\n    using :func:`unconstrained_bridge_augmentation`. Otherwise, the problem\n    becomes NP-hard and is the solution is approximated by\n    :func:`weighted_bridge_augmentation`.\n\n    See Also\n    --------\n    :func:`k_edge_augmentation`\n    '
    if G.number_of_nodes() < 3:
        raise nx.NetworkXUnfeasible('impossible to bridge connect less than 3 nodes')
    if avail is None:
        return unconstrained_bridge_augmentation(G)
    else:
        return weighted_bridge_augmentation(G, avail, weight=weight)

def _ordered(u, v):
    if False:
        i = 10
        return i + 15
    'Returns the nodes in an undirected edge in lower-triangular order'
    return (u, v) if u < v else (v, u)

def _unpack_available_edges(avail, weight=None, G=None):
    if False:
        i = 10
        return i + 15
    'Helper to separate avail into edges and corresponding weights'
    if weight is None:
        weight = 'weight'
    if isinstance(avail, dict):
        avail_uv = list(avail.keys())
        avail_w = list(avail.values())
    else:

        def _try_getitem(d):
            if False:
                while True:
                    i = 10
            try:
                return d[weight]
            except TypeError:
                return d
        avail_uv = [tup[0:2] for tup in avail]
        avail_w = [1 if len(tup) == 2 else _try_getitem(tup[-1]) for tup in avail]
    if G is not None:
        flags = [not G.has_edge(u, v) for (u, v) in avail_uv]
        avail_uv = list(it.compress(avail_uv, flags))
        avail_w = list(it.compress(avail_w, flags))
    return (avail_uv, avail_w)
MetaEdge = namedtuple('MetaEdge', ('meta_uv', 'uv', 'w'))

def _lightest_meta_edges(mapping, avail_uv, avail_w):
    if False:
        while True:
            i = 10
    "Maps available edges in the original graph to edges in the metagraph.\n\n    Parameters\n    ----------\n    mapping : dict\n        mapping produced by :func:`collapse`, that maps each node in the\n        original graph to a node in the meta graph\n\n    avail_uv : list\n        list of edges\n\n    avail_w : list\n        list of edge weights\n\n    Notes\n    -----\n    Each node in the metagraph is a k-edge-connected component in the original\n    graph.  We don't care about any edge within the same k-edge-connected\n    component, so we ignore self edges.  We also are only interested in the\n    minimum weight edge bridging each k-edge-connected component so, we group\n    the edges by meta-edge and take the lightest in each group.\n\n    Examples\n    --------\n    >>> # Each group represents a meta-node\n    >>> groups = ([1, 2, 3], [4, 5], [6])\n    >>> mapping = {n: meta_n for meta_n, ns in enumerate(groups) for n in ns}\n    >>> avail_uv = [(1, 2), (3, 6), (1, 4), (5, 2), (6, 1), (2, 6), (3, 1)]\n    >>> avail_w = [20, 99, 20, 15, 50, 99, 20]\n    >>> sorted(_lightest_meta_edges(mapping, avail_uv, avail_w))\n    [MetaEdge(meta_uv=(0, 1), uv=(5, 2), w=15), MetaEdge(meta_uv=(0, 2), uv=(6, 1), w=50)]\n    "
    grouped_wuv = defaultdict(list)
    for (w, (u, v)) in zip(avail_w, avail_uv):
        meta_uv = _ordered(mapping[u], mapping[v])
        grouped_wuv[meta_uv].append((w, u, v))
    for ((mu, mv), choices_wuv) in grouped_wuv.items():
        if mu != mv:
            (w, u, v) = min(choices_wuv)
            yield MetaEdge((mu, mv), (u, v), w)

@nx._dispatch
def unconstrained_one_edge_augmentation(G):
    if False:
        while True:
            i = 10
    'Finds the smallest set of edges to connect G.\n\n    This is a variant of the unweighted MST problem.\n    If G is not empty, a feasible solution always exists.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n       An undirected graph.\n\n    Yields\n    ------\n    edge : tuple\n        Edges in the one-edge-augmentation of G\n\n    See Also\n    --------\n    :func:`one_edge_augmentation`\n    :func:`k_edge_augmentation`\n\n    Examples\n    --------\n    >>> G = nx.Graph([(1, 2), (2, 3), (4, 5)])\n    >>> G.add_nodes_from([6, 7, 8])\n    >>> sorted(unconstrained_one_edge_augmentation(G))\n    [(1, 4), (4, 6), (6, 7), (7, 8)]\n    '
    ccs1 = list(nx.connected_components(G))
    C = collapse(G, ccs1)
    meta_nodes = list(C.nodes())
    meta_aug = list(zip(meta_nodes, meta_nodes[1:]))
    inverse = defaultdict(list)
    for (k, v) in C.graph['mapping'].items():
        inverse[v].append(k)
    for (mu, mv) in meta_aug:
        yield (inverse[mu][0], inverse[mv][0])

@nx._dispatch
def weighted_one_edge_augmentation(G, avail, weight=None, partial=False):
    if False:
        for i in range(10):
            print('nop')
    'Finds the minimum weight set of edges to connect G if one exists.\n\n    This is a variant of the weighted MST problem.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n       An undirected graph.\n\n    avail : dict or a set of 2 or 3 tuples\n        For more details, see :func:`k_edge_augmentation`.\n\n    weight : string\n        key to use to find weights if ``avail`` is a set of 3-tuples.\n        For more details, see :func:`k_edge_augmentation`.\n\n    partial : boolean\n        If partial is True and no feasible k-edge-augmentation exists, then the\n        augmenting edges minimize the number of connected components.\n\n    Yields\n    ------\n    edge : tuple\n        Edges in the subset of avail chosen to connect G.\n\n    See Also\n    --------\n    :func:`one_edge_augmentation`\n    :func:`k_edge_augmentation`\n\n    Examples\n    --------\n    >>> G = nx.Graph([(1, 2), (2, 3), (4, 5)])\n    >>> G.add_nodes_from([6, 7, 8])\n    >>> # any edge not in avail has an implicit weight of infinity\n    >>> avail = [(1, 3), (1, 5), (4, 7), (4, 8), (6, 1), (8, 1), (8, 2)]\n    >>> sorted(weighted_one_edge_augmentation(G, avail))\n    [(1, 5), (4, 7), (6, 1), (8, 1)]\n    >>> # find another solution by giving large weights to edges in the\n    >>> # previous solution (note some of the old edges must be used)\n    >>> avail = [(1, 3), (1, 5, 99), (4, 7, 9), (6, 1, 99), (8, 1, 99), (8, 2)]\n    >>> sorted(weighted_one_edge_augmentation(G, avail))\n    [(1, 5), (4, 7), (6, 1), (8, 2)]\n    '
    (avail_uv, avail_w) = _unpack_available_edges(avail, weight=weight, G=G)
    C = collapse(G, nx.connected_components(G))
    mapping = C.graph['mapping']
    candidate_mapping = _lightest_meta_edges(mapping, avail_uv, avail_w)
    C.add_edges_from(((mu, mv, {'weight': w, 'generator': uv}) for ((mu, mv), uv, w) in candidate_mapping))
    meta_mst = nx.minimum_spanning_tree(C)
    if not partial and (not nx.is_connected(meta_mst)):
        raise nx.NetworkXUnfeasible('Not possible to connect G with available edges')
    for (mu, mv, d) in meta_mst.edges(data=True):
        if 'generator' in d:
            edge = d['generator']
            yield edge

@nx._dispatch
def unconstrained_bridge_augmentation(G):
    if False:
        while True:
            i = 10
    'Finds an optimal 2-edge-augmentation of G using the fewest edges.\n\n    This is an implementation of the algorithm detailed in [1]_.\n    The basic idea is to construct a meta-graph of bridge-ccs, connect leaf\n    nodes of the trees to connect the entire graph, and finally connect the\n    leafs of the tree in dfs-preorder to bridge connect the entire graph.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n       An undirected graph.\n\n    Yields\n    ------\n    edge : tuple\n        Edges in the bridge augmentation of G\n\n    Notes\n    -----\n    Input: a graph G.\n    First find the bridge components of G and collapse each bridge-cc into a\n    node of a metagraph graph C, which is guaranteed to be a forest of trees.\n\n    C contains p "leafs" --- nodes with exactly one incident edge.\n    C contains q "isolated nodes" --- nodes with no incident edges.\n\n    Theorem: If p + q > 1, then at least :math:`ceil(p / 2) + q` edges are\n        needed to bridge connect C. This algorithm achieves this min number.\n\n    The method first adds enough edges to make G into a tree and then pairs\n    leafs in a simple fashion.\n\n    Let n be the number of trees in C. Let v(i) be an isolated vertex in the\n    i-th tree if one exists, otherwise it is a pair of distinct leafs nodes\n    in the i-th tree. Alternating edges from these sets (i.e.  adding edges\n    A1 = [(v(i)[0], v(i + 1)[1]), v(i + 1)[0], v(i + 2)[1])...]) connects C\n    into a tree T. This tree has p\' = p + 2q - 2(n -1) leafs and no isolated\n    vertices. A1 has n - 1 edges. The next step finds ceil(p\' / 2) edges to\n    biconnect any tree with p\' leafs.\n\n    Convert T into an arborescence T\' by picking an arbitrary root node with\n    degree >= 2 and directing all edges away from the root. Note the\n    implementation implicitly constructs T\'.\n\n    The leafs of T are the nodes with no existing edges in T\'.\n    Order the leafs of T\' by DFS preorder. Then break this list in half\n    and add the zipped pairs to A2.\n\n    The set A = A1 + A2 is the minimum augmentation in the metagraph.\n\n    To convert this to edges in the original graph\n\n    References\n    ----------\n    .. [1] Eswaran, Kapali P., and R. Endre Tarjan. (1975) Augmentation problems.\n        http://epubs.siam.org/doi/abs/10.1137/0205044\n\n    See Also\n    --------\n    :func:`bridge_augmentation`\n    :func:`k_edge_augmentation`\n\n    Examples\n    --------\n    >>> G = nx.path_graph((1, 2, 3, 4, 5, 6, 7))\n    >>> sorted(unconstrained_bridge_augmentation(G))\n    [(1, 7)]\n    >>> G = nx.path_graph((1, 2, 3, 2, 4, 5, 6, 7))\n    >>> sorted(unconstrained_bridge_augmentation(G))\n    [(1, 3), (3, 7)]\n    >>> G = nx.Graph([(0, 1), (0, 2), (1, 2)])\n    >>> G.add_node(4)\n    >>> sorted(unconstrained_bridge_augmentation(G))\n    [(1, 4), (4, 0)]\n    '
    bridge_ccs = list(nx.connectivity.bridge_components(G))
    C = collapse(G, bridge_ccs)
    vset1 = [tuple(cc) * 2 if len(cc) == 1 else sorted(cc, key=C.degree)[0:2] for cc in nx.connected_components(C)]
    if len(vset1) > 1:
        nodes1 = [vs[0] for vs in vset1]
        nodes2 = [vs[1] for vs in vset1]
        A1 = list(zip(nodes1[1:], nodes2))
    else:
        A1 = []
    T = C.copy()
    T.add_edges_from(A1)
    leafs = [n for (n, d) in T.degree() if d == 1]
    if len(leafs) == 1:
        A2 = []
    if len(leafs) == 2:
        A2 = [tuple(leafs)]
    else:
        try:
            root = next((n for (n, d) in T.degree() if d > 1))
        except StopIteration:
            return
        v2 = [n for n in nx.dfs_preorder_nodes(T, root) if T.degree(n) == 1]
        half = math.ceil(len(v2) / 2)
        A2 = list(zip(v2[:half], v2[-half:]))
    aug_tree_edges = A1 + A2
    inverse = defaultdict(list)
    for (k, v) in C.graph['mapping'].items():
        inverse[v].append(k)
    inverse = {mu: sorted(mapped, key=lambda u: (G.degree(u), u)) for (mu, mapped) in inverse.items()}
    G2 = G.copy()
    for (mu, mv) in aug_tree_edges:
        for (u, v) in it.product(inverse[mu], inverse[mv]):
            if not G2.has_edge(u, v):
                G2.add_edge(u, v)
                yield (u, v)
                break

@nx._dispatch
def weighted_bridge_augmentation(G, avail, weight=None):
    if False:
        i = 10
        return i + 15
    'Finds an approximate min-weight 2-edge-augmentation of G.\n\n    This is an implementation of the approximation algorithm detailed in [1]_.\n    It chooses a set of edges from avail to add to G that renders it\n    2-edge-connected if such a subset exists.  This is done by finding a\n    minimum spanning arborescence of a specially constructed metagraph.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n       An undirected graph.\n\n    avail : set of 2 or 3 tuples.\n        candidate edges (with optional weights) to choose from\n\n    weight : string\n        key to use to find weights if avail is a set of 3-tuples where the\n        third item in each tuple is a dictionary.\n\n    Yields\n    ------\n    edge : tuple\n        Edges in the subset of avail chosen to bridge augment G.\n\n    Notes\n    -----\n    Finding a weighted 2-edge-augmentation is NP-hard.\n    Any edge not in ``avail`` is considered to have a weight of infinity.\n    The approximation factor is 2 if ``G`` is connected and 3 if it is not.\n    Runs in :math:`O(m + n log(n))` time\n\n    References\n    ----------\n    .. [1] Khuller, Samir, and Ramakrishna Thurimella. (1993) Approximation\n        algorithms for graph augmentation.\n        http://www.sciencedirect.com/science/article/pii/S0196677483710102\n\n    See Also\n    --------\n    :func:`bridge_augmentation`\n    :func:`k_edge_augmentation`\n\n    Examples\n    --------\n    >>> G = nx.path_graph((1, 2, 3, 4))\n    >>> # When the weights are equal, (1, 4) is the best\n    >>> avail = [(1, 4, 1), (1, 3, 1), (2, 4, 1)]\n    >>> sorted(weighted_bridge_augmentation(G, avail))\n    [(1, 4)]\n    >>> # Giving (1, 4) a high weight makes the two edge solution the best.\n    >>> avail = [(1, 4, 1000), (1, 3, 1), (2, 4, 1)]\n    >>> sorted(weighted_bridge_augmentation(G, avail))\n    [(1, 3), (2, 4)]\n    >>> # ------\n    >>> G = nx.path_graph((1, 2, 3, 4))\n    >>> G.add_node(5)\n    >>> avail = [(1, 5, 11), (2, 5, 10), (4, 3, 1), (4, 5, 1)]\n    >>> sorted(weighted_bridge_augmentation(G, avail=avail))\n    [(1, 5), (4, 5)]\n    >>> avail = [(1, 5, 11), (2, 5, 10), (4, 3, 1), (4, 5, 51)]\n    >>> sorted(weighted_bridge_augmentation(G, avail=avail))\n    [(1, 5), (2, 5), (4, 5)]\n    '
    if weight is None:
        weight = 'weight'
    if not nx.is_connected(G):
        H = G.copy()
        connectors = list(one_edge_augmentation(H, avail=avail, weight=weight))
        H.add_edges_from(connectors)
        yield from connectors
    else:
        connectors = []
        H = G
    if len(avail) == 0:
        if nx.has_bridges(H):
            raise nx.NetworkXUnfeasible('no augmentation possible')
    (avail_uv, avail_w) = _unpack_available_edges(avail, weight=weight, G=H)
    bridge_ccs = nx.connectivity.bridge_components(H)
    C = collapse(H, bridge_ccs)
    mapping = C.graph['mapping']
    meta_to_wuv = {(mu, mv): (w, uv) for ((mu, mv), uv, w) in _lightest_meta_edges(mapping, avail_uv, avail_w)}
    try:
        root = next((n for (n, d) in C.degree() if d == 1))
    except StopIteration:
        return
    TR = nx.dfs_tree(C, root)
    D = nx.reverse(TR).copy()
    nx.set_edge_attributes(D, name='weight', values=0)
    lca_gen = nx.tree_all_pairs_lowest_common_ancestor(TR, root=root, pairs=meta_to_wuv.keys())
    for ((mu, mv), lca) in lca_gen:
        (w, uv) = meta_to_wuv[mu, mv]
        if lca == mu:
            D.add_edge(lca, mv, weight=w, generator=uv)
        elif lca == mv:
            D.add_edge(lca, mu, weight=w, generator=uv)
        else:
            D.add_edge(lca, mu, weight=w, generator=uv)
            D.add_edge(lca, mv, weight=w, generator=uv)
    try:
        A = _minimum_rooted_branching(D, root)
    except nx.NetworkXException as err:
        raise nx.NetworkXUnfeasible('no 2-edge-augmentation possible') from err
    bridge_connectors = set()
    for (mu, mv) in A.edges():
        data = D.get_edge_data(mu, mv)
        if 'generator' in data:
            edge = data['generator']
            bridge_connectors.add(edge)
    yield from bridge_connectors

def _minimum_rooted_branching(D, root):
    if False:
        for i in range(10):
            print('nop')
    'Helper function to compute a minimum rooted branching (aka rooted\n    arborescence)\n\n    Before the branching can be computed, the directed graph must be rooted by\n    removing the predecessors of root.\n\n    A branching / arborescence of rooted graph G is a subgraph that contains a\n    directed path from the root to every other vertex. It is the directed\n    analog of the minimum spanning tree problem.\n\n    References\n    ----------\n    [1] Khuller, Samir (2002) Advanced Algorithms Lecture 24 Notes.\n    https://web.archive.org/web/20121030033722/https://www.cs.umd.edu/class/spring2011/cmsc651/lec07.pdf\n    '
    rooted = D.copy()
    rooted.remove_edges_from([(u, root) for u in D.predecessors(root)])
    A = nx.minimum_spanning_arborescence(rooted)
    return A

@nx._dispatch
def collapse(G, grouped_nodes):
    if False:
        print('Hello World!')
    'Collapses each group of nodes into a single node.\n\n    This is similar to condensation, but works on undirected graphs.\n\n    Parameters\n    ----------\n    G : NetworkX Graph\n\n    grouped_nodes:  list or generator\n       Grouping of nodes to collapse. The grouping must be disjoint.\n       If grouped_nodes are strongly_connected_components then this is\n       equivalent to :func:`condensation`.\n\n    Returns\n    -------\n    C : NetworkX Graph\n       The collapsed graph C of G with respect to the node grouping.  The node\n       labels are integers corresponding to the index of the component in the\n       list of grouped_nodes.  C has a graph attribute named \'mapping\' with a\n       dictionary mapping the original nodes to the nodes in C to which they\n       belong.  Each node in C also has a node attribute \'members\' with the set\n       of original nodes in G that form the group that the node in C\n       represents.\n\n    Examples\n    --------\n    >>> # Collapses a graph using disjoint groups, but not necessarily connected\n    >>> G = nx.Graph([(1, 0), (2, 3), (3, 1), (3, 4), (4, 5), (5, 6), (5, 7)])\n    >>> G.add_node("A")\n    >>> grouped_nodes = [{0, 1, 2, 3}, {5, 6, 7}]\n    >>> C = collapse(G, grouped_nodes)\n    >>> members = nx.get_node_attributes(C, "members")\n    >>> sorted(members.keys())\n    [0, 1, 2, 3]\n    >>> member_values = set(map(frozenset, members.values()))\n    >>> assert {0, 1, 2, 3} in member_values\n    >>> assert {4} in member_values\n    >>> assert {5, 6, 7} in member_values\n    >>> assert {"A"} in member_values\n    '
    mapping = {}
    members = {}
    C = G.__class__()
    i = 0
    remaining = set(G.nodes())
    for (i, group) in enumerate(grouped_nodes):
        group = set(group)
        assert remaining.issuperset(group), 'grouped nodes must exist in G and be disjoint'
        remaining.difference_update(group)
        members[i] = group
        mapping.update(((n, i) for n in group))
    for (i, node) in enumerate(remaining, start=i + 1):
        group = {node}
        members[i] = group
        mapping.update(((n, i) for n in group))
    number_of_groups = i + 1
    C.add_nodes_from(range(number_of_groups))
    C.add_edges_from(((mapping[u], mapping[v]) for (u, v) in G.edges() if mapping[u] != mapping[v]))
    nx.set_node_attributes(C, name='members', values=members)
    C.graph['mapping'] = mapping
    return C

@nx._dispatch
def complement_edges(G):
    if False:
        return 10
    'Returns only the edges in the complement of G\n\n    Parameters\n    ----------\n    G : NetworkX Graph\n\n    Yields\n    ------\n    edge : tuple\n        Edges in the complement of G\n\n    Examples\n    --------\n    >>> G = nx.path_graph((1, 2, 3, 4))\n    >>> sorted(complement_edges(G))\n    [(1, 3), (1, 4), (2, 4)]\n    >>> G = nx.path_graph((1, 2, 3, 4), nx.DiGraph())\n    >>> sorted(complement_edges(G))\n    [(1, 3), (1, 4), (2, 1), (2, 4), (3, 1), (3, 2), (4, 1), (4, 2), (4, 3)]\n    >>> G = nx.complete_graph(1000)\n    >>> sorted(complement_edges(G))\n    []\n    '
    G_adj = G._adj
    if G.is_directed():
        for (u, v) in it.combinations(G.nodes(), 2):
            if v not in G_adj[u]:
                yield (u, v)
            if u not in G_adj[v]:
                yield (v, u)
    else:
        for (u, v) in it.combinations(G.nodes(), 2):
            if v not in G_adj[u]:
                yield (u, v)

def _compat_shuffle(rng, input):
    if False:
        i = 10
        return i + 15
    'wrapper around rng.shuffle for python 2 compatibility reasons'
    rng.shuffle(input)

@not_implemented_for('multigraph')
@not_implemented_for('directed')
@py_random_state(4)
@nx._dispatch
def greedy_k_edge_augmentation(G, k, avail=None, weight=None, seed=None):
    if False:
        for i in range(10):
            print('nop')
    'Greedy algorithm for finding a k-edge-augmentation\n\n    Parameters\n    ----------\n    G : NetworkX graph\n       An undirected graph.\n\n    k : integer\n        Desired edge connectivity\n\n    avail : dict or a set of 2 or 3 tuples\n        For more details, see :func:`k_edge_augmentation`.\n\n    weight : string\n        key to use to find weights if ``avail`` is a set of 3-tuples.\n        For more details, see :func:`k_edge_augmentation`.\n\n    seed : integer, random_state, or None (default)\n        Indicator of random number generation state.\n        See :ref:`Randomness<randomness>`.\n\n    Yields\n    ------\n    edge : tuple\n        Edges in the greedy augmentation of G\n\n    Notes\n    -----\n    The algorithm is simple. Edges are incrementally added between parts of the\n    graph that are not yet locally k-edge-connected. Then edges are from the\n    augmenting set are pruned as long as local-edge-connectivity is not broken.\n\n    This algorithm is greedy and does not provide optimality guarantees. It\n    exists only to provide :func:`k_edge_augmentation` with the ability to\n    generate a feasible solution for arbitrary k.\n\n    See Also\n    --------\n    :func:`k_edge_augmentation`\n\n    Examples\n    --------\n    >>> G = nx.path_graph((1, 2, 3, 4, 5, 6, 7))\n    >>> sorted(greedy_k_edge_augmentation(G, k=2))\n    [(1, 7)]\n    >>> sorted(greedy_k_edge_augmentation(G, k=1, avail=[]))\n    []\n    >>> G = nx.path_graph((1, 2, 3, 4, 5, 6, 7))\n    >>> avail = {(u, v): 1 for (u, v) in complement_edges(G)}\n    >>> # randomized pruning process can produce different solutions\n    >>> sorted(greedy_k_edge_augmentation(G, k=4, avail=avail, seed=2))\n    [(1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (2, 4), (2, 6), (3, 7), (5, 7)]\n    >>> sorted(greedy_k_edge_augmentation(G, k=4, avail=avail, seed=3))\n    [(1, 3), (1, 5), (1, 6), (2, 4), (2, 6), (3, 7), (4, 7), (5, 7)]\n    '
    aug_edges = []
    done = is_k_edge_connected(G, k)
    if done:
        return
    if avail is None:
        avail_uv = list(complement_edges(G))
        avail_w = [1] * len(avail_uv)
    else:
        (avail_uv, avail_w) = _unpack_available_edges(avail, weight=weight, G=G)
    tiebreaker = [sum(map(G.degree, uv)) for uv in avail_uv]
    avail_wduv = sorted(zip(avail_w, tiebreaker, avail_uv))
    avail_uv = [uv for (w, d, uv) in avail_wduv]
    H = G.copy()
    for (u, v) in avail_uv:
        done = False
        if not is_locally_k_edge_connected(H, u, v, k=k):
            aug_edges.append((u, v))
            H.add_edge(u, v)
            if H.degree(u) >= k and H.degree(v) >= k:
                done = is_k_edge_connected(H, k)
        if done:
            break
    if not done:
        raise nx.NetworkXUnfeasible('not able to k-edge-connect with available edges')
    _compat_shuffle(seed, aug_edges)
    for (u, v) in list(aug_edges):
        if H.degree(u) <= k or H.degree(v) <= k:
            continue
        H.remove_edge(u, v)
        aug_edges.remove((u, v))
        if not is_k_edge_connected(H, k=k):
            H.add_edge(u, v)
            aug_edges.append((u, v))
    yield from aug_edges