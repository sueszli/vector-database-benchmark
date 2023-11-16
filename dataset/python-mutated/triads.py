"""Functions for analyzing triads of a graph."""
from collections import defaultdict
from itertools import combinations, permutations
import networkx as nx
from networkx.utils import not_implemented_for, py_random_state
__all__ = ['triadic_census', 'is_triad', 'all_triplets', 'all_triads', 'triads_by_type', 'triad_type', 'random_triad']
TRICODES = (1, 2, 2, 3, 2, 4, 6, 8, 2, 6, 5, 7, 3, 8, 7, 11, 2, 6, 4, 8, 5, 9, 9, 13, 6, 10, 9, 14, 7, 14, 12, 15, 2, 5, 6, 7, 6, 9, 10, 14, 4, 9, 9, 12, 8, 13, 14, 15, 3, 7, 8, 11, 7, 12, 14, 15, 8, 14, 13, 15, 11, 15, 15, 16)
TRIAD_NAMES = ('003', '012', '102', '021D', '021U', '021C', '111D', '111U', '030T', '030C', '201', '120D', '120U', '120C', '210', '300')
TRICODE_TO_NAME = {i: TRIAD_NAMES[code - 1] for (i, code) in enumerate(TRICODES)}

def _tricode(G, v, u, w):
    if False:
        print('Hello World!')
    "Returns the integer code of the given triad.\n\n    This is some fancy magic that comes from Batagelj and Mrvar's paper. It\n    treats each edge joining a pair of `v`, `u`, and `w` as a bit in\n    the binary representation of an integer.\n\n    "
    combos = ((v, u, 1), (u, v, 2), (v, w, 4), (w, v, 8), (u, w, 16), (w, u, 32))
    return sum((x for (u, v, x) in combos if v in G[u]))

@not_implemented_for('undirected')
@nx._dispatch
def triadic_census(G, nodelist=None):
    if False:
        while True:
            i = 10
    'Determines the triadic census of a directed graph.\n\n    The triadic census is a count of how many of the 16 possible types of\n    triads are present in a directed graph. If a list of nodes is passed, then\n    only those triads are taken into account which have elements of nodelist in them.\n\n    Parameters\n    ----------\n    G : digraph\n       A NetworkX DiGraph\n    nodelist : list\n        List of nodes for which you want to calculate triadic census\n\n    Returns\n    -------\n    census : dict\n       Dictionary with triad type as keys and number of occurrences as values.\n\n    Examples\n    --------\n    >>> G = nx.DiGraph([(1, 2), (2, 3), (3, 1), (3, 4), (4, 1), (4, 2)])\n    >>> triadic_census = nx.triadic_census(G)\n    >>> for key, value in triadic_census.items():\n    ...     print(f"{key}: {value}")\n    ...\n    003: 0\n    012: 0\n    102: 0\n    021D: 0\n    021U: 0\n    021C: 0\n    111D: 0\n    111U: 0\n    030T: 2\n    030C: 2\n    201: 0\n    120D: 0\n    120U: 0\n    120C: 0\n    210: 0\n    300: 0\n\n    Notes\n    -----\n    This algorithm has complexity $O(m)$ where $m$ is the number of edges in\n    the graph.\n\n    Raises\n    ------\n    ValueError\n        If `nodelist` contains duplicate nodes or nodes not in `G`.\n        If you want to ignore this you can preprocess with `set(nodelist) & G.nodes`\n\n    See also\n    --------\n    triad_graph\n\n    References\n    ----------\n    .. [1] Vladimir Batagelj and Andrej Mrvar, A subquadratic triad census\n        algorithm for large sparse networks with small maximum degree,\n        University of Ljubljana,\n        http://vlado.fmf.uni-lj.si/pub/networks/doc/triads/triads.pdf\n\n    '
    nodeset = set(G.nbunch_iter(nodelist))
    if nodelist is not None and len(nodelist) != len(nodeset):
        raise ValueError('nodelist includes duplicate nodes or nodes not in G')
    N = len(G)
    Nnot = N - len(nodeset)
    m = {n: i for (i, n) in enumerate(nodeset)}
    if Nnot:
        not_nodeset = G.nodes - nodeset
        m.update(((n, i + N) for (i, n) in enumerate(not_nodeset)))
    nbrs = {n: G.pred[n].keys() | G.succ[n].keys() for n in G}
    dbl_nbrs = {n: G.pred[n].keys() & G.succ[n].keys() for n in G}
    if Nnot:
        sgl_nbrs = {n: G.pred[n].keys() ^ G.succ[n].keys() for n in not_nodeset}
        sgl = sum((1 for n in not_nodeset for nbr in sgl_nbrs[n] if nbr not in nodeset))
        sgl_edges_outside = sgl // 2
        dbl = sum((1 for n in not_nodeset for nbr in dbl_nbrs[n] if nbr not in nodeset))
        dbl_edges_outside = dbl // 2
    census = {name: 0 for name in TRIAD_NAMES}
    for v in nodeset:
        vnbrs = nbrs[v]
        dbl_vnbrs = dbl_nbrs[v]
        if Nnot:
            sgl_unbrs_bdy = sgl_unbrs_out = dbl_unbrs_bdy = dbl_unbrs_out = 0
        for u in vnbrs:
            if m[u] <= m[v]:
                continue
            unbrs = nbrs[u]
            neighbors = (vnbrs | unbrs) - {u, v}
            for w in neighbors:
                if m[u] < m[w] or (m[v] < m[w] < m[u] and v not in nbrs[w]):
                    code = _tricode(G, v, u, w)
                    census[TRICODE_TO_NAME[code]] += 1
            if u in dbl_vnbrs:
                census['102'] += N - len(neighbors) - 2
            else:
                census['012'] += N - len(neighbors) - 2
            if Nnot and u not in nodeset:
                sgl_unbrs = sgl_nbrs[u]
                sgl_unbrs_bdy += len(sgl_unbrs & vnbrs - nodeset)
                sgl_unbrs_out += len(sgl_unbrs - vnbrs - nodeset)
                dbl_unbrs = dbl_nbrs[u]
                dbl_unbrs_bdy += len(dbl_unbrs & vnbrs - nodeset)
                dbl_unbrs_out += len(dbl_unbrs - vnbrs - nodeset)
        if Nnot:
            census['012'] += sgl_edges_outside - (sgl_unbrs_out + sgl_unbrs_bdy // 2)
            census['102'] += dbl_edges_outside - (dbl_unbrs_out + dbl_unbrs_bdy // 2)
    total_triangles = N * (N - 1) * (N - 2) // 6
    triangles_without_nodeset = Nnot * (Nnot - 1) * (Nnot - 2) // 6
    total_census = total_triangles - triangles_without_nodeset
    census['003'] = total_census - sum(census.values())
    return census

@nx._dispatch
def is_triad(G):
    if False:
        while True:
            i = 10
    'Returns True if the graph G is a triad, else False.\n\n    Parameters\n    ----------\n    G : graph\n       A NetworkX Graph\n\n    Returns\n    -------\n    istriad : boolean\n       Whether G is a valid triad\n\n    Examples\n    --------\n    >>> G = nx.DiGraph([(1, 2), (2, 3), (3, 1)])\n    >>> nx.is_triad(G)\n    True\n    >>> G.add_edge(0, 1)\n    >>> nx.is_triad(G)\n    False\n    '
    if isinstance(G, nx.Graph):
        if G.order() == 3 and nx.is_directed(G):
            if not any(((n, n) in G.edges() for n in G.nodes())):
                return True
    return False

@not_implemented_for('undirected')
@nx._dispatch
def all_triplets(G):
    if False:
        i = 10
        return i + 15
    'Returns a generator of all possible sets of 3 nodes in a DiGraph.\n\n    .. deprecated:: 3.3\n\n       all_triplets is deprecated and will be removed in NetworkX version 3.5.\n       Use `itertools.combinations` instead::\n\n          all_triplets = itertools.combinations(G, 3)\n\n    Parameters\n    ----------\n    G : digraph\n       A NetworkX DiGraph\n\n    Returns\n    -------\n    triplets : generator of 3-tuples\n       Generator of tuples of 3 nodes\n\n    Examples\n    --------\n    >>> G = nx.DiGraph([(1, 2), (2, 3), (3, 4)])\n    >>> list(nx.all_triplets(G))\n    [(1, 2, 3), (1, 2, 4), (1, 3, 4), (2, 3, 4)]\n\n    '
    import warnings
    warnings.warn('\n\nall_triplets is deprecated and will be rmoved in v3.5.\nUse `itertools.combinations(G, 3)` instead.', category=DeprecationWarning, stacklevel=4)
    triplets = combinations(G.nodes(), 3)
    return triplets

@not_implemented_for('undirected')
@nx._dispatch
def all_triads(G):
    if False:
        while True:
            i = 10
    'A generator of all possible triads in G.\n\n    Parameters\n    ----------\n    G : digraph\n       A NetworkX DiGraph\n\n    Returns\n    -------\n    all_triads : generator of DiGraphs\n       Generator of triads (order-3 DiGraphs)\n\n    Examples\n    --------\n    >>> G = nx.DiGraph([(1, 2), (2, 3), (3, 1), (3, 4), (4, 1), (4, 2)])\n    >>> for triad in nx.all_triads(G):\n    ...     print(triad.edges)\n    [(1, 2), (2, 3), (3, 1)]\n    [(1, 2), (4, 1), (4, 2)]\n    [(3, 1), (3, 4), (4, 1)]\n    [(2, 3), (3, 4), (4, 2)]\n\n    '
    triplets = combinations(G.nodes(), 3)
    for triplet in triplets:
        yield G.subgraph(triplet).copy()

@not_implemented_for('undirected')
@nx._dispatch
def triads_by_type(G):
    if False:
        i = 10
        return i + 15
    'Returns a list of all triads for each triad type in a directed graph.\n    There are exactly 16 different types of triads possible. Suppose 1, 2, 3 are three\n    nodes, they will be classified as a particular triad type if their connections\n    are as follows:\n\n    - 003: 1, 2, 3\n    - 012: 1 -> 2, 3\n    - 102: 1 <-> 2, 3\n    - 021D: 1 <- 2 -> 3\n    - 021U: 1 -> 2 <- 3\n    - 021C: 1 -> 2 -> 3\n    - 111D: 1 <-> 2 <- 3\n    - 111U: 1 <-> 2 -> 3\n    - 030T: 1 -> 2 -> 3, 1 -> 3\n    - 030C: 1 <- 2 <- 3, 1 -> 3\n    - 201: 1 <-> 2 <-> 3\n    - 120D: 1 <- 2 -> 3, 1 <-> 3\n    - 120U: 1 -> 2 <- 3, 1 <-> 3\n    - 120C: 1 -> 2 -> 3, 1 <-> 3\n    - 210: 1 -> 2 <-> 3, 1 <-> 3\n    - 300: 1 <-> 2 <-> 3, 1 <-> 3\n\n    Refer to the :doc:`example gallery </auto_examples/graph/plot_triad_types>`\n    for visual examples of the triad types.\n\n    Parameters\n    ----------\n    G : digraph\n       A NetworkX DiGraph\n\n    Returns\n    -------\n    tri_by_type : dict\n       Dictionary with triad types as keys and lists of triads as values.\n\n    Examples\n    --------\n    >>> G = nx.DiGraph([(1, 2), (1, 3), (2, 3), (3, 1), (5, 6), (5, 4), (6, 7)])\n    >>> dict = nx.triads_by_type(G)\n    >>> dict[\'120C\'][0].edges()\n    OutEdgeView([(1, 2), (1, 3), (2, 3), (3, 1)])\n    >>> dict[\'012\'][0].edges()\n    OutEdgeView([(1, 2)])\n\n    References\n    ----------\n    .. [1] Snijders, T. (2012). "Transitivity and triads." University of\n        Oxford.\n        https://web.archive.org/web/20170830032057/http://www.stats.ox.ac.uk/~snijders/Trans_Triads_ha.pdf\n    '
    all_tri = all_triads(G)
    tri_by_type = defaultdict(list)
    for triad in all_tri:
        name = triad_type(triad)
        tri_by_type[name].append(triad)
    return tri_by_type

@not_implemented_for('undirected')
@nx._dispatch
def triad_type(G):
    if False:
        while True:
            i = 10
    'Returns the sociological triad type for a triad.\n\n    Parameters\n    ----------\n    G : digraph\n       A NetworkX DiGraph with 3 nodes\n\n    Returns\n    -------\n    triad_type : str\n       A string identifying the triad type\n\n    Examples\n    --------\n    >>> G = nx.DiGraph([(1, 2), (2, 3), (3, 1)])\n    >>> nx.triad_type(G)\n    \'030C\'\n    >>> G.add_edge(1, 3)\n    >>> nx.triad_type(G)\n    \'120C\'\n\n    Notes\n    -----\n    There can be 6 unique edges in a triad (order-3 DiGraph) (so 2^^6=64 unique\n    triads given 3 nodes). These 64 triads each display exactly 1 of 16\n    topologies of triads (topologies can be permuted). These topologies are\n    identified by the following notation:\n\n    {m}{a}{n}{type} (for example: 111D, 210, 102)\n\n    Here:\n\n    {m}     = number of mutual ties (takes 0, 1, 2, 3); a mutual tie is (0,1)\n              AND (1,0)\n    {a}     = number of asymmetric ties (takes 0, 1, 2, 3); an asymmetric tie\n              is (0,1) BUT NOT (1,0) or vice versa\n    {n}     = number of null ties (takes 0, 1, 2, 3); a null tie is NEITHER\n              (0,1) NOR (1,0)\n    {type}  = a letter (takes U, D, C, T) corresponding to up, down, cyclical\n              and transitive. This is only used for topologies that can have\n              more than one form (eg: 021D and 021U).\n\n    References\n    ----------\n    .. [1] Snijders, T. (2012). "Transitivity and triads." University of\n        Oxford.\n        https://web.archive.org/web/20170830032057/http://www.stats.ox.ac.uk/~snijders/Trans_Triads_ha.pdf\n    '
    if not is_triad(G):
        raise nx.NetworkXAlgorithmError('G is not a triad (order-3 DiGraph)')
    num_edges = len(G.edges())
    if num_edges == 0:
        return '003'
    elif num_edges == 1:
        return '012'
    elif num_edges == 2:
        (e1, e2) = G.edges()
        if set(e1) == set(e2):
            return '102'
        elif e1[0] == e2[0]:
            return '021D'
        elif e1[1] == e2[1]:
            return '021U'
        elif e1[1] == e2[0] or e2[1] == e1[0]:
            return '021C'
    elif num_edges == 3:
        for (e1, e2, e3) in permutations(G.edges(), 3):
            if set(e1) == set(e2):
                if e3[0] in e1:
                    return '111U'
                return '111D'
            elif set(e1).symmetric_difference(set(e2)) == set(e3):
                if {e1[0], e2[0], e3[0]} == {e1[0], e2[0], e3[0]} == set(G.nodes()):
                    return '030C'
                return '030T'
    elif num_edges == 4:
        for (e1, e2, e3, e4) in permutations(G.edges(), 4):
            if set(e1) == set(e2):
                if set(e3) == set(e4):
                    return '201'
                if {e3[0]} == {e4[0]} == set(e3).intersection(set(e4)):
                    return '120D'
                if {e3[1]} == {e4[1]} == set(e3).intersection(set(e4)):
                    return '120U'
                if e3[1] == e4[0]:
                    return '120C'
    elif num_edges == 5:
        return '210'
    elif num_edges == 6:
        return '300'

@not_implemented_for('undirected')
@py_random_state(1)
@nx._dispatch
def random_triad(G, seed=None):
    if False:
        print('Hello World!')
    'Returns a random triad from a directed graph.\n\n    .. deprecated:: 3.3\n\n       random_triad is deprecated and will be removed in version 3.5.\n       Use random sampling directly instead::\n\n          G.subgraph(random.sample(list(G), 3))\n\n    Parameters\n    ----------\n    G : digraph\n       A NetworkX DiGraph\n    seed : integer, random_state, or None (default)\n        Indicator of random number generation state.\n        See :ref:`Randomness<randomness>`.\n\n    Returns\n    -------\n    G2 : subgraph\n       A randomly selected triad (order-3 NetworkX DiGraph)\n\n    Raises\n    ------\n    NetworkXError\n        If the input Graph has less than 3 nodes.\n\n    Examples\n    --------\n    >>> G = nx.DiGraph([(1, 2), (1, 3), (2, 3), (3, 1), (5, 6), (5, 4), (6, 7)])\n    >>> triad = nx.random_triad(G, seed=1)\n    >>> triad.edges\n    OutEdgeView([(1, 2)])\n\n    '
    import warnings
    warnings.warn('\n\nrandom_triad is deprecated and will be removed in NetworkX v3.5.\nUse random.sample instead, e.g.::\n\n\tG.subgraph(random.sample(list(G), 3))\n', category=DeprecationWarning, stacklevel=5)
    if len(G) < 3:
        raise nx.NetworkXError(f'G needs at least 3 nodes to form a triad; (it has {len(G)} nodes)')
    nodes = seed.sample(list(G.nodes()), 3)
    G2 = G.subgraph(nodes)
    return G2