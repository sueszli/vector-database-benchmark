"""Generate graphs with a given degree sequence or expected degree sequence.
"""
import heapq
import math
from itertools import chain, combinations, zip_longest
from operator import itemgetter
import networkx as nx
from networkx.utils import py_random_state, random_weighted_sample
__all__ = ['configuration_model', 'directed_configuration_model', 'expected_degree_graph', 'havel_hakimi_graph', 'directed_havel_hakimi_graph', 'degree_sequence_tree', 'random_degree_sequence_graph']
chaini = chain.from_iterable

def _to_stublist(degree_sequence):
    if False:
        for i in range(10):
            print('nop')
    'Returns a list of degree-repeated node numbers.\n\n    ``degree_sequence`` is a list of nonnegative integers representing\n    the degrees of nodes in a graph.\n\n    This function returns a list of node numbers with multiplicities\n    according to the given degree sequence. For example, if the first\n    element of ``degree_sequence`` is ``3``, then the first node number,\n    ``0``, will appear at the head of the returned list three times. The\n    node numbers are assumed to be the numbers zero through\n    ``len(degree_sequence) - 1``.\n\n    Examples\n    --------\n\n    >>> degree_sequence = [1, 2, 3]\n    >>> _to_stublist(degree_sequence)\n    [0, 1, 1, 2, 2, 2]\n\n    If a zero appears in the sequence, that means the node exists but\n    has degree zero, so that number will be skipped in the returned\n    list::\n\n    >>> degree_sequence = [2, 0, 1]\n    >>> _to_stublist(degree_sequence)\n    [0, 0, 2]\n\n    '
    return list(chaini(([n] * d for (n, d) in enumerate(degree_sequence))))

def _configuration_model(deg_sequence, create_using, directed=False, in_deg_sequence=None, seed=None):
    if False:
        while True:
            i = 10
    'Helper function for generating either undirected or directed\n    configuration model graphs.\n\n    ``deg_sequence`` is a list of nonnegative integers representing the\n    degree of the node whose label is the index of the list element.\n\n    ``create_using`` see :func:`~networkx.empty_graph`.\n\n    ``directed`` and ``in_deg_sequence`` are required if you want the\n    returned graph to be generated using the directed configuration\n    model algorithm. If ``directed`` is ``False``, then ``deg_sequence``\n    is interpreted as the degree sequence of an undirected graph and\n    ``in_deg_sequence`` is ignored. Otherwise, if ``directed`` is\n    ``True``, then ``deg_sequence`` is interpreted as the out-degree\n    sequence and ``in_deg_sequence`` as the in-degree sequence of a\n    directed graph.\n\n    .. note::\n\n       ``deg_sequence`` and ``in_deg_sequence`` need not be the same\n       length.\n\n    ``seed`` is a random.Random or numpy.random.RandomState instance\n\n    This function returns a graph, directed if and only if ``directed``\n    is ``True``, generated according to the configuration model\n    algorithm. For more information on the algorithm, see the\n    :func:`configuration_model` or :func:`directed_configuration_model`\n    functions.\n\n    '
    n = len(deg_sequence)
    G = nx.empty_graph(n, create_using)
    if n == 0:
        return G
    if directed:
        pairs = zip_longest(deg_sequence, in_deg_sequence, fillvalue=0)
        (out_deg, in_deg) = zip(*pairs)
        out_stublist = _to_stublist(out_deg)
        in_stublist = _to_stublist(in_deg)
        seed.shuffle(out_stublist)
        seed.shuffle(in_stublist)
    else:
        stublist = _to_stublist(deg_sequence)
        n = len(stublist)
        half = n // 2
        seed.shuffle(stublist)
        (out_stublist, in_stublist) = (stublist[:half], stublist[half:])
    G.add_edges_from(zip(out_stublist, in_stublist))
    return G

@py_random_state(2)
@nx._dispatch(graphs=None)
def configuration_model(deg_sequence, create_using=None, seed=None):
    if False:
        for i in range(10):
            print('nop')
    'Returns a random graph with the given degree sequence.\n\n    The configuration model generates a random pseudograph (graph with\n    parallel edges and self loops) by randomly assigning edges to\n    match the given degree sequence.\n\n    Parameters\n    ----------\n    deg_sequence :  list of nonnegative integers\n        Each list entry corresponds to the degree of a node.\n    create_using : NetworkX graph constructor, optional (default MultiGraph)\n        Graph type to create. If graph instance, then cleared before populated.\n    seed : integer, random_state, or None (default)\n        Indicator of random number generation state.\n        See :ref:`Randomness<randomness>`.\n\n    Returns\n    -------\n    G : MultiGraph\n        A graph with the specified degree sequence.\n        Nodes are labeled starting at 0 with an index\n        corresponding to the position in deg_sequence.\n\n    Raises\n    ------\n    NetworkXError\n        If the degree sequence does not have an even sum.\n\n    See Also\n    --------\n    is_graphical\n\n    Notes\n    -----\n    As described by Newman [1]_.\n\n    A non-graphical degree sequence (not realizable by some simple\n    graph) is allowed since this function returns graphs with self\n    loops and parallel edges.  An exception is raised if the degree\n    sequence does not have an even sum.\n\n    This configuration model construction process can lead to\n    duplicate edges and loops.  You can remove the self-loops and\n    parallel edges (see below) which will likely result in a graph\n    that doesn\'t have the exact degree sequence specified.\n\n    The density of self-loops and parallel edges tends to decrease as\n    the number of nodes increases. However, typically the number of\n    self-loops will approach a Poisson distribution with a nonzero mean,\n    and similarly for the number of parallel edges.  Consider a node\n    with *k* stubs. The probability of being joined to another stub of\n    the same node is basically (*k* - *1*) / *N*, where *k* is the\n    degree and *N* is the number of nodes. So the probability of a\n    self-loop scales like *c* / *N* for some constant *c*. As *N* grows,\n    this means we expect *c* self-loops. Similarly for parallel edges.\n\n    References\n    ----------\n    .. [1] M.E.J. Newman, "The structure and function of complex networks",\n       SIAM REVIEW 45-2, pp 167-256, 2003.\n\n    Examples\n    --------\n    You can create a degree sequence following a particular distribution\n    by using the one of the distribution functions in\n    :mod:`~networkx.utils.random_sequence` (or one of your own). For\n    example, to create an undirected multigraph on one hundred nodes\n    with degree sequence chosen from the power law distribution:\n\n    >>> sequence = nx.random_powerlaw_tree_sequence(100, tries=5000)\n    >>> G = nx.configuration_model(sequence)\n    >>> len(G)\n    100\n    >>> actual_degrees = [d for v, d in G.degree()]\n    >>> actual_degrees == sequence\n    True\n\n    The returned graph is a multigraph, which may have parallel\n    edges. To remove any parallel edges from the returned graph:\n\n    >>> G = nx.Graph(G)\n\n    Similarly, to remove self-loops:\n\n    >>> G.remove_edges_from(nx.selfloop_edges(G))\n\n    '
    if sum(deg_sequence) % 2 != 0:
        msg = 'Invalid degree sequence: sum of degrees must be even, not odd'
        raise nx.NetworkXError(msg)
    G = nx.empty_graph(0, create_using, default=nx.MultiGraph)
    if G.is_directed():
        raise nx.NetworkXNotImplemented('not implemented for directed graphs')
    G = _configuration_model(deg_sequence, G, seed=seed)
    return G

@py_random_state(3)
@nx._dispatch(graphs=None)
def directed_configuration_model(in_degree_sequence, out_degree_sequence, create_using=None, seed=None):
    if False:
        i = 10
        return i + 15
    'Returns a directed_random graph with the given degree sequences.\n\n    The configuration model generates a random directed pseudograph\n    (graph with parallel edges and self loops) by randomly assigning\n    edges to match the given degree sequences.\n\n    Parameters\n    ----------\n    in_degree_sequence :  list of nonnegative integers\n       Each list entry corresponds to the in-degree of a node.\n    out_degree_sequence :  list of nonnegative integers\n       Each list entry corresponds to the out-degree of a node.\n    create_using : NetworkX graph constructor, optional (default MultiDiGraph)\n        Graph type to create. If graph instance, then cleared before populated.\n    seed : integer, random_state, or None (default)\n        Indicator of random number generation state.\n        See :ref:`Randomness<randomness>`.\n\n    Returns\n    -------\n    G : MultiDiGraph\n        A graph with the specified degree sequences.\n        Nodes are labeled starting at 0 with an index\n        corresponding to the position in deg_sequence.\n\n    Raises\n    ------\n    NetworkXError\n        If the degree sequences do not have the same sum.\n\n    See Also\n    --------\n    configuration_model\n\n    Notes\n    -----\n    Algorithm as described by Newman [1]_.\n\n    A non-graphical degree sequence (not realizable by some simple\n    graph) is allowed since this function returns graphs with self\n    loops and parallel edges.  An exception is raised if the degree\n    sequences does not have the same sum.\n\n    This configuration model construction process can lead to\n    duplicate edges and loops.  You can remove the self-loops and\n    parallel edges (see below) which will likely result in a graph\n    that doesn\'t have the exact degree sequence specified.  This\n    "finite-size effect" decreases as the size of the graph increases.\n\n    References\n    ----------\n    .. [1] Newman, M. E. J. and Strogatz, S. H. and Watts, D. J.\n       Random graphs with arbitrary degree distributions and their applications\n       Phys. Rev. E, 64, 026118 (2001)\n\n    Examples\n    --------\n    One can modify the in- and out-degree sequences from an existing\n    directed graph in order to create a new directed graph. For example,\n    here we modify the directed path graph:\n\n    >>> D = nx.DiGraph([(0, 1), (1, 2), (2, 3)])\n    >>> din = list(d for n, d in D.in_degree())\n    >>> dout = list(d for n, d in D.out_degree())\n    >>> din.append(1)\n    >>> dout[0] = 2\n    >>> # We now expect an edge from node 0 to a new node, node 3.\n    ... D = nx.directed_configuration_model(din, dout)\n\n    The returned graph is a directed multigraph, which may have parallel\n    edges. To remove any parallel edges from the returned graph:\n\n    >>> D = nx.DiGraph(D)\n\n    Similarly, to remove self-loops:\n\n    >>> D.remove_edges_from(nx.selfloop_edges(D))\n\n    '
    if sum(in_degree_sequence) != sum(out_degree_sequence):
        msg = 'Invalid degree sequences: sequences must have equal sums'
        raise nx.NetworkXError(msg)
    if create_using is None:
        create_using = nx.MultiDiGraph
    G = _configuration_model(out_degree_sequence, create_using, directed=True, in_deg_sequence=in_degree_sequence, seed=seed)
    name = 'directed configuration_model {} nodes {} edges'
    return G

@py_random_state(1)
@nx._dispatch(graphs=None)
def expected_degree_graph(w, seed=None, selfloops=True):
    if False:
        while True:
            i = 10
    "Returns a random graph with given expected degrees.\n\n    Given a sequence of expected degrees $W=(w_0,w_1,\\ldots,w_{n-1})$\n    of length $n$ this algorithm assigns an edge between node $u$ and\n    node $v$ with probability\n\n    .. math::\n\n       p_{uv} = \\frac{w_u w_v}{\\sum_k w_k} .\n\n    Parameters\n    ----------\n    w : list\n        The list of expected degrees.\n    selfloops: bool (default=True)\n        Set to False to remove the possibility of self-loop edges.\n    seed : integer, random_state, or None (default)\n        Indicator of random number generation state.\n        See :ref:`Randomness<randomness>`.\n\n    Returns\n    -------\n    Graph\n\n    Examples\n    --------\n    >>> z = [10 for i in range(100)]\n    >>> G = nx.expected_degree_graph(z)\n\n    Notes\n    -----\n    The nodes have integer labels corresponding to index of expected degrees\n    input sequence.\n\n    The complexity of this algorithm is $\\mathcal{O}(n+m)$ where $n$ is the\n    number of nodes and $m$ is the expected number of edges.\n\n    The model in [1]_ includes the possibility of self-loop edges.\n    Set selfloops=False to produce a graph without self loops.\n\n    For finite graphs this model doesn't produce exactly the given\n    expected degree sequence.  Instead the expected degrees are as\n    follows.\n\n    For the case without self loops (selfloops=False),\n\n    .. math::\n\n       E[deg(u)] = \\sum_{v \\ne u} p_{uv}\n                = w_u \\left( 1 - \\frac{w_u}{\\sum_k w_k} \\right) .\n\n\n    NetworkX uses the standard convention that a self-loop edge counts 2\n    in the degree of a node, so with self loops (selfloops=True),\n\n    .. math::\n\n       E[deg(u)] =  \\sum_{v \\ne u} p_{uv}  + 2 p_{uu}\n                = w_u \\left( 1 + \\frac{w_u}{\\sum_k w_k} \\right) .\n\n    References\n    ----------\n    .. [1] Fan Chung and L. Lu, Connected components in random graphs with\n       given expected degree sequences, Ann. Combinatorics, 6,\n       pp. 125-145, 2002.\n    .. [2] Joel Miller and Aric Hagberg,\n       Efficient generation of networks with given expected degrees,\n       in Algorithms and Models for the Web-Graph (WAW 2011),\n       Alan Frieze, Paul Horn, and Paweł Prałat (Eds), LNCS 6732,\n       pp. 115-126, 2011.\n    "
    n = len(w)
    G = nx.empty_graph(n)
    if n == 0 or max(w) == 0:
        return G
    rho = 1 / sum(w)
    order = sorted(enumerate(w), key=itemgetter(1), reverse=True)
    mapping = {c: u for (c, (u, v)) in enumerate(order)}
    seq = [v for (u, v) in order]
    last = n
    if not selfloops:
        last -= 1
    for u in range(last):
        v = u
        if not selfloops:
            v += 1
        factor = seq[u] * rho
        p = min(seq[v] * factor, 1)
        while v < n and p > 0:
            if p != 1:
                r = seed.random()
                v += math.floor(math.log(r, 1 - p))
            if v < n:
                q = min(seq[v] * factor, 1)
                if seed.random() < q / p:
                    G.add_edge(mapping[u], mapping[v])
                v += 1
                p = q
    return G

@nx._dispatch(graphs=None)
def havel_hakimi_graph(deg_sequence, create_using=None):
    if False:
        return 10
    'Returns a simple graph with given degree sequence constructed\n    using the Havel-Hakimi algorithm.\n\n    Parameters\n    ----------\n    deg_sequence: list of integers\n        Each integer corresponds to the degree of a node (need not be sorted).\n    create_using : NetworkX graph constructor, optional (default=nx.Graph)\n        Graph type to create. If graph instance, then cleared before populated.\n        Directed graphs are not allowed.\n\n    Raises\n    ------\n    NetworkXException\n        For a non-graphical degree sequence (i.e. one\n        not realizable by some simple graph).\n\n    Notes\n    -----\n    The Havel-Hakimi algorithm constructs a simple graph by\n    successively connecting the node of highest degree to other nodes\n    of highest degree, resorting remaining nodes by degree, and\n    repeating the process. The resulting graph has a high\n    degree-associativity.  Nodes are labeled 1,.., len(deg_sequence),\n    corresponding to their position in deg_sequence.\n\n    The basic algorithm is from Hakimi [1]_ and was generalized by\n    Kleitman and Wang [2]_.\n\n    References\n    ----------\n    .. [1] Hakimi S., On Realizability of a Set of Integers as\n       Degrees of the Vertices of a Linear Graph. I,\n       Journal of SIAM, 10(3), pp. 496-506 (1962)\n    .. [2] Kleitman D.J. and Wang D.L.\n       Algorithms for Constructing Graphs and Digraphs with Given Valences\n       and Factors  Discrete Mathematics, 6(1), pp. 79-88 (1973)\n    '
    if not nx.is_graphical(deg_sequence):
        raise nx.NetworkXError('Invalid degree sequence')
    p = len(deg_sequence)
    G = nx.empty_graph(p, create_using)
    if G.is_directed():
        raise nx.NetworkXError('Directed graphs are not supported')
    num_degs = [[] for i in range(p)]
    (dmax, dsum, n) = (0, 0, 0)
    for d in deg_sequence:
        if d > 0:
            num_degs[d].append(n)
            (dmax, dsum, n) = (max(dmax, d), dsum + d, n + 1)
    if n == 0:
        return G
    modstubs = [(0, 0)] * (dmax + 1)
    while n > 0:
        while len(num_degs[dmax]) == 0:
            dmax -= 1
        if dmax > n - 1:
            raise nx.NetworkXError('Non-graphical integer sequence')
        source = num_degs[dmax].pop()
        n -= 1
        mslen = 0
        k = dmax
        for i in range(dmax):
            while len(num_degs[k]) == 0:
                k -= 1
            target = num_degs[k].pop()
            G.add_edge(source, target)
            n -= 1
            if k > 1:
                modstubs[mslen] = (k - 1, target)
                mslen += 1
        for i in range(mslen):
            (stubval, stubtarget) = modstubs[i]
            num_degs[stubval].append(stubtarget)
            n += 1
    return G

@nx._dispatch(graphs=None)
def directed_havel_hakimi_graph(in_deg_sequence, out_deg_sequence, create_using=None):
    if False:
        return 10
    'Returns a directed graph with the given degree sequences.\n\n    Parameters\n    ----------\n    in_deg_sequence :  list of integers\n        Each list entry corresponds to the in-degree of a node.\n    out_deg_sequence : list of integers\n        Each list entry corresponds to the out-degree of a node.\n    create_using : NetworkX graph constructor, optional (default DiGraph)\n        Graph type to create. If graph instance, then cleared before populated.\n\n    Returns\n    -------\n    G : DiGraph\n        A graph with the specified degree sequences.\n        Nodes are labeled starting at 0 with an index\n        corresponding to the position in deg_sequence\n\n    Raises\n    ------\n    NetworkXError\n        If the degree sequences are not digraphical.\n\n    See Also\n    --------\n    configuration_model\n\n    Notes\n    -----\n    Algorithm as described by Kleitman and Wang [1]_.\n\n    References\n    ----------\n    .. [1] D.J. Kleitman and D.L. Wang\n       Algorithms for Constructing Graphs and Digraphs with Given Valences\n       and Factors Discrete Mathematics, 6(1), pp. 79-88 (1973)\n    '
    in_deg_sequence = nx.utils.make_list_of_ints(in_deg_sequence)
    out_deg_sequence = nx.utils.make_list_of_ints(out_deg_sequence)
    (sumin, sumout) = (0, 0)
    (nin, nout) = (len(in_deg_sequence), len(out_deg_sequence))
    maxn = max(nin, nout)
    G = nx.empty_graph(maxn, create_using, default=nx.DiGraph)
    if maxn == 0:
        return G
    maxin = 0
    (stubheap, zeroheap) = ([], [])
    for n in range(maxn):
        (in_deg, out_deg) = (0, 0)
        if n < nout:
            out_deg = out_deg_sequence[n]
        if n < nin:
            in_deg = in_deg_sequence[n]
        if in_deg < 0 or out_deg < 0:
            raise nx.NetworkXError('Invalid degree sequences. Sequence values must be positive.')
        (sumin, sumout, maxin) = (sumin + in_deg, sumout + out_deg, max(maxin, in_deg))
        if in_deg > 0:
            stubheap.append((-1 * out_deg, -1 * in_deg, n))
        elif out_deg > 0:
            zeroheap.append((-1 * out_deg, n))
    if sumin != sumout:
        raise nx.NetworkXError('Invalid degree sequences. Sequences must have equal sums.')
    heapq.heapify(stubheap)
    heapq.heapify(zeroheap)
    modstubs = [(0, 0, 0)] * (maxin + 1)
    while stubheap:
        (freeout, freein, target) = heapq.heappop(stubheap)
        freein *= -1
        if freein > len(stubheap) + len(zeroheap):
            raise nx.NetworkXError('Non-digraphical integer sequence')
        mslen = 0
        for i in range(freein):
            if zeroheap and (not stubheap or stubheap[0][0] > zeroheap[0][0]):
                (stubout, stubsource) = heapq.heappop(zeroheap)
                stubin = 0
            else:
                (stubout, stubin, stubsource) = heapq.heappop(stubheap)
            if stubout == 0:
                raise nx.NetworkXError('Non-digraphical integer sequence')
            G.add_edge(stubsource, target)
            if stubout + 1 < 0 or stubin < 0:
                modstubs[mslen] = (stubout + 1, stubin, stubsource)
                mslen += 1
        for i in range(mslen):
            stub = modstubs[i]
            if stub[1] < 0:
                heapq.heappush(stubheap, stub)
            else:
                heapq.heappush(zeroheap, (stub[0], stub[2]))
        if freeout < 0:
            heapq.heappush(zeroheap, (freeout, target))
    return G

@nx._dispatch(graphs=None)
def degree_sequence_tree(deg_sequence, create_using=None):
    if False:
        print('Hello World!')
    'Make a tree for the given degree sequence.\n\n    A tree has #nodes-#edges=1 so\n    the degree sequence must have\n    len(deg_sequence)-sum(deg_sequence)/2=1\n    '
    degree_sum = sum(deg_sequence)
    if degree_sum % 2 != 0:
        msg = 'Invalid degree sequence: sum of degrees must be even, not odd'
        raise nx.NetworkXError(msg)
    if len(deg_sequence) - degree_sum // 2 != 1:
        msg = 'Invalid degree sequence: tree must have number of nodes equal to one less than the number of edges'
        raise nx.NetworkXError(msg)
    G = nx.empty_graph(0, create_using)
    if G.is_directed():
        raise nx.NetworkXError('Directed Graph not supported')
    deg = sorted((s for s in deg_sequence if s > 1), reverse=True)
    n = len(deg) + 2
    nx.add_path(G, range(n))
    last = n
    for source in range(1, n - 1):
        nedges = deg.pop() - 2
        for target in range(last, last + nedges):
            G.add_edge(source, target)
        last += nedges
    if len(G) > len(deg_sequence):
        G.remove_node(0)
    return G

@py_random_state(1)
@nx._dispatch(graphs=None)
def random_degree_sequence_graph(sequence, seed=None, tries=10):
    if False:
        i = 10
        return i + 15
    'Returns a simple random graph with the given degree sequence.\n\n    If the maximum degree $d_m$ in the sequence is $O(m^{1/4})$ then the\n    algorithm produces almost uniform random graphs in $O(m d_m)$ time\n    where $m$ is the number of edges.\n\n    Parameters\n    ----------\n    sequence :  list of integers\n        Sequence of degrees\n    seed : integer, random_state, or None (default)\n        Indicator of random number generation state.\n        See :ref:`Randomness<randomness>`.\n    tries : int, optional\n        Maximum number of tries to create a graph\n\n    Returns\n    -------\n    G : Graph\n        A graph with the specified degree sequence.\n        Nodes are labeled starting at 0 with an index\n        corresponding to the position in the sequence.\n\n    Raises\n    ------\n    NetworkXUnfeasible\n        If the degree sequence is not graphical.\n    NetworkXError\n        If a graph is not produced in specified number of tries\n\n    See Also\n    --------\n    is_graphical, configuration_model\n\n    Notes\n    -----\n    The generator algorithm [1]_ is not guaranteed to produce a graph.\n\n    References\n    ----------\n    .. [1] Moshen Bayati, Jeong Han Kim, and Amin Saberi,\n       A sequential algorithm for generating random graphs.\n       Algorithmica, Volume 58, Number 4, 860-910,\n       DOI: 10.1007/s00453-009-9340-1\n\n    Examples\n    --------\n    >>> sequence = [1, 2, 2, 3]\n    >>> G = nx.random_degree_sequence_graph(sequence, seed=42)\n    >>> sorted(d for n, d in G.degree())\n    [1, 2, 2, 3]\n    '
    DSRG = DegreeSequenceRandomGraph(sequence, seed)
    for try_n in range(tries):
        try:
            return DSRG.generate()
        except nx.NetworkXUnfeasible:
            pass
    raise nx.NetworkXError(f'failed to generate graph in {tries} tries')

class DegreeSequenceRandomGraph:

    def __init__(self, degree, rng):
        if False:
            return 10
        if not nx.is_graphical(degree):
            raise nx.NetworkXUnfeasible('degree sequence is not graphical')
        self.rng = rng
        self.degree = list(degree)
        self.m = sum(self.degree) / 2.0
        try:
            self.dmax = max(self.degree)
        except ValueError:
            self.dmax = 0

    def generate(self):
        if False:
            i = 10
            return i + 15
        self.remaining_degree = dict(enumerate(self.degree))
        self.graph = nx.Graph()
        self.graph.add_nodes_from(self.remaining_degree)
        for (n, d) in list(self.remaining_degree.items()):
            if d == 0:
                del self.remaining_degree[n]
        if len(self.remaining_degree) > 0:
            self.phase1()
            self.phase2()
            self.phase3()
        return self.graph

    def update_remaining(self, u, v, aux_graph=None):
        if False:
            while True:
                i = 10
        if aux_graph is not None:
            aux_graph.remove_edge(u, v)
        if self.remaining_degree[u] == 1:
            del self.remaining_degree[u]
            if aux_graph is not None:
                aux_graph.remove_node(u)
        else:
            self.remaining_degree[u] -= 1
        if self.remaining_degree[v] == 1:
            del self.remaining_degree[v]
            if aux_graph is not None:
                aux_graph.remove_node(v)
        else:
            self.remaining_degree[v] -= 1

    def p(self, u, v):
        if False:
            i = 10
            return i + 15
        return 1 - self.degree[u] * self.degree[v] / (4.0 * self.m)

    def q(self, u, v):
        if False:
            for i in range(10):
                print('nop')
        norm = max(self.remaining_degree.values()) ** 2
        return self.remaining_degree[u] * self.remaining_degree[v] / norm

    def suitable_edge(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns True if and only if an arbitrary remaining node can\n        potentially be joined with some other remaining node.\n\n        '
        nodes = iter(self.remaining_degree)
        u = next(nodes)
        return any((v not in self.graph[u] for v in nodes))

    def phase1(self):
        if False:
            print('Hello World!')
        rem_deg = self.remaining_degree
        while sum(rem_deg.values()) >= 2 * self.dmax ** 2:
            (u, v) = sorted(random_weighted_sample(rem_deg, 2, self.rng))
            if self.graph.has_edge(u, v):
                continue
            if self.rng.random() < self.p(u, v):
                self.graph.add_edge(u, v)
                self.update_remaining(u, v)

    def phase2(self):
        if False:
            while True:
                i = 10
        remaining_deg = self.remaining_degree
        rng = self.rng
        while len(remaining_deg) >= 2 * self.dmax:
            while True:
                (u, v) = sorted(rng.sample(list(remaining_deg.keys()), 2))
                if self.graph.has_edge(u, v):
                    continue
                if rng.random() < self.q(u, v):
                    break
            if rng.random() < self.p(u, v):
                self.graph.add_edge(u, v)
                self.update_remaining(u, v)

    def phase3(self):
        if False:
            for i in range(10):
                print('nop')
        potential_edges = combinations(self.remaining_degree, 2)
        H = nx.Graph([(u, v) for (u, v) in potential_edges if not self.graph.has_edge(u, v)])
        rng = self.rng
        while self.remaining_degree:
            if not self.suitable_edge():
                raise nx.NetworkXUnfeasible('no suitable edges left')
            while True:
                (u, v) = sorted(rng.choice(list(H.edges())))
                if rng.random() < self.q(u, v):
                    break
            if rng.random() < self.p(u, v):
                self.graph.add_edge(u, v)
                self.update_remaining(u, v, aux_graph=H)