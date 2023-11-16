"""
Generators for random graphs.

"""
import itertools
import math
from collections import defaultdict
import networkx as nx
from networkx.utils import py_random_state
from .classic import complete_graph, empty_graph, path_graph, star_graph
from .degree_seq import degree_sequence_tree
__all__ = ['fast_gnp_random_graph', 'gnp_random_graph', 'dense_gnm_random_graph', 'gnm_random_graph', 'erdos_renyi_graph', 'binomial_graph', 'newman_watts_strogatz_graph', 'watts_strogatz_graph', 'connected_watts_strogatz_graph', 'random_regular_graph', 'barabasi_albert_graph', 'dual_barabasi_albert_graph', 'extended_barabasi_albert_graph', 'powerlaw_cluster_graph', 'random_lobster', 'random_shell_graph', 'random_powerlaw_tree', 'random_powerlaw_tree_sequence', 'random_kernel_graph']

@py_random_state(2)
@nx._dispatch(graphs=None)
def fast_gnp_random_graph(n, p, seed=None, directed=False):
    if False:
        return 10
    'Returns a $G_{n,p}$ random graph, also known as an Erdős-Rényi graph or\n    a binomial graph.\n\n    Parameters\n    ----------\n    n : int\n        The number of nodes.\n    p : float\n        Probability for edge creation.\n    seed : integer, random_state, or None (default)\n        Indicator of random number generation state.\n        See :ref:`Randomness<randomness>`.\n    directed : bool, optional (default=False)\n        If True, this function returns a directed graph.\n\n    Notes\n    -----\n    The $G_{n,p}$ graph algorithm chooses each of the $[n (n - 1)] / 2$\n    (undirected) or $n (n - 1)$ (directed) possible edges with probability $p$.\n\n    This algorithm [1]_ runs in $O(n + m)$ time, where `m` is the expected number of\n    edges, which equals $p n (n - 1) / 2$. This should be faster than\n    :func:`gnp_random_graph` when $p$ is small and the expected number of edges\n    is small (that is, the graph is sparse).\n\n    See Also\n    --------\n    gnp_random_graph\n\n    References\n    ----------\n    .. [1] Vladimir Batagelj and Ulrik Brandes,\n       "Efficient generation of large random networks",\n       Phys. Rev. E, 71, 036113, 2005.\n    '
    G = empty_graph(n)
    if p <= 0 or p >= 1:
        return nx.gnp_random_graph(n, p, seed=seed, directed=directed)
    lp = math.log(1.0 - p)
    if directed:
        G = nx.DiGraph(G)
        v = 1
        w = -1
        while v < n:
            lr = math.log(1.0 - seed.random())
            w = w + 1 + int(lr / lp)
            while w >= v and v < n:
                w = w - v
                v = v + 1
            if v < n:
                G.add_edge(w, v)
    v = 1
    w = -1
    while v < n:
        lr = math.log(1.0 - seed.random())
        w = w + 1 + int(lr / lp)
        while w >= v and v < n:
            w = w - v
            v = v + 1
        if v < n:
            G.add_edge(v, w)
    return G

@py_random_state(2)
@nx._dispatch(graphs=None)
def gnp_random_graph(n, p, seed=None, directed=False):
    if False:
        i = 10
        return i + 15
    'Returns a $G_{n,p}$ random graph, also known as an Erdős-Rényi graph\n    or a binomial graph.\n\n    The $G_{n,p}$ model chooses each of the possible edges with probability $p$.\n\n    Parameters\n    ----------\n    n : int\n        The number of nodes.\n    p : float\n        Probability for edge creation.\n    seed : integer, random_state, or None (default)\n        Indicator of random number generation state.\n        See :ref:`Randomness<randomness>`.\n    directed : bool, optional (default=False)\n        If True, this function returns a directed graph.\n\n    See Also\n    --------\n    fast_gnp_random_graph\n\n    Notes\n    -----\n    This algorithm [2]_ runs in $O(n^2)$ time.  For sparse graphs (that is, for\n    small values of $p$), :func:`fast_gnp_random_graph` is a faster algorithm.\n\n    :func:`binomial_graph` and :func:`erdos_renyi_graph` are\n    aliases for :func:`gnp_random_graph`.\n\n    >>> nx.binomial_graph is nx.gnp_random_graph\n    True\n    >>> nx.erdos_renyi_graph is nx.gnp_random_graph\n    True\n\n    References\n    ----------\n    .. [1] P. Erdős and A. Rényi, On Random Graphs, Publ. Math. 6, 290 (1959).\n    .. [2] E. N. Gilbert, Random Graphs, Ann. Math. Stat., 30, 1141 (1959).\n    '
    if directed:
        edges = itertools.permutations(range(n), 2)
        G = nx.DiGraph()
    else:
        edges = itertools.combinations(range(n), 2)
        G = nx.Graph()
    G.add_nodes_from(range(n))
    if p <= 0:
        return G
    if p >= 1:
        return complete_graph(n, create_using=G)
    for e in edges:
        if seed.random() < p:
            G.add_edge(*e)
    return G
binomial_graph = gnp_random_graph
erdos_renyi_graph = gnp_random_graph

@py_random_state(2)
@nx._dispatch(graphs=None)
def dense_gnm_random_graph(n, m, seed=None):
    if False:
        i = 10
        return i + 15
    "Returns a $G_{n,m}$ random graph.\n\n    In the $G_{n,m}$ model, a graph is chosen uniformly at random from the set\n    of all graphs with $n$ nodes and $m$ edges.\n\n    This algorithm should be faster than :func:`gnm_random_graph` for dense\n    graphs.\n\n    Parameters\n    ----------\n    n : int\n        The number of nodes.\n    m : int\n        The number of edges.\n    seed : integer, random_state, or None (default)\n        Indicator of random number generation state.\n        See :ref:`Randomness<randomness>`.\n\n    See Also\n    --------\n    gnm_random_graph\n\n    Notes\n    -----\n    Algorithm by Keith M. Briggs Mar 31, 2006.\n    Inspired by Knuth's Algorithm S (Selection sampling technique),\n    in section 3.4.2 of [1]_.\n\n    References\n    ----------\n    .. [1] Donald E. Knuth, The Art of Computer Programming,\n        Volume 2/Seminumerical algorithms, Third Edition, Addison-Wesley, 1997.\n    "
    mmax = n * (n - 1) // 2
    if m >= mmax:
        G = complete_graph(n)
    else:
        G = empty_graph(n)
    if n == 1 or m >= mmax:
        return G
    u = 0
    v = 1
    t = 0
    k = 0
    while True:
        if seed.randrange(mmax - t) < m - k:
            G.add_edge(u, v)
            k += 1
            if k == m:
                return G
        t += 1
        v += 1
        if v == n:
            u += 1
            v = u + 1

@py_random_state(2)
@nx._dispatch(graphs=None)
def gnm_random_graph(n, m, seed=None, directed=False):
    if False:
        i = 10
        return i + 15
    'Returns a $G_{n,m}$ random graph.\n\n    In the $G_{n,m}$ model, a graph is chosen uniformly at random from the set\n    of all graphs with $n$ nodes and $m$ edges.\n\n    This algorithm should be faster than :func:`dense_gnm_random_graph` for\n    sparse graphs.\n\n    Parameters\n    ----------\n    n : int\n        The number of nodes.\n    m : int\n        The number of edges.\n    seed : integer, random_state, or None (default)\n        Indicator of random number generation state.\n        See :ref:`Randomness<randomness>`.\n    directed : bool, optional (default=False)\n        If True return a directed graph\n\n    See also\n    --------\n    dense_gnm_random_graph\n\n    '
    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    G.add_nodes_from(range(n))
    if n == 1:
        return G
    max_edges = n * (n - 1)
    if not directed:
        max_edges /= 2.0
    if m >= max_edges:
        return complete_graph(n, create_using=G)
    nlist = list(G)
    edge_count = 0
    while edge_count < m:
        u = seed.choice(nlist)
        v = seed.choice(nlist)
        if u == v or G.has_edge(u, v):
            continue
        else:
            G.add_edge(u, v)
            edge_count = edge_count + 1
    return G

@py_random_state(3)
@nx._dispatch(graphs=None)
def newman_watts_strogatz_graph(n, k, p, seed=None):
    if False:
        i = 10
        return i + 15
    'Returns a Newman–Watts–Strogatz small-world graph.\n\n    Parameters\n    ----------\n    n : int\n        The number of nodes.\n    k : int\n        Each node is joined with its `k` nearest neighbors in a ring\n        topology.\n    p : float\n        The probability of adding a new edge for each edge.\n    seed : integer, random_state, or None (default)\n        Indicator of random number generation state.\n        See :ref:`Randomness<randomness>`.\n\n    Notes\n    -----\n    First create a ring over $n$ nodes [1]_.  Then each node in the ring is\n    connected with its $k$ nearest neighbors (or $k - 1$ neighbors if $k$\n    is odd).  Then shortcuts are created by adding new edges as follows: for\n    each edge $(u, v)$ in the underlying "$n$-ring with $k$ nearest\n    neighbors" with probability $p$ add a new edge $(u, w)$ with\n    randomly-chosen existing node $w$.  In contrast with\n    :func:`watts_strogatz_graph`, no edges are removed.\n\n    See Also\n    --------\n    watts_strogatz_graph\n\n    References\n    ----------\n    .. [1] M. E. J. Newman and D. J. Watts,\n       Renormalization group analysis of the small-world network model,\n       Physics Letters A, 263, 341, 1999.\n       https://doi.org/10.1016/S0375-9601(99)00757-4\n    '
    if k > n:
        raise nx.NetworkXError('k>=n, choose smaller k or larger n')
    if k == n:
        return nx.complete_graph(n)
    G = empty_graph(n)
    nlist = list(G.nodes())
    fromv = nlist
    for j in range(1, k // 2 + 1):
        tov = fromv[j:] + fromv[0:j]
        for i in range(len(fromv)):
            G.add_edge(fromv[i], tov[i])
    e = list(G.edges())
    for (u, v) in e:
        if seed.random() < p:
            w = seed.choice(nlist)
            while w == u or G.has_edge(u, w):
                w = seed.choice(nlist)
                if G.degree(u) >= n - 1:
                    break
            else:
                G.add_edge(u, w)
    return G

@py_random_state(3)
@nx._dispatch(graphs=None)
def watts_strogatz_graph(n, k, p, seed=None):
    if False:
        print('Hello World!')
    'Returns a Watts–Strogatz small-world graph.\n\n    Parameters\n    ----------\n    n : int\n        The number of nodes\n    k : int\n        Each node is joined with its `k` nearest neighbors in a ring\n        topology.\n    p : float\n        The probability of rewiring each edge\n    seed : integer, random_state, or None (default)\n        Indicator of random number generation state.\n        See :ref:`Randomness<randomness>`.\n\n    See Also\n    --------\n    newman_watts_strogatz_graph\n    connected_watts_strogatz_graph\n\n    Notes\n    -----\n    First create a ring over $n$ nodes [1]_.  Then each node in the ring is joined\n    to its $k$ nearest neighbors (or $k - 1$ neighbors if $k$ is odd).\n    Then shortcuts are created by replacing some edges as follows: for each\n    edge $(u, v)$ in the underlying "$n$-ring with $k$ nearest neighbors"\n    with probability $p$ replace it with a new edge $(u, w)$ with uniformly\n    random choice of existing node $w$.\n\n    In contrast with :func:`newman_watts_strogatz_graph`, the random rewiring\n    does not increase the number of edges. The rewired graph is not guaranteed\n    to be connected as in :func:`connected_watts_strogatz_graph`.\n\n    References\n    ----------\n    .. [1] Duncan J. Watts and Steven H. Strogatz,\n       Collective dynamics of small-world networks,\n       Nature, 393, pp. 440--442, 1998.\n    '
    if k > n:
        raise nx.NetworkXError('k>n, choose smaller k or larger n')
    if k == n:
        return nx.complete_graph(n)
    G = nx.Graph()
    nodes = list(range(n))
    for j in range(1, k // 2 + 1):
        targets = nodes[j:] + nodes[0:j]
        G.add_edges_from(zip(nodes, targets))
    for j in range(1, k // 2 + 1):
        targets = nodes[j:] + nodes[0:j]
        for (u, v) in zip(nodes, targets):
            if seed.random() < p:
                w = seed.choice(nodes)
                while w == u or G.has_edge(u, w):
                    w = seed.choice(nodes)
                    if G.degree(u) >= n - 1:
                        break
                else:
                    G.remove_edge(u, v)
                    G.add_edge(u, w)
    return G

@py_random_state(4)
@nx._dispatch(graphs=None)
def connected_watts_strogatz_graph(n, k, p, tries=100, seed=None):
    if False:
        print('Hello World!')
    'Returns a connected Watts–Strogatz small-world graph.\n\n    Attempts to generate a connected graph by repeated generation of\n    Watts–Strogatz small-world graphs.  An exception is raised if the maximum\n    number of tries is exceeded.\n\n    Parameters\n    ----------\n    n : int\n        The number of nodes\n    k : int\n        Each node is joined with its `k` nearest neighbors in a ring\n        topology.\n    p : float\n        The probability of rewiring each edge\n    tries : int\n        Number of attempts to generate a connected graph.\n    seed : integer, random_state, or None (default)\n        Indicator of random number generation state.\n        See :ref:`Randomness<randomness>`.\n\n    Notes\n    -----\n    First create a ring over $n$ nodes [1]_.  Then each node in the ring is joined\n    to its $k$ nearest neighbors (or $k - 1$ neighbors if $k$ is odd).\n    Then shortcuts are created by replacing some edges as follows: for each\n    edge $(u, v)$ in the underlying "$n$-ring with $k$ nearest neighbors"\n    with probability $p$ replace it with a new edge $(u, w)$ with uniformly\n    random choice of existing node $w$.\n    The entire process is repeated until a connected graph results.\n\n    See Also\n    --------\n    newman_watts_strogatz_graph\n    watts_strogatz_graph\n\n    References\n    ----------\n    .. [1] Duncan J. Watts and Steven H. Strogatz,\n       Collective dynamics of small-world networks,\n       Nature, 393, pp. 440--442, 1998.\n    '
    for i in range(tries):
        G = watts_strogatz_graph(n, k, p, seed)
        if nx.is_connected(G):
            return G
    raise nx.NetworkXError('Maximum number of tries exceeded')

@py_random_state(2)
@nx._dispatch(graphs=None)
def random_regular_graph(d, n, seed=None):
    if False:
        return 10
    "Returns a random $d$-regular graph on $n$ nodes.\n\n    A regular graph is a graph where each node has the same number of neighbors.\n\n    The resulting graph has no self-loops or parallel edges.\n\n    Parameters\n    ----------\n    d : int\n      The degree of each node.\n    n : integer\n      The number of nodes. The value of $n \\times d$ must be even.\n    seed : integer, random_state, or None (default)\n        Indicator of random number generation state.\n        See :ref:`Randomness<randomness>`.\n\n    Notes\n    -----\n    The nodes are numbered from $0$ to $n - 1$.\n\n    Kim and Vu's paper [2]_ shows that this algorithm samples in an\n    asymptotically uniform way from the space of random graphs when\n    $d = O(n^{1 / 3 - \\epsilon})$.\n\n    Raises\n    ------\n\n    NetworkXError\n        If $n \\times d$ is odd or $d$ is greater than or equal to $n$.\n\n    References\n    ----------\n    .. [1] A. Steger and N. Wormald,\n       Generating random regular graphs quickly,\n       Probability and Computing 8 (1999), 377-396, 1999.\n       https://doi.org/10.1017/S0963548399003867\n\n    .. [2] Jeong Han Kim and Van H. Vu,\n       Generating random regular graphs,\n       Proceedings of the thirty-fifth ACM symposium on Theory of computing,\n       San Diego, CA, USA, pp 213--222, 2003.\n       http://portal.acm.org/citation.cfm?id=780542.780576\n    "
    if n * d % 2 != 0:
        raise nx.NetworkXError('n * d must be even')
    if not 0 <= d < n:
        raise nx.NetworkXError('the 0 <= d < n inequality must be satisfied')
    if d == 0:
        return empty_graph(n)

    def _suitable(edges, potential_edges):
        if False:
            while True:
                i = 10
        if not potential_edges:
            return True
        for s1 in potential_edges:
            for s2 in potential_edges:
                if s1 == s2:
                    break
                if s1 > s2:
                    (s1, s2) = (s2, s1)
                if (s1, s2) not in edges:
                    return True
        return False

    def _try_creation():
        if False:
            for i in range(10):
                print('nop')
        edges = set()
        stubs = list(range(n)) * d
        while stubs:
            potential_edges = defaultdict(lambda : 0)
            seed.shuffle(stubs)
            stubiter = iter(stubs)
            for (s1, s2) in zip(stubiter, stubiter):
                if s1 > s2:
                    (s1, s2) = (s2, s1)
                if s1 != s2 and (s1, s2) not in edges:
                    edges.add((s1, s2))
                else:
                    potential_edges[s1] += 1
                    potential_edges[s2] += 1
            if not _suitable(edges, potential_edges):
                return None
            stubs = [node for (node, potential) in potential_edges.items() for _ in range(potential)]
        return edges
    edges = _try_creation()
    while edges is None:
        edges = _try_creation()
    G = nx.Graph()
    G.add_edges_from(edges)
    return G

def _random_subset(seq, m, rng):
    if False:
        for i in range(10):
            print('nop')
    'Return m unique elements from seq.\n\n    This differs from random.sample which can return repeated\n    elements if seq holds repeated elements.\n\n    Note: rng is a random.Random or numpy.random.RandomState instance.\n    '
    targets = set()
    while len(targets) < m:
        x = rng.choice(seq)
        targets.add(x)
    return targets

@py_random_state(2)
@nx._dispatch(graphs=None)
def barabasi_albert_graph(n, m, seed=None, initial_graph=None):
    if False:
        print('Hello World!')
    'Returns a random graph using Barabási–Albert preferential attachment\n\n    A graph of $n$ nodes is grown by attaching new nodes each with $m$\n    edges that are preferentially attached to existing nodes with high degree.\n\n    Parameters\n    ----------\n    n : int\n        Number of nodes\n    m : int\n        Number of edges to attach from a new node to existing nodes\n    seed : integer, random_state, or None (default)\n        Indicator of random number generation state.\n        See :ref:`Randomness<randomness>`.\n    initial_graph : Graph or None (default)\n        Initial network for Barabási–Albert algorithm.\n        It should be a connected graph for most use cases.\n        A copy of `initial_graph` is used.\n        If None, starts from a star graph on (m+1) nodes.\n\n    Returns\n    -------\n    G : Graph\n\n    Raises\n    ------\n    NetworkXError\n        If `m` does not satisfy ``1 <= m < n``, or\n        the initial graph number of nodes m0 does not satisfy ``m <= m0 <= n``.\n\n    References\n    ----------\n    .. [1] A. L. Barabási and R. Albert "Emergence of scaling in\n       random networks", Science 286, pp 509-512, 1999.\n    '
    if m < 1 or m >= n:
        raise nx.NetworkXError(f'Barabási–Albert network must have m >= 1 and m < n, m = {m}, n = {n}')
    if initial_graph is None:
        G = star_graph(m)
    else:
        if len(initial_graph) < m or len(initial_graph) > n:
            raise nx.NetworkXError(f'Barabási–Albert initial graph needs between m={m} and n={n} nodes')
        G = initial_graph.copy()
    repeated_nodes = [n for (n, d) in G.degree() for _ in range(d)]
    source = len(G)
    while source < n:
        targets = _random_subset(repeated_nodes, m, seed)
        G.add_edges_from(zip([source] * m, targets))
        repeated_nodes.extend(targets)
        repeated_nodes.extend([source] * m)
        source += 1
    return G

@py_random_state(4)
@nx._dispatch(graphs=None)
def dual_barabasi_albert_graph(n, m1, m2, p, seed=None, initial_graph=None):
    if False:
        print('Hello World!')
    'Returns a random graph using dual Barabási–Albert preferential attachment\n\n    A graph of $n$ nodes is grown by attaching new nodes each with either $m_1$\n    edges (with probability $p$) or $m_2$ edges (with probability $1-p$) that\n    are preferentially attached to existing nodes with high degree.\n\n    Parameters\n    ----------\n    n : int\n        Number of nodes\n    m1 : int\n        Number of edges to link each new node to existing nodes with probability $p$\n    m2 : int\n        Number of edges to link each new node to existing nodes with probability $1-p$\n    p : float\n        The probability of attaching $m_1$ edges (as opposed to $m_2$ edges)\n    seed : integer, random_state, or None (default)\n        Indicator of random number generation state.\n        See :ref:`Randomness<randomness>`.\n    initial_graph : Graph or None (default)\n        Initial network for Barabási–Albert algorithm.\n        A copy of `initial_graph` is used.\n        It should be connected for most use cases.\n        If None, starts from an star graph on max(m1, m2) + 1 nodes.\n\n    Returns\n    -------\n    G : Graph\n\n    Raises\n    ------\n    NetworkXError\n        If `m1` and `m2` do not satisfy ``1 <= m1,m2 < n``, or\n        `p` does not satisfy ``0 <= p <= 1``, or\n        the initial graph number of nodes m0 does not satisfy m1, m2 <= m0 <= n.\n\n    References\n    ----------\n    .. [1] N. Moshiri "The dual-Barabasi-Albert model", arXiv:1810.10538.\n    '
    if m1 < 1 or m1 >= n:
        raise nx.NetworkXError(f'Dual Barabási–Albert must have m1 >= 1 and m1 < n, m1 = {m1}, n = {n}')
    if m2 < 1 or m2 >= n:
        raise nx.NetworkXError(f'Dual Barabási–Albert must have m2 >= 1 and m2 < n, m2 = {m2}, n = {n}')
    if p < 0 or p > 1:
        raise nx.NetworkXError(f'Dual Barabási–Albert network must have 0 <= p <= 1, p = {p}')
    if p == 1:
        return barabasi_albert_graph(n, m1, seed)
    elif p == 0:
        return barabasi_albert_graph(n, m2, seed)
    if initial_graph is None:
        G = star_graph(max(m1, m2))
    else:
        if len(initial_graph) < max(m1, m2) or len(initial_graph) > n:
            raise nx.NetworkXError(f'Barabási–Albert initial graph must have between max(m1, m2) = {max(m1, m2)} and n = {n} nodes')
        G = initial_graph.copy()
    targets = list(G)
    repeated_nodes = [n for (n, d) in G.degree() for _ in range(d)]
    source = len(G)
    while source < n:
        if seed.random() < p:
            m = m1
        else:
            m = m2
        targets = _random_subset(repeated_nodes, m, seed)
        G.add_edges_from(zip([source] * m, targets))
        repeated_nodes.extend(targets)
        repeated_nodes.extend([source] * m)
        source += 1
    return G

@py_random_state(4)
@nx._dispatch(graphs=None)
def extended_barabasi_albert_graph(n, m, p, q, seed=None):
    if False:
        print('Hello World!')
    'Returns an extended Barabási–Albert model graph.\n\n    An extended Barabási–Albert model graph is a random graph constructed\n    using preferential attachment. The extended model allows new edges,\n    rewired edges or new nodes. Based on the probabilities $p$ and $q$\n    with $p + q < 1$, the growing behavior of the graph is determined as:\n\n    1) With $p$ probability, $m$ new edges are added to the graph,\n    starting from randomly chosen existing nodes and attached preferentially at the other end.\n\n    2) With $q$ probability, $m$ existing edges are rewired\n    by randomly choosing an edge and rewiring one end to a preferentially chosen node.\n\n    3) With $(1 - p - q)$ probability, $m$ new nodes are added to the graph\n    with edges attached preferentially.\n\n    When $p = q = 0$, the model behaves just like the Barabási–Alber model.\n\n    Parameters\n    ----------\n    n : int\n        Number of nodes\n    m : int\n        Number of edges with which a new node attaches to existing nodes\n    p : float\n        Probability value for adding an edge between existing nodes. p + q < 1\n    q : float\n        Probability value of rewiring of existing edges. p + q < 1\n    seed : integer, random_state, or None (default)\n        Indicator of random number generation state.\n        See :ref:`Randomness<randomness>`.\n\n    Returns\n    -------\n    G : Graph\n\n    Raises\n    ------\n    NetworkXError\n        If `m` does not satisfy ``1 <= m < n`` or ``1 >= p + q``\n\n    References\n    ----------\n    .. [1] Albert, R., & Barabási, A. L. (2000)\n       Topology of evolving networks: local events and universality\n       Physical review letters, 85(24), 5234.\n    '
    if m < 1 or m >= n:
        msg = f'Extended Barabasi-Albert network needs m>=1 and m<n, m={m}, n={n}'
        raise nx.NetworkXError(msg)
    if p + q >= 1:
        msg = f'Extended Barabasi-Albert network needs p + q <= 1, p={p}, q={q}'
        raise nx.NetworkXError(msg)
    G = empty_graph(m)
    attachment_preference = []
    attachment_preference.extend(range(m))
    new_node = m
    while new_node < n:
        a_probability = seed.random()
        clique_degree = len(G) - 1
        clique_size = len(G) * clique_degree / 2
        if a_probability < p and G.size() <= clique_size - m:
            eligible_nodes = [nd for (nd, deg) in G.degree() if deg < clique_degree]
            for i in range(m):
                src_node = seed.choice(eligible_nodes)
                prohibited_nodes = list(G[src_node])
                prohibited_nodes.append(src_node)
                dest_node = seed.choice([nd for nd in attachment_preference if nd not in prohibited_nodes])
                G.add_edge(src_node, dest_node)
                attachment_preference.append(src_node)
                attachment_preference.append(dest_node)
                if G.degree(src_node) == clique_degree:
                    eligible_nodes.remove(src_node)
                if G.degree(dest_node) == clique_degree and dest_node in eligible_nodes:
                    eligible_nodes.remove(dest_node)
        elif p <= a_probability < p + q and m <= G.size() < clique_size:
            eligible_nodes = [nd for (nd, deg) in G.degree() if 0 < deg < clique_degree]
            for i in range(m):
                node = seed.choice(eligible_nodes)
                neighbor_nodes = list(G[node])
                src_node = seed.choice(neighbor_nodes)
                neighbor_nodes.append(node)
                dest_node = seed.choice([nd for nd in attachment_preference if nd not in neighbor_nodes])
                G.remove_edge(node, src_node)
                G.add_edge(node, dest_node)
                attachment_preference.remove(src_node)
                attachment_preference.append(dest_node)
                if G.degree(src_node) == 0 and src_node in eligible_nodes:
                    eligible_nodes.remove(src_node)
                if dest_node in eligible_nodes:
                    if G.degree(dest_node) == clique_degree:
                        eligible_nodes.remove(dest_node)
                elif G.degree(dest_node) == 1:
                    eligible_nodes.append(dest_node)
        else:
            targets = _random_subset(attachment_preference, m, seed)
            G.add_edges_from(zip([new_node] * m, targets))
            attachment_preference.extend(targets)
            attachment_preference.extend([new_node] * (m + 1))
            new_node += 1
    return G

@py_random_state(3)
@nx._dispatch(graphs=None)
def powerlaw_cluster_graph(n, m, p, seed=None):
    if False:
        for i in range(10):
            print('nop')
    'Holme and Kim algorithm for growing graphs with powerlaw\n    degree distribution and approximate average clustering.\n\n    Parameters\n    ----------\n    n : int\n        the number of nodes\n    m : int\n        the number of random edges to add for each new node\n    p : float,\n        Probability of adding a triangle after adding a random edge\n    seed : integer, random_state, or None (default)\n        Indicator of random number generation state.\n        See :ref:`Randomness<randomness>`.\n\n    Notes\n    -----\n    The average clustering has a hard time getting above a certain\n    cutoff that depends on `m`.  This cutoff is often quite low.  The\n    transitivity (fraction of triangles to possible triangles) seems to\n    decrease with network size.\n\n    It is essentially the Barabási–Albert (BA) growth model with an\n    extra step that each random edge is followed by a chance of\n    making an edge to one of its neighbors too (and thus a triangle).\n\n    This algorithm improves on BA in the sense that it enables a\n    higher average clustering to be attained if desired.\n\n    It seems possible to have a disconnected graph with this algorithm\n    since the initial `m` nodes may not be all linked to a new node\n    on the first iteration like the BA model.\n\n    Raises\n    ------\n    NetworkXError\n        If `m` does not satisfy ``1 <= m <= n`` or `p` does not\n        satisfy ``0 <= p <= 1``.\n\n    References\n    ----------\n    .. [1] P. Holme and B. J. Kim,\n       "Growing scale-free networks with tunable clustering",\n       Phys. Rev. E, 65, 026107, 2002.\n    '
    if m < 1 or n < m:
        raise nx.NetworkXError(f'NetworkXError must have m>1 and m<n, m={m},n={n}')
    if p > 1 or p < 0:
        raise nx.NetworkXError(f'NetworkXError p must be in [0,1], p={p}')
    G = empty_graph(m)
    repeated_nodes = list(G.nodes())
    source = m
    while source < n:
        possible_targets = _random_subset(repeated_nodes, m, seed)
        target = possible_targets.pop()
        G.add_edge(source, target)
        repeated_nodes.append(target)
        count = 1
        while count < m:
            if seed.random() < p:
                neighborhood = [nbr for nbr in G.neighbors(target) if not G.has_edge(source, nbr) and nbr != source]
                if neighborhood:
                    nbr = seed.choice(neighborhood)
                    G.add_edge(source, nbr)
                    repeated_nodes.append(nbr)
                    count = count + 1
                    continue
            target = possible_targets.pop()
            G.add_edge(source, target)
            repeated_nodes.append(target)
            count = count + 1
        repeated_nodes.extend([source] * m)
        source += 1
    return G

@py_random_state(3)
@nx._dispatch(graphs=None)
def random_lobster(n, p1, p2, seed=None):
    if False:
        print('Hello World!')
    'Returns a random lobster graph.\n\n    A lobster is a tree that reduces to a caterpillar when pruning all\n    leaf nodes. A caterpillar is a tree that reduces to a path graph\n    when pruning all leaf nodes; setting `p2` to zero produces a caterpillar.\n\n    This implementation iterates on the probabilities `p1` and `p2` to add\n    edges at levels 1 and 2, respectively. Graphs are therefore constructed\n    iteratively with uniform randomness at each level rather than being selected\n    uniformly at random from the set of all possible lobsters.\n\n    Parameters\n    ----------\n    n : int\n        The expected number of nodes in the backbone\n    p1 : float\n        Probability of adding an edge to the backbone\n    p2 : float\n        Probability of adding an edge one level beyond backbone\n    seed : integer, random_state, or None (default)\n        Indicator of random number generation state.\n        See :ref:`Randomness<randomness>`.\n\n    Raises\n    ------\n    NetworkXError\n        If `p1` or `p2` parameters are >= 1 because the while loops would never finish.\n    '
    (p1, p2) = (abs(p1), abs(p2))
    if any((p >= 1 for p in [p1, p2])):
        raise nx.NetworkXError('Probability values for `p1` and `p2` must both be < 1.')
    llen = int(2 * seed.random() * n + 0.5)
    L = path_graph(llen)
    current_node = llen - 1
    for n in range(llen):
        while seed.random() < p1:
            current_node += 1
            L.add_edge(n, current_node)
            cat_node = current_node
            while seed.random() < p2:
                current_node += 1
                L.add_edge(cat_node, current_node)
    return L

@py_random_state(1)
@nx._dispatch(graphs=None)
def random_shell_graph(constructor, seed=None):
    if False:
        while True:
            i = 10
    'Returns a random shell graph for the constructor given.\n\n    Parameters\n    ----------\n    constructor : list of three-tuples\n        Represents the parameters for a shell, starting at the center\n        shell.  Each element of the list must be of the form `(n, m,\n        d)`, where `n` is the number of nodes in the shell, `m` is\n        the number of edges in the shell, and `d` is the ratio of\n        inter-shell (next) edges to intra-shell edges. If `d` is zero,\n        there will be no intra-shell edges, and if `d` is one there\n        will be all possible intra-shell edges.\n    seed : integer, random_state, or None (default)\n        Indicator of random number generation state.\n        See :ref:`Randomness<randomness>`.\n\n    Examples\n    --------\n    >>> constructor = [(10, 20, 0.8), (20, 40, 0.8)]\n    >>> G = nx.random_shell_graph(constructor)\n\n    '
    G = empty_graph(0)
    glist = []
    intra_edges = []
    nnodes = 0
    for (n, m, d) in constructor:
        inter_edges = int(m * d)
        intra_edges.append(m - inter_edges)
        g = nx.convert_node_labels_to_integers(gnm_random_graph(n, inter_edges, seed=seed), first_label=nnodes)
        glist.append(g)
        nnodes += n
        G = nx.operators.union(G, g)
    for gi in range(len(glist) - 1):
        nlist1 = list(glist[gi])
        nlist2 = list(glist[gi + 1])
        total_edges = intra_edges[gi]
        edge_count = 0
        while edge_count < total_edges:
            u = seed.choice(nlist1)
            v = seed.choice(nlist2)
            if u == v or G.has_edge(u, v):
                continue
            else:
                G.add_edge(u, v)
                edge_count = edge_count + 1
    return G

@py_random_state(2)
@nx._dispatch(graphs=None)
def random_powerlaw_tree(n, gamma=3, seed=None, tries=100):
    if False:
        while True:
            i = 10
    'Returns a tree with a power law degree distribution.\n\n    Parameters\n    ----------\n    n : int\n        The number of nodes.\n    gamma : float\n        Exponent of the power law.\n    seed : integer, random_state, or None (default)\n        Indicator of random number generation state.\n        See :ref:`Randomness<randomness>`.\n    tries : int\n        Number of attempts to adjust the sequence to make it a tree.\n\n    Raises\n    ------\n    NetworkXError\n        If no valid sequence is found within the maximum number of\n        attempts.\n\n    Notes\n    -----\n    A trial power law degree sequence is chosen and then elements are\n    swapped with new elements from a powerlaw distribution until the\n    sequence makes a tree (by checking, for example, that the number of\n    edges is one smaller than the number of nodes).\n\n    '
    seq = random_powerlaw_tree_sequence(n, gamma=gamma, seed=seed, tries=tries)
    G = degree_sequence_tree(seq)
    return G

@py_random_state(2)
@nx._dispatch(graphs=None)
def random_powerlaw_tree_sequence(n, gamma=3, seed=None, tries=100):
    if False:
        i = 10
        return i + 15
    'Returns a degree sequence for a tree with a power law distribution.\n\n    Parameters\n    ----------\n    n : int,\n        The number of nodes.\n    gamma : float\n        Exponent of the power law.\n    seed : integer, random_state, or None (default)\n        Indicator of random number generation state.\n        See :ref:`Randomness<randomness>`.\n    tries : int\n        Number of attempts to adjust the sequence to make it a tree.\n\n    Raises\n    ------\n    NetworkXError\n        If no valid sequence is found within the maximum number of\n        attempts.\n\n    Notes\n    -----\n    A trial power law degree sequence is chosen and then elements are\n    swapped with new elements from a power law distribution until\n    the sequence makes a tree (by checking, for example, that the number of\n    edges is one smaller than the number of nodes).\n\n    '
    z = nx.utils.powerlaw_sequence(n, exponent=gamma, seed=seed)
    zseq = [min(n, max(round(s), 0)) for s in z]
    z = nx.utils.powerlaw_sequence(tries, exponent=gamma, seed=seed)
    swap = [min(n, max(round(s), 0)) for s in z]
    for deg in swap:
        if 2 * n - sum(zseq) == 2:
            return zseq
        index = seed.randint(0, n - 1)
        zseq[index] = swap.pop()
    raise nx.NetworkXError(f'Exceeded max ({tries}) attempts for a valid tree sequence.')

@py_random_state(3)
@nx._dispatch(graphs=None)
def random_kernel_graph(n, kernel_integral, kernel_root=None, seed=None):
    if False:
        i = 10
        return i + 15
    'Returns an random graph based on the specified kernel.\n\n    The algorithm chooses each of the $[n(n-1)]/2$ possible edges with\n    probability specified by a kernel $\\kappa(x,y)$ [1]_.  The kernel\n    $\\kappa(x,y)$ must be a symmetric (in $x,y$), non-negative,\n    bounded function.\n\n    Parameters\n    ----------\n    n : int\n        The number of nodes\n    kernel_integral : function\n        Function that returns the definite integral of the kernel $\\kappa(x,y)$,\n        $F(y,a,b) := \\int_a^b \\kappa(x,y)dx$\n    kernel_root: function (optional)\n        Function that returns the root $b$ of the equation $F(y,a,b) = r$.\n        If None, the root is found using :func:`scipy.optimize.brentq`\n        (this requires SciPy).\n    seed : integer, random_state, or None (default)\n        Indicator of random number generation state.\n        See :ref:`Randomness<randomness>`.\n\n    Notes\n    -----\n    The kernel is specified through its definite integral which must be\n    provided as one of the arguments. If the integral and root of the\n    kernel integral can be found in $O(1)$ time then this algorithm runs in\n    time $O(n+m)$ where m is the expected number of edges [2]_.\n\n    The nodes are set to integers from $0$ to $n-1$.\n\n    Examples\n    --------\n    Generate an Erdős–Rényi random graph $G(n,c/n)$, with kernel\n    $\\kappa(x,y)=c$ where $c$ is the mean expected degree.\n\n    >>> def integral(u, w, z):\n    ...     return c * (z - w)\n    >>> def root(u, w, r):\n    ...     return r / c + w\n    >>> c = 1\n    >>> graph = nx.random_kernel_graph(1000, integral, root)\n\n    See Also\n    --------\n    gnp_random_graph\n    expected_degree_graph\n\n    References\n    ----------\n    .. [1] Bollobás, Béla,  Janson, S. and Riordan, O.\n       "The phase transition in inhomogeneous random graphs",\n       *Random Structures Algorithms*, 31, 3--122, 2007.\n\n    .. [2] Hagberg A, Lemons N (2015),\n       "Fast Generation of Sparse Random Kernel Graphs".\n       PLoS ONE 10(9): e0135177, 2015. doi:10.1371/journal.pone.0135177\n    '
    if kernel_root is None:
        import scipy as sp

        def kernel_root(y, a, r):
            if False:
                print('Hello World!')

            def my_function(b):
                if False:
                    for i in range(10):
                        print('nop')
                return kernel_integral(y, a, b) - r
            return sp.optimize.brentq(my_function, a, 1)
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    (i, j) = (1, 1)
    while i < n:
        r = -math.log(1 - seed.random())
        if kernel_integral(i / n, j / n, 1) <= r:
            (i, j) = (i + 1, i + 1)
        else:
            j = math.ceil(n * kernel_root(i / n, j / n, r))
            graph.add_edge(i - 1, j - 1)
    return graph