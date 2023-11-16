"""Generators for classes of graphs used in studying social networks."""
import itertools
import math
import networkx as nx
from networkx.utils import py_random_state
__all__ = ['caveman_graph', 'connected_caveman_graph', 'relaxed_caveman_graph', 'random_partition_graph', 'planted_partition_graph', 'gaussian_random_partition_graph', 'ring_of_cliques', 'windmill_graph', 'stochastic_block_model', 'LFR_benchmark_graph']

@nx._dispatch(graphs=None)
def caveman_graph(l, k):
    if False:
        print('Hello World!')
    "Returns a caveman graph of `l` cliques of size `k`.\n\n    Parameters\n    ----------\n    l : int\n      Number of cliques\n    k : int\n      Size of cliques\n\n    Returns\n    -------\n    G : NetworkX Graph\n      caveman graph\n\n    Notes\n    -----\n    This returns an undirected graph, it can be converted to a directed\n    graph using :func:`nx.to_directed`, or a multigraph using\n    ``nx.MultiGraph(nx.caveman_graph(l, k))``. Only the undirected version is\n    described in [1]_ and it is unclear which of the directed\n    generalizations is most useful.\n\n    Examples\n    --------\n    >>> G = nx.caveman_graph(3, 3)\n\n    See also\n    --------\n\n    connected_caveman_graph\n\n    References\n    ----------\n    .. [1] Watts, D. J. 'Networks, Dynamics, and the Small-World Phenomenon.'\n       Amer. J. Soc. 105, 493-527, 1999.\n    "
    G = nx.empty_graph(l * k)
    if k > 1:
        for start in range(0, l * k, k):
            edges = itertools.combinations(range(start, start + k), 2)
            G.add_edges_from(edges)
    return G

@nx._dispatch(graphs=None)
def connected_caveman_graph(l, k):
    if False:
        while True:
            i = 10
    "Returns a connected caveman graph of `l` cliques of size `k`.\n\n    The connected caveman graph is formed by creating `n` cliques of size\n    `k`, then a single edge in each clique is rewired to a node in an\n    adjacent clique.\n\n    Parameters\n    ----------\n    l : int\n      number of cliques\n    k : int\n      size of cliques (k at least 2 or NetworkXError is raised)\n\n    Returns\n    -------\n    G : NetworkX Graph\n      connected caveman graph\n\n    Raises\n    ------\n    NetworkXError\n        If the size of cliques `k` is smaller than 2.\n\n    Notes\n    -----\n    This returns an undirected graph, it can be converted to a directed\n    graph using :func:`nx.to_directed`, or a multigraph using\n    ``nx.MultiGraph(nx.caveman_graph(l, k))``. Only the undirected version is\n    described in [1]_ and it is unclear which of the directed\n    generalizations is most useful.\n\n    Examples\n    --------\n    >>> G = nx.connected_caveman_graph(3, 3)\n\n    References\n    ----------\n    .. [1] Watts, D. J. 'Networks, Dynamics, and the Small-World Phenomenon.'\n       Amer. J. Soc. 105, 493-527, 1999.\n    "
    if k < 2:
        raise nx.NetworkXError('The size of cliques in a connected caveman graph must be at least 2.')
    G = nx.caveman_graph(l, k)
    for start in range(0, l * k, k):
        G.remove_edge(start, start + 1)
        G.add_edge(start, (start - 1) % (l * k))
    return G

@py_random_state(3)
@nx._dispatch(graphs=None)
def relaxed_caveman_graph(l, k, p, seed=None):
    if False:
        return 10
    'Returns a relaxed caveman graph.\n\n    A relaxed caveman graph starts with `l` cliques of size `k`.  Edges are\n    then randomly rewired with probability `p` to link different cliques.\n\n    Parameters\n    ----------\n    l : int\n      Number of groups\n    k : int\n      Size of cliques\n    p : float\n      Probability of rewiring each edge.\n    seed : integer, random_state, or None (default)\n        Indicator of random number generation state.\n        See :ref:`Randomness<randomness>`.\n\n    Returns\n    -------\n    G : NetworkX Graph\n      Relaxed Caveman Graph\n\n    Raises\n    ------\n    NetworkXError\n     If p is not in [0,1]\n\n    Examples\n    --------\n    >>> G = nx.relaxed_caveman_graph(2, 3, 0.1, seed=42)\n\n    References\n    ----------\n    .. [1] Santo Fortunato, Community Detection in Graphs,\n       Physics Reports Volume 486, Issues 3-5, February 2010, Pages 75-174.\n       https://arxiv.org/abs/0906.0612\n    '
    G = nx.caveman_graph(l, k)
    nodes = list(G)
    for (u, v) in G.edges():
        if seed.random() < p:
            x = seed.choice(nodes)
            if G.has_edge(u, x):
                continue
            G.remove_edge(u, v)
            G.add_edge(u, x)
    return G

@py_random_state(3)
@nx._dispatch(graphs=None)
def random_partition_graph(sizes, p_in, p_out, seed=None, directed=False):
    if False:
        while True:
            i = 10
    'Returns the random partition graph with a partition of sizes.\n\n    A partition graph is a graph of communities with sizes defined by\n    s in sizes. Nodes in the same group are connected with probability\n    p_in and nodes of different groups are connected with probability\n    p_out.\n\n    Parameters\n    ----------\n    sizes : list of ints\n      Sizes of groups\n    p_in : float\n      probability of edges with in groups\n    p_out : float\n      probability of edges between groups\n    directed : boolean optional, default=False\n      Whether to create a directed graph\n    seed : integer, random_state, or None (default)\n        Indicator of random number generation state.\n        See :ref:`Randomness<randomness>`.\n\n    Returns\n    -------\n    G : NetworkX Graph or DiGraph\n      random partition graph of size sum(gs)\n\n    Raises\n    ------\n    NetworkXError\n      If p_in or p_out is not in [0,1]\n\n    Examples\n    --------\n    >>> G = nx.random_partition_graph([10, 10, 10], 0.25, 0.01)\n    >>> len(G)\n    30\n    >>> partition = G.graph["partition"]\n    >>> len(partition)\n    3\n\n    Notes\n    -----\n    This is a generalization of the planted-l-partition described in\n    [1]_.  It allows for the creation of groups of any size.\n\n    The partition is store as a graph attribute \'partition\'.\n\n    References\n    ----------\n    .. [1] Santo Fortunato \'Community Detection in Graphs\' Physical Reports\n       Volume 486, Issue 3-5 p. 75-174. https://arxiv.org/abs/0906.0612\n    '
    if not 0.0 <= p_in <= 1.0:
        raise nx.NetworkXError('p_in must be in [0,1]')
    if not 0.0 <= p_out <= 1.0:
        raise nx.NetworkXError('p_out must be in [0,1]')
    num_blocks = len(sizes)
    p = [[p_out for s in range(num_blocks)] for r in range(num_blocks)]
    for r in range(num_blocks):
        p[r][r] = p_in
    return stochastic_block_model(sizes, p, nodelist=None, seed=seed, directed=directed, selfloops=False, sparse=True)

@py_random_state(4)
@nx._dispatch(graphs=None)
def planted_partition_graph(l, k, p_in, p_out, seed=None, directed=False):
    if False:
        print('Hello World!')
    "Returns the planted l-partition graph.\n\n    This model partitions a graph with n=l*k vertices in\n    l groups with k vertices each. Vertices of the same\n    group are linked with a probability p_in, and vertices\n    of different groups are linked with probability p_out.\n\n    Parameters\n    ----------\n    l : int\n      Number of groups\n    k : int\n      Number of vertices in each group\n    p_in : float\n      probability of connecting vertices within a group\n    p_out : float\n      probability of connected vertices between groups\n    seed : integer, random_state, or None (default)\n        Indicator of random number generation state.\n        See :ref:`Randomness<randomness>`.\n    directed : bool,optional (default=False)\n      If True return a directed graph\n\n    Returns\n    -------\n    G : NetworkX Graph or DiGraph\n      planted l-partition graph\n\n    Raises\n    ------\n    NetworkXError\n      If p_in,p_out are not in [0,1] or\n\n    Examples\n    --------\n    >>> G = nx.planted_partition_graph(4, 3, 0.5, 0.1, seed=42)\n\n    See Also\n    --------\n    random_partition_model\n\n    References\n    ----------\n    .. [1] A. Condon, R.M. Karp, Algorithms for graph partitioning\n        on the planted partition model,\n        Random Struct. Algor. 18 (2001) 116-140.\n\n    .. [2] Santo Fortunato 'Community Detection in Graphs' Physical Reports\n       Volume 486, Issue 3-5 p. 75-174. https://arxiv.org/abs/0906.0612\n    "
    return random_partition_graph([k] * l, p_in, p_out, seed=seed, directed=directed)

@py_random_state(6)
@nx._dispatch(graphs=None)
def gaussian_random_partition_graph(n, s, v, p_in, p_out, directed=False, seed=None):
    if False:
        while True:
            i = 10
    'Generate a Gaussian random partition graph.\n\n    A Gaussian random partition graph is created by creating k partitions\n    each with a size drawn from a normal distribution with mean s and variance\n    s/v. Nodes are connected within clusters with probability p_in and\n    between clusters with probability p_out[1]\n\n    Parameters\n    ----------\n    n : int\n      Number of nodes in the graph\n    s : float\n      Mean cluster size\n    v : float\n      Shape parameter. The variance of cluster size distribution is s/v.\n    p_in : float\n      Probability of intra cluster connection.\n    p_out : float\n      Probability of inter cluster connection.\n    directed : boolean, optional default=False\n      Whether to create a directed graph or not\n    seed : integer, random_state, or None (default)\n        Indicator of random number generation state.\n        See :ref:`Randomness<randomness>`.\n\n    Returns\n    -------\n    G : NetworkX Graph or DiGraph\n      gaussian random partition graph\n\n    Raises\n    ------\n    NetworkXError\n      If s is > n\n      If p_in or p_out is not in [0,1]\n\n    Notes\n    -----\n    Note the number of partitions is dependent on s,v and n, and that the\n    last partition may be considerably smaller, as it is sized to simply\n    fill out the nodes [1]\n\n    See Also\n    --------\n    random_partition_graph\n\n    Examples\n    --------\n    >>> G = nx.gaussian_random_partition_graph(100, 10, 10, 0.25, 0.1)\n    >>> len(G)\n    100\n\n    References\n    ----------\n    .. [1] Ulrik Brandes, Marco Gaertler, Dorothea Wagner,\n       Experiments on Graph Clustering Algorithms,\n       In the proceedings of the 11th Europ. Symp. Algorithms, 2003.\n    '
    if s > n:
        raise nx.NetworkXError('s must be <= n')
    assigned = 0
    sizes = []
    while True:
        size = int(seed.gauss(s, s / v + 0.5))
        if size < 1:
            continue
        if assigned + size >= n:
            sizes.append(n - assigned)
            break
        assigned += size
        sizes.append(size)
    return random_partition_graph(sizes, p_in, p_out, seed=seed, directed=directed)

@nx._dispatch(graphs=None)
def ring_of_cliques(num_cliques, clique_size):
    if False:
        return 10
    'Defines a "ring of cliques" graph.\n\n    A ring of cliques graph is consisting of cliques, connected through single\n    links. Each clique is a complete graph.\n\n    Parameters\n    ----------\n    num_cliques : int\n        Number of cliques\n    clique_size : int\n        Size of cliques\n\n    Returns\n    -------\n    G : NetworkX Graph\n        ring of cliques graph\n\n    Raises\n    ------\n    NetworkXError\n        If the number of cliques is lower than 2 or\n        if the size of cliques is smaller than 2.\n\n    Examples\n    --------\n    >>> G = nx.ring_of_cliques(8, 4)\n\n    See Also\n    --------\n    connected_caveman_graph\n\n    Notes\n    -----\n    The `connected_caveman_graph` graph removes a link from each clique to\n    connect it with the next clique. Instead, the `ring_of_cliques` graph\n    simply adds the link without removing any link from the cliques.\n    '
    if num_cliques < 2:
        raise nx.NetworkXError('A ring of cliques must have at least two cliques')
    if clique_size < 2:
        raise nx.NetworkXError('The cliques must have at least two nodes')
    G = nx.Graph()
    for i in range(num_cliques):
        edges = itertools.combinations(range(i * clique_size, i * clique_size + clique_size), 2)
        G.add_edges_from(edges)
        G.add_edge(i * clique_size + 1, (i + 1) * clique_size % (num_cliques * clique_size))
    return G

@nx._dispatch(graphs=None)
def windmill_graph(n, k):
    if False:
        return 10
    'Generate a windmill graph.\n    A windmill graph is a graph of `n` cliques each of size `k` that are all\n    joined at one node.\n    It can be thought of as taking a disjoint union of `n` cliques of size `k`,\n    selecting one point from each, and contracting all of the selected points.\n    Alternatively, one could generate `n` cliques of size `k-1` and one node\n    that is connected to all other nodes in the graph.\n\n    Parameters\n    ----------\n    n : int\n        Number of cliques\n    k : int\n        Size of cliques\n\n    Returns\n    -------\n    G : NetworkX Graph\n        windmill graph with n cliques of size k\n\n    Raises\n    ------\n    NetworkXError\n        If the number of cliques is less than two\n        If the size of the cliques are less than two\n\n    Examples\n    --------\n    >>> G = nx.windmill_graph(4, 5)\n\n    Notes\n    -----\n    The node labeled `0` will be the node connected to all other nodes.\n    Note that windmill graphs are usually denoted `Wd(k,n)`, so the parameters\n    are in the opposite order as the parameters of this method.\n    '
    if n < 2:
        msg = 'A windmill graph must have at least two cliques'
        raise nx.NetworkXError(msg)
    if k < 2:
        raise nx.NetworkXError('The cliques must have at least two nodes')
    G = nx.disjoint_union_all(itertools.chain([nx.complete_graph(k)], (nx.complete_graph(k - 1) for _ in range(n - 1))))
    G.add_edges_from(((0, i) for i in range(k, G.number_of_nodes())))
    return G

@py_random_state(3)
@nx._dispatch(graphs=None)
def stochastic_block_model(sizes, p, nodelist=None, seed=None, directed=False, selfloops=False, sparse=True):
    if False:
        for i in range(10):
            print('nop')
    'Returns a stochastic block model graph.\n\n    This model partitions the nodes in blocks of arbitrary sizes, and places\n    edges between pairs of nodes independently, with a probability that depends\n    on the blocks.\n\n    Parameters\n    ----------\n    sizes : list of ints\n        Sizes of blocks\n    p : list of list of floats\n        Element (r,s) gives the density of edges going from the nodes\n        of group r to nodes of group s.\n        p must match the number of groups (len(sizes) == len(p)),\n        and it must be symmetric if the graph is undirected.\n    nodelist : list, optional\n        The block tags are assigned according to the node identifiers\n        in nodelist. If nodelist is None, then the ordering is the\n        range [0,sum(sizes)-1].\n    seed : integer, random_state, or None (default)\n        Indicator of random number generation state.\n        See :ref:`Randomness<randomness>`.\n    directed : boolean optional, default=False\n        Whether to create a directed graph or not.\n    selfloops : boolean optional, default=False\n        Whether to include self-loops or not.\n    sparse: boolean optional, default=True\n        Use the sparse heuristic to speed up the generator.\n\n    Returns\n    -------\n    g : NetworkX Graph or DiGraph\n        Stochastic block model graph of size sum(sizes)\n\n    Raises\n    ------\n    NetworkXError\n      If probabilities are not in [0,1].\n      If the probability matrix is not square (directed case).\n      If the probability matrix is not symmetric (undirected case).\n      If the sizes list does not match nodelist or the probability matrix.\n      If nodelist contains duplicate.\n\n    Examples\n    --------\n    >>> sizes = [75, 75, 300]\n    >>> probs = [[0.25, 0.05, 0.02], [0.05, 0.35, 0.07], [0.02, 0.07, 0.40]]\n    >>> g = nx.stochastic_block_model(sizes, probs, seed=0)\n    >>> len(g)\n    450\n    >>> H = nx.quotient_graph(g, g.graph["partition"], relabel=True)\n    >>> for v in H.nodes(data=True):\n    ...     print(round(v[1]["density"], 3))\n    ...\n    0.245\n    0.348\n    0.405\n    >>> for v in H.edges(data=True):\n    ...     print(round(1.0 * v[2]["weight"] / (sizes[v[0]] * sizes[v[1]]), 3))\n    ...\n    0.051\n    0.022\n    0.07\n\n    See Also\n    --------\n    random_partition_graph\n    planted_partition_graph\n    gaussian_random_partition_graph\n    gnp_random_graph\n\n    References\n    ----------\n    .. [1] Holland, P. W., Laskey, K. B., & Leinhardt, S.,\n           "Stochastic blockmodels: First steps",\n           Social networks, 5(2), 109-137, 1983.\n    '
    if len(sizes) != len(p):
        raise nx.NetworkXException("'sizes' and 'p' do not match.")
    for row in p:
        if len(p) != len(row):
            raise nx.NetworkXException("'p' must be a square matrix.")
    if not directed:
        p_transpose = [list(i) for i in zip(*p)]
        for i in zip(p, p_transpose):
            for j in zip(i[0], i[1]):
                if abs(j[0] - j[1]) > 1e-08:
                    raise nx.NetworkXException("'p' must be symmetric.")
    for row in p:
        for prob in row:
            if prob < 0 or prob > 1:
                raise nx.NetworkXException("Entries of 'p' not in [0,1].")
    if nodelist is not None:
        if len(nodelist) != sum(sizes):
            raise nx.NetworkXException("'nodelist' and 'sizes' do not match.")
        if len(nodelist) != len(set(nodelist)):
            raise nx.NetworkXException('nodelist contains duplicate.')
    else:
        nodelist = range(sum(sizes))
    block_range = range(len(sizes))
    if directed:
        g = nx.DiGraph()
        block_iter = itertools.product(block_range, block_range)
    else:
        g = nx.Graph()
        block_iter = itertools.combinations_with_replacement(block_range, 2)
    size_cumsum = [sum(sizes[0:x]) for x in range(len(sizes) + 1)]
    g.graph['partition'] = [set(nodelist[size_cumsum[x]:size_cumsum[x + 1]]) for x in range(len(size_cumsum) - 1)]
    for (block_id, nodes) in enumerate(g.graph['partition']):
        for node in nodes:
            g.add_node(node, block=block_id)
    g.name = 'stochastic_block_model'
    parts = g.graph['partition']
    for (i, j) in block_iter:
        if i == j:
            if directed:
                if selfloops:
                    edges = itertools.product(parts[i], parts[i])
                else:
                    edges = itertools.permutations(parts[i], 2)
            else:
                edges = itertools.combinations(parts[i], 2)
                if selfloops:
                    edges = itertools.chain(edges, zip(parts[i], parts[i]))
            for e in edges:
                if seed.random() < p[i][j]:
                    g.add_edge(*e)
        else:
            edges = itertools.product(parts[i], parts[j])
        if sparse:
            if p[i][j] == 1:
                for e in edges:
                    g.add_edge(*e)
            elif p[i][j] > 0:
                while True:
                    try:
                        logrand = math.log(seed.random())
                        skip = math.floor(logrand / math.log(1 - p[i][j]))
                        next(itertools.islice(edges, skip, skip), None)
                        e = next(edges)
                        g.add_edge(*e)
                    except StopIteration:
                        break
        else:
            for e in edges:
                if seed.random() < p[i][j]:
                    g.add_edge(*e)
    return g

def _zipf_rv_below(gamma, xmin, threshold, seed):
    if False:
        return 10
    'Returns a random value chosen from the bounded Zipf distribution.\n\n    Repeatedly draws values from the Zipf distribution until the\n    threshold is met, then returns that value.\n    '
    result = nx.utils.zipf_rv(gamma, xmin, seed)
    while result > threshold:
        result = nx.utils.zipf_rv(gamma, xmin, seed)
    return result

def _powerlaw_sequence(gamma, low, high, condition, length, max_iters, seed):
    if False:
        for i in range(10):
            print('nop')
    'Returns a list of numbers obeying a constrained power law distribution.\n\n    ``gamma`` and ``low`` are the parameters for the Zipf distribution.\n\n    ``high`` is the maximum allowed value for values draw from the Zipf\n    distribution. For more information, see :func:`_zipf_rv_below`.\n\n    ``condition`` and ``length`` are Boolean-valued functions on\n    lists. While generating the list, random values are drawn and\n    appended to the list until ``length`` is satisfied by the created\n    list. Once ``condition`` is satisfied, the sequence generated in\n    this way is returned.\n\n    ``max_iters`` indicates the number of times to generate a list\n    satisfying ``length``. If the number of iterations exceeds this\n    value, :exc:`~networkx.exception.ExceededMaxIterations` is raised.\n\n    seed : integer, random_state, or None (default)\n        Indicator of random number generation state.\n        See :ref:`Randomness<randomness>`.\n    '
    for i in range(max_iters):
        seq = []
        while not length(seq):
            seq.append(_zipf_rv_below(gamma, low, high, seed))
        if condition(seq):
            return seq
    raise nx.ExceededMaxIterations('Could not create power law sequence')

def _hurwitz_zeta(x, q, tolerance):
    if False:
        i = 10
        return i + 15
    'The Hurwitz zeta function, or the Riemann zeta function of two arguments.\n\n    ``x`` must be greater than one and ``q`` must be positive.\n\n    This function repeatedly computes subsequent partial sums until\n    convergence, as decided by ``tolerance``.\n    '
    z = 0
    z_prev = -float('inf')
    k = 0
    while abs(z - z_prev) > tolerance:
        z_prev = z
        z += 1 / (k + q) ** x
        k += 1
    return z

def _generate_min_degree(gamma, average_degree, max_degree, tolerance, max_iters):
    if False:
        for i in range(10):
            print('nop')
    'Returns a minimum degree from the given average degree.'
    try:
        from scipy.special import zeta
    except ImportError:

        def zeta(x, q):
            if False:
                for i in range(10):
                    print('nop')
            return _hurwitz_zeta(x, q, tolerance)
    min_deg_top = max_degree
    min_deg_bot = 1
    min_deg_mid = (min_deg_top - min_deg_bot) / 2 + min_deg_bot
    itrs = 0
    mid_avg_deg = 0
    while abs(mid_avg_deg - average_degree) > tolerance:
        if itrs > max_iters:
            raise nx.ExceededMaxIterations('Could not match average_degree')
        mid_avg_deg = 0
        for x in range(int(min_deg_mid), max_degree + 1):
            mid_avg_deg += x ** (-gamma + 1) / zeta(gamma, min_deg_mid)
        if mid_avg_deg > average_degree:
            min_deg_top = min_deg_mid
            min_deg_mid = (min_deg_top - min_deg_bot) / 2 + min_deg_bot
        else:
            min_deg_bot = min_deg_mid
            min_deg_mid = (min_deg_top - min_deg_bot) / 2 + min_deg_bot
        itrs += 1
    return round(min_deg_mid)

def _generate_communities(degree_seq, community_sizes, mu, max_iters, seed):
    if False:
        for i in range(10):
            print('nop')
    'Returns a list of sets, each of which represents a community.\n\n    ``degree_seq`` is the degree sequence that must be met by the\n    graph.\n\n    ``community_sizes`` is the community size distribution that must be\n    met by the generated list of sets.\n\n    ``mu`` is a float in the interval [0, 1] indicating the fraction of\n    intra-community edges incident to each node.\n\n    ``max_iters`` is the number of times to try to add a node to a\n    community. This must be greater than the length of\n    ``degree_seq``, otherwise this function will always fail. If\n    the number of iterations exceeds this value,\n    :exc:`~networkx.exception.ExceededMaxIterations` is raised.\n\n    seed : integer, random_state, or None (default)\n        Indicator of random number generation state.\n        See :ref:`Randomness<randomness>`.\n\n    The communities returned by this are sets of integers in the set {0,\n    ..., *n* - 1}, where *n* is the length of ``degree_seq``.\n\n    '
    result = [set() for _ in community_sizes]
    n = len(degree_seq)
    free = list(range(n))
    for i in range(max_iters):
        v = free.pop()
        c = seed.choice(range(len(community_sizes)))
        s = round(degree_seq[v] * (1 - mu))
        if s < community_sizes[c]:
            result[c].add(v)
        else:
            free.append(v)
        if len(result[c]) > community_sizes[c]:
            free.append(result[c].pop())
        if not free:
            return result
    msg = 'Could not assign communities; try increasing min_community'
    raise nx.ExceededMaxIterations(msg)

@py_random_state(11)
@nx._dispatch(graphs=None)
def LFR_benchmark_graph(n, tau1, tau2, mu, average_degree=None, min_degree=None, max_degree=None, min_community=None, max_community=None, tol=1e-07, max_iters=500, seed=None):
    if False:
        for i in range(10):
            print('nop')
    'Returns the LFR benchmark graph.\n\n    This algorithm proceeds as follows:\n\n    1) Find a degree sequence with a power law distribution, and minimum\n       value ``min_degree``, which has approximate average degree\n       ``average_degree``. This is accomplished by either\n\n       a) specifying ``min_degree`` and not ``average_degree``,\n       b) specifying ``average_degree`` and not ``min_degree``, in which\n          case a suitable minimum degree will be found.\n\n       ``max_degree`` can also be specified, otherwise it will be set to\n       ``n``. Each node *u* will have $\\mu \\mathrm{deg}(u)$ edges\n       joining it to nodes in communities other than its own and $(1 -\n       \\mu) \\mathrm{deg}(u)$ edges joining it to nodes in its own\n       community.\n    2) Generate community sizes according to a power law distribution\n       with exponent ``tau2``. If ``min_community`` and\n       ``max_community`` are not specified they will be selected to be\n       ``min_degree`` and ``max_degree``, respectively.  Community sizes\n       are generated until the sum of their sizes equals ``n``.\n    3) Each node will be randomly assigned a community with the\n       condition that the community is large enough for the node\'s\n       intra-community degree, $(1 - \\mu) \\mathrm{deg}(u)$ as\n       described in step 2. If a community grows too large, a random node\n       will be selected for reassignment to a new community, until all\n       nodes have been assigned a community.\n    4) Each node *u* then adds $(1 - \\mu) \\mathrm{deg}(u)$\n       intra-community edges and $\\mu \\mathrm{deg}(u)$ inter-community\n       edges.\n\n    Parameters\n    ----------\n    n : int\n        Number of nodes in the created graph.\n\n    tau1 : float\n        Power law exponent for the degree distribution of the created\n        graph. This value must be strictly greater than one.\n\n    tau2 : float\n        Power law exponent for the community size distribution in the\n        created graph. This value must be strictly greater than one.\n\n    mu : float\n        Fraction of inter-community edges incident to each node. This\n        value must be in the interval [0, 1].\n\n    average_degree : float\n        Desired average degree of nodes in the created graph. This value\n        must be in the interval [0, *n*]. Exactly one of this and\n        ``min_degree`` must be specified, otherwise a\n        :exc:`NetworkXError` is raised.\n\n    min_degree : int\n        Minimum degree of nodes in the created graph. This value must be\n        in the interval [0, *n*]. Exactly one of this and\n        ``average_degree`` must be specified, otherwise a\n        :exc:`NetworkXError` is raised.\n\n    max_degree : int\n        Maximum degree of nodes in the created graph. If not specified,\n        this is set to ``n``, the total number of nodes in the graph.\n\n    min_community : int\n        Minimum size of communities in the graph. If not specified, this\n        is set to ``min_degree``.\n\n    max_community : int\n        Maximum size of communities in the graph. If not specified, this\n        is set to ``n``, the total number of nodes in the graph.\n\n    tol : float\n        Tolerance when comparing floats, specifically when comparing\n        average degree values.\n\n    max_iters : int\n        Maximum number of iterations to try to create the community sizes,\n        degree distribution, and community affiliations.\n\n    seed : integer, random_state, or None (default)\n        Indicator of random number generation state.\n        See :ref:`Randomness<randomness>`.\n\n    Returns\n    -------\n    G : NetworkX graph\n        The LFR benchmark graph generated according to the specified\n        parameters.\n\n        Each node in the graph has a node attribute ``\'community\'`` that\n        stores the community (that is, the set of nodes) that includes\n        it.\n\n    Raises\n    ------\n    NetworkXError\n        If any of the parameters do not meet their upper and lower bounds:\n\n        - ``tau1`` and ``tau2`` must be strictly greater than 1.\n        - ``mu`` must be in [0, 1].\n        - ``max_degree`` must be in {1, ..., *n*}.\n        - ``min_community`` and ``max_community`` must be in {0, ...,\n          *n*}.\n\n        If not exactly one of ``average_degree`` and ``min_degree`` is\n        specified.\n\n        If ``min_degree`` is not specified and a suitable ``min_degree``\n        cannot be found.\n\n    ExceededMaxIterations\n        If a valid degree sequence cannot be created within\n        ``max_iters`` number of iterations.\n\n        If a valid set of community sizes cannot be created within\n        ``max_iters`` number of iterations.\n\n        If a valid community assignment cannot be created within ``10 *\n        n * max_iters`` number of iterations.\n\n    Examples\n    --------\n    Basic usage::\n\n        >>> from networkx.generators.community import LFR_benchmark_graph\n        >>> n = 250\n        >>> tau1 = 3\n        >>> tau2 = 1.5\n        >>> mu = 0.1\n        >>> G = LFR_benchmark_graph(\n        ...     n, tau1, tau2, mu, average_degree=5, min_community=20, seed=10\n        ... )\n\n    Continuing the example above, you can get the communities from the\n    node attributes of the graph::\n\n        >>> communities = {frozenset(G.nodes[v]["community"]) for v in G}\n\n    Notes\n    -----\n    This algorithm differs slightly from the original way it was\n    presented in [1].\n\n    1) Rather than connecting the graph via a configuration model then\n       rewiring to match the intra-community and inter-community\n       degrees, we do this wiring explicitly at the end, which should be\n       equivalent.\n    2) The code posted on the author\'s website [2] calculates the random\n       power law distributed variables and their average using\n       continuous approximations, whereas we use the discrete\n       distributions here as both degree and community size are\n       discrete.\n\n    Though the authors describe the algorithm as quite robust, testing\n    during development indicates that a somewhat narrower parameter set\n    is likely to successfully produce a graph. Some suggestions have\n    been provided in the event of exceptions.\n\n    References\n    ----------\n    .. [1] "Benchmark graphs for testing community detection algorithms",\n           Andrea Lancichinetti, Santo Fortunato, and Filippo Radicchi,\n           Phys. Rev. E 78, 046110 2008\n    .. [2] https://www.santofortunato.net/resources\n\n    '
    if not tau1 > 1:
        raise nx.NetworkXError('tau1 must be greater than one')
    if not tau2 > 1:
        raise nx.NetworkXError('tau2 must be greater than one')
    if not 0 <= mu <= 1:
        raise nx.NetworkXError('mu must be in the interval [0, 1]')
    if max_degree is None:
        max_degree = n
    elif not 0 < max_degree <= n:
        raise nx.NetworkXError('max_degree must be in the interval (0, n]')
    if not (min_degree is None) ^ (average_degree is None):
        raise nx.NetworkXError('Must assign exactly one of min_degree and average_degree')
    if min_degree is None:
        min_degree = _generate_min_degree(tau1, average_degree, max_degree, tol, max_iters)
    (low, high) = (min_degree, max_degree)

    def condition(seq):
        if False:
            while True:
                i = 10
        return sum(seq) % 2 == 0

    def length(seq):
        if False:
            return 10
        return len(seq) >= n
    deg_seq = _powerlaw_sequence(tau1, low, high, condition, length, max_iters, seed)
    if min_community is None:
        min_community = min(deg_seq)
    if max_community is None:
        max_community = max(deg_seq)
    (low, high) = (min_community, max_community)

    def condition(seq):
        if False:
            while True:
                i = 10
        return sum(seq) == n

    def length(seq):
        if False:
            i = 10
            return i + 15
        return sum(seq) >= n
    comms = _powerlaw_sequence(tau2, low, high, condition, length, max_iters, seed)
    max_iters *= 10 * n
    communities = _generate_communities(deg_seq, comms, mu, max_iters, seed)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for c in communities:
        for u in c:
            while G.degree(u) < round(deg_seq[u] * (1 - mu)):
                v = seed.choice(list(c))
                G.add_edge(u, v)
            while G.degree(u) < deg_seq[u]:
                v = seed.choice(range(n))
                if v not in c:
                    G.add_edge(u, v)
            G.nodes[u]['community'] = c
    return G