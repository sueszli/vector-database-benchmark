"""Functions for estimating the small-world-ness of graphs.

A small world network is characterized by a small average shortest path length,
and a large clustering coefficient.

Small-worldness is commonly measured with the coefficient sigma or omega.

Both coefficients compare the average clustering coefficient and shortest path
length of a given graph against the same quantities for an equivalent random
or lattice graph.

For more information, see the Wikipedia article on small-world network [1]_.

.. [1] Small-world network:: https://en.wikipedia.org/wiki/Small-world_network

"""
import networkx as nx
from networkx.utils import not_implemented_for, py_random_state
__all__ = ['random_reference', 'lattice_reference', 'sigma', 'omega']

@not_implemented_for('directed')
@not_implemented_for('multigraph')
@py_random_state(3)
@nx._dispatch
def random_reference(G, niter=1, connectivity=True, seed=None):
    if False:
        for i in range(10):
            print('nop')
    'Compute a random graph by swapping edges of a given graph.\n\n    Parameters\n    ----------\n    G : graph\n        An undirected graph with 4 or more nodes.\n\n    niter : integer (optional, default=1)\n        An edge is rewired approximately `niter` times.\n\n    connectivity : boolean (optional, default=True)\n        When True, ensure connectivity for the randomized graph.\n\n    seed : integer, random_state, or None (default)\n        Indicator of random number generation state.\n        See :ref:`Randomness<randomness>`.\n\n    Returns\n    -------\n    G : graph\n        The randomized graph.\n\n    Raises\n    ------\n    NetworkXError\n        If there are fewer than 4 nodes or 2 edges in `G`\n\n    Notes\n    -----\n    The implementation is adapted from the algorithm by Maslov and Sneppen\n    (2002) [1]_.\n\n    References\n    ----------\n    .. [1] Maslov, Sergei, and Kim Sneppen.\n           "Specificity and stability in topology of protein networks."\n           Science 296.5569 (2002): 910-913.\n    '
    if len(G) < 4:
        raise nx.NetworkXError('Graph has fewer than four nodes.')
    if len(G.edges) < 2:
        raise nx.NetworkXError('Graph has fewer that 2 edges')
    from networkx.utils import cumulative_distribution, discrete_sequence
    local_conn = nx.connectivity.local_edge_connectivity
    G = G.copy()
    (keys, degrees) = zip(*G.degree())
    cdf = cumulative_distribution(degrees)
    nnodes = len(G)
    nedges = nx.number_of_edges(G)
    niter = niter * nedges
    ntries = int(nnodes * nedges / (nnodes * (nnodes - 1) / 2))
    swapcount = 0
    for i in range(niter):
        n = 0
        while n < ntries:
            (ai, ci) = discrete_sequence(2, cdistribution=cdf, seed=seed)
            if ai == ci:
                continue
            a = keys[ai]
            c = keys[ci]
            b = seed.choice(list(G.neighbors(a)))
            d = seed.choice(list(G.neighbors(c)))
            if b in [a, c, d] or d in [a, b, c]:
                continue
            if d not in G[a] and b not in G[c]:
                G.add_edge(a, d)
                G.add_edge(c, b)
                G.remove_edge(a, b)
                G.remove_edge(c, d)
                if connectivity and local_conn(G, a, b) == 0:
                    G.remove_edge(a, d)
                    G.remove_edge(c, b)
                    G.add_edge(a, b)
                    G.add_edge(c, d)
                else:
                    swapcount += 1
                    break
            n += 1
    return G

@not_implemented_for('directed')
@not_implemented_for('multigraph')
@py_random_state(4)
@nx._dispatch
def lattice_reference(G, niter=5, D=None, connectivity=True, seed=None):
    if False:
        i = 10
        return i + 15
    'Latticize the given graph by swapping edges.\n\n    Parameters\n    ----------\n    G : graph\n        An undirected graph.\n\n    niter : integer (optional, default=1)\n        An edge is rewired approximately niter times.\n\n    D : numpy.array (optional, default=None)\n        Distance to the diagonal matrix.\n\n    connectivity : boolean (optional, default=True)\n        Ensure connectivity for the latticized graph when set to True.\n\n    seed : integer, random_state, or None (default)\n        Indicator of random number generation state.\n        See :ref:`Randomness<randomness>`.\n\n    Returns\n    -------\n    G : graph\n        The latticized graph.\n\n    Raises\n    ------\n    NetworkXError\n        If there are fewer than 4 nodes or 2 edges in `G`\n\n    Notes\n    -----\n    The implementation is adapted from the algorithm by Sporns et al. [1]_.\n    which is inspired from the original work by Maslov and Sneppen(2002) [2]_.\n\n    References\n    ----------\n    .. [1] Sporns, Olaf, and Jonathan D. Zwi.\n       "The small world of the cerebral cortex."\n       Neuroinformatics 2.2 (2004): 145-162.\n    .. [2] Maslov, Sergei, and Kim Sneppen.\n       "Specificity and stability in topology of protein networks."\n       Science 296.5569 (2002): 910-913.\n    '
    import numpy as np
    from networkx.utils import cumulative_distribution, discrete_sequence
    local_conn = nx.connectivity.local_edge_connectivity
    if len(G) < 4:
        raise nx.NetworkXError('Graph has fewer than four nodes.')
    if len(G.edges) < 2:
        raise nx.NetworkXError('Graph has fewer that 2 edges')
    G = G.copy()
    (keys, degrees) = zip(*G.degree())
    cdf = cumulative_distribution(degrees)
    nnodes = len(G)
    nedges = nx.number_of_edges(G)
    if D is None:
        D = np.zeros((nnodes, nnodes))
        un = np.arange(1, nnodes)
        um = np.arange(nnodes - 1, 0, -1)
        u = np.append((0,), np.where(un < um, un, um))
        for v in range(int(np.ceil(nnodes / 2))):
            D[nnodes - v - 1, :] = np.append(u[v + 1:], u[:v + 1])
            D[v, :] = D[nnodes - v - 1, :][::-1]
    niter = niter * nedges
    max_attempts = int(nnodes * nedges / (nnodes * (nnodes - 1) / 2))
    for _ in range(niter):
        n = 0
        while n < max_attempts:
            (ai, ci) = discrete_sequence(2, cdistribution=cdf, seed=seed)
            if ai == ci:
                continue
            a = keys[ai]
            c = keys[ci]
            b = seed.choice(list(G.neighbors(a)))
            d = seed.choice(list(G.neighbors(c)))
            bi = keys.index(b)
            di = keys.index(d)
            if b in [a, c, d] or d in [a, b, c]:
                continue
            if d not in G[a] and b not in G[c]:
                if D[ai, bi] + D[ci, di] >= D[ai, ci] + D[bi, di]:
                    G.add_edge(a, d)
                    G.add_edge(c, b)
                    G.remove_edge(a, b)
                    G.remove_edge(c, d)
                    if connectivity and local_conn(G, a, b) == 0:
                        G.remove_edge(a, d)
                        G.remove_edge(c, b)
                        G.add_edge(a, b)
                        G.add_edge(c, d)
                    else:
                        break
            n += 1
    return G

@not_implemented_for('directed')
@not_implemented_for('multigraph')
@py_random_state(3)
@nx._dispatch
def sigma(G, niter=100, nrand=10, seed=None):
    if False:
        while True:
            i = 10
    'Returns the small-world coefficient (sigma) of the given graph.\n\n    The small-world coefficient is defined as:\n    sigma = C/Cr / L/Lr\n    where C and L are respectively the average clustering coefficient and\n    average shortest path length of G. Cr and Lr are respectively the average\n    clustering coefficient and average shortest path length of an equivalent\n    random graph.\n\n    A graph is commonly classified as small-world if sigma>1.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n        An undirected graph.\n    niter : integer (optional, default=100)\n        Approximate number of rewiring per edge to compute the equivalent\n        random graph.\n    nrand : integer (optional, default=10)\n        Number of random graphs generated to compute the average clustering\n        coefficient (Cr) and average shortest path length (Lr).\n    seed : integer, random_state, or None (default)\n        Indicator of random number generation state.\n        See :ref:`Randomness<randomness>`.\n\n    Returns\n    -------\n    sigma : float\n        The small-world coefficient of G.\n\n    Notes\n    -----\n    The implementation is adapted from Humphries et al. [1]_ [2]_.\n\n    References\n    ----------\n    .. [1] The brainstem reticular formation is a small-world, not scale-free,\n           network M. D. Humphries, K. Gurney and T. J. Prescott,\n           Proc. Roy. Soc. B 2006 273, 503-511, doi:10.1098/rspb.2005.3354.\n    .. [2] Humphries and Gurney (2008).\n           "Network \'Small-World-Ness\': A Quantitative Method for Determining\n           Canonical Network Equivalence".\n           PLoS One. 3 (4). PMID 18446219. doi:10.1371/journal.pone.0002051.\n    '
    import numpy as np
    randMetrics = {'C': [], 'L': []}
    for i in range(nrand):
        Gr = random_reference(G, niter=niter, seed=seed)
        randMetrics['C'].append(nx.transitivity(Gr))
        randMetrics['L'].append(nx.average_shortest_path_length(Gr))
    C = nx.transitivity(G)
    L = nx.average_shortest_path_length(G)
    Cr = np.mean(randMetrics['C'])
    Lr = np.mean(randMetrics['L'])
    sigma = C / Cr / (L / Lr)
    return sigma

@not_implemented_for('directed')
@not_implemented_for('multigraph')
@py_random_state(3)
@nx._dispatch
def omega(G, niter=5, nrand=10, seed=None):
    if False:
        print('Hello World!')
    'Returns the small-world coefficient (omega) of a graph\n\n    The small-world coefficient of a graph G is:\n\n    omega = Lr/L - C/Cl\n\n    where C and L are respectively the average clustering coefficient and\n    average shortest path length of G. Lr is the average shortest path length\n    of an equivalent random graph and Cl is the average clustering coefficient\n    of an equivalent lattice graph.\n\n    The small-world coefficient (omega) measures how much G is like a lattice\n    or a random graph. Negative values mean G is similar to a lattice whereas\n    positive values mean G is a random graph.\n    Values close to 0 mean that G has small-world characteristics.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n        An undirected graph.\n\n    niter: integer (optional, default=5)\n        Approximate number of rewiring per edge to compute the equivalent\n        random graph.\n\n    nrand: integer (optional, default=10)\n        Number of random graphs generated to compute the maximal clustering\n        coefficient (Cr) and average shortest path length (Lr).\n\n    seed : integer, random_state, or None (default)\n        Indicator of random number generation state.\n        See :ref:`Randomness<randomness>`.\n\n\n    Returns\n    -------\n    omega : float\n        The small-world coefficient (omega)\n\n    Notes\n    -----\n    The implementation is adapted from the algorithm by Telesford et al. [1]_.\n\n    References\n    ----------\n    .. [1] Telesford, Joyce, Hayasaka, Burdette, and Laurienti (2011).\n           "The Ubiquity of Small-World Networks".\n           Brain Connectivity. 1 (0038): 367-75.  PMC 3604768. PMID 22432451.\n           doi:10.1089/brain.2011.0038.\n    '
    import numpy as np
    randMetrics = {'C': [], 'L': []}
    Cl = nx.average_clustering(G)
    niter_lattice_reference = niter
    niter_random_reference = niter * 2
    for _ in range(nrand):
        Gr = random_reference(G, niter=niter_random_reference, seed=seed)
        randMetrics['L'].append(nx.average_shortest_path_length(Gr))
        Gl = lattice_reference(G, niter=niter_lattice_reference, seed=seed)
        Cl_temp = nx.average_clustering(Gl)
        if Cl_temp > Cl:
            Cl = Cl_temp
    C = nx.average_clustering(G)
    L = nx.average_shortest_path_length(G)
    Lr = np.mean(randMetrics['L'])
    omega = Lr / L - C / Cl
    return omega