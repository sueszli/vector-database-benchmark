"""Functions for computing rich-club coefficients."""
from itertools import accumulate
import networkx as nx
from networkx.utils import not_implemented_for
__all__ = ['rich_club_coefficient']

@not_implemented_for('directed')
@not_implemented_for('multigraph')
@nx._dispatch
def rich_club_coefficient(G, normalized=True, Q=100, seed=None):
    if False:
        i = 10
        return i + 15
    'Returns the rich-club coefficient of the graph `G`.\n\n    For each degree *k*, the *rich-club coefficient* is the ratio of the\n    number of actual to the number of potential edges for nodes with\n    degree greater than *k*:\n\n    .. math::\n\n        \\phi(k) = \\frac{2 E_k}{N_k (N_k - 1)}\n\n    where `N_k` is the number of nodes with degree larger than *k*, and\n    `E_k` is the number of edges among those nodes.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n        Undirected graph with neither parallel edges nor self-loops.\n    normalized : bool (optional)\n        Normalize using randomized network as in [1]_\n    Q : float (optional, default=100)\n        If `normalized` is True, perform `Q * m` double-edge\n        swaps, where `m` is the number of edges in `G`, to use as a\n        null-model for normalization.\n    seed : integer, random_state, or None (default)\n        Indicator of random number generation state.\n        See :ref:`Randomness<randomness>`.\n\n    Returns\n    -------\n    rc : dictionary\n       A dictionary, keyed by degree, with rich-club coefficient values.\n\n    Examples\n    --------\n    >>> G = nx.Graph([(0, 1), (0, 2), (1, 2), (1, 3), (1, 4), (4, 5)])\n    >>> rc = nx.rich_club_coefficient(G, normalized=False, seed=42)\n    >>> rc[0]\n    0.4\n\n    Notes\n    -----\n    The rich club definition and algorithm are found in [1]_.  This\n    algorithm ignores any edge weights and is not defined for directed\n    graphs or graphs with parallel edges or self loops.\n\n    Estimates for appropriate values of `Q` are found in [2]_.\n\n    References\n    ----------\n    .. [1] Julian J. McAuley, Luciano da Fontoura Costa,\n       and Tibério S. Caetano,\n       "The rich-club phenomenon across complex network hierarchies",\n       Applied Physics Letters Vol 91 Issue 8, August 2007.\n       https://arxiv.org/abs/physics/0701290\n    .. [2] R. Milo, N. Kashtan, S. Itzkovitz, M. E. J. Newman, U. Alon,\n       "Uniform generation of random graphs with arbitrary degree\n       sequences", 2006. https://arxiv.org/abs/cond-mat/0312028\n    '
    if nx.number_of_selfloops(G) > 0:
        raise Exception('rich_club_coefficient is not implemented for graphs with self loops.')
    rc = _compute_rc(G)
    if normalized:
        R = G.copy()
        E = R.number_of_edges()
        nx.double_edge_swap(R, Q * E, max_tries=Q * E * 10, seed=seed)
        rcran = _compute_rc(R)
        rc = {k: v / rcran[k] for (k, v) in rc.items()}
    return rc

def _compute_rc(G):
    if False:
        print('Hello World!')
    'Returns the rich-club coefficient for each degree in the graph\n    `G`.\n\n    `G` is an undirected graph without multiedges.\n\n    Returns a dictionary mapping degree to rich-club coefficient for\n    that degree.\n\n    '
    deghist = nx.degree_histogram(G)
    total = sum(deghist)
    nks = (total - cs for cs in accumulate(deghist) if total - cs > 1)
    edge_degrees = sorted((sorted(map(G.degree, e)) for e in G.edges()), reverse=True)
    ek = G.number_of_edges()
    (k1, k2) = edge_degrees.pop()
    rc = {}
    for (d, nk) in enumerate(nks):
        while k1 <= d:
            if len(edge_degrees) == 0:
                ek = 0
                break
            (k1, k2) = edge_degrees.pop()
            ek -= 1
        rc[d] = 2 * ek / (nk * (nk - 1))
    return rc