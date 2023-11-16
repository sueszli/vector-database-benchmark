"""Generates graphs with a given eigenvector structure"""
import networkx as nx
from networkx.utils import np_random_state
__all__ = ['spectral_graph_forge']

@np_random_state(3)
@nx._dispatch
def spectral_graph_forge(G, alpha, transformation='identity', seed=None):
    if False:
        i = 10
        return i + 15
    'Returns a random simple graph with spectrum resembling that of `G`\n\n    This algorithm, called Spectral Graph Forge (SGF), computes the\n    eigenvectors of a given graph adjacency matrix, filters them and\n    builds a random graph with a similar eigenstructure.\n    SGF has been proved to be particularly useful for synthesizing\n    realistic social networks and it can also be used to anonymize\n    graph sensitive data.\n\n    Parameters\n    ----------\n    G : Graph\n    alpha :  float\n        Ratio representing the percentage of eigenvectors of G to consider,\n        values in [0,1].\n    transformation : string, optional\n        Represents the intended matrix linear transformation, possible values\n        are \'identity\' and \'modularity\'\n    seed : integer, random_state, or None (default)\n        Indicator of numpy random number generation state.\n        See :ref:`Randomness<randomness>`.\n\n    Returns\n    -------\n    H : Graph\n        A graph with a similar eigenvector structure of the input one.\n\n    Raises\n    ------\n    NetworkXError\n        If transformation has a value different from \'identity\' or \'modularity\'\n\n    Notes\n    -----\n    Spectral Graph Forge (SGF) generates a random simple graph resembling the\n    global properties of the given one.\n    It leverages the low-rank approximation of the associated adjacency matrix\n    driven by the *alpha* precision parameter.\n    SGF preserves the number of nodes of the input graph and their ordering.\n    This way, nodes of output graphs resemble the properties of the input one\n    and attributes can be directly mapped.\n\n    It considers the graph adjacency matrices which can optionally be\n    transformed to other symmetric real matrices (currently transformation\n    options include *identity* and *modularity*).\n    The *modularity* transformation, in the sense of Newman\'s modularity matrix\n    allows the focusing on community structure related properties of the graph.\n\n    SGF applies a low-rank approximation whose fixed rank is computed from the\n    ratio *alpha* of the input graph adjacency matrix dimension.\n    This step performs a filtering on the input eigenvectors similar to the low\n    pass filtering common in telecommunications.\n\n    The filtered values (after truncation) are used as input to a Bernoulli\n    sampling for constructing a random adjacency matrix.\n\n    References\n    ----------\n    ..  [1] L. Baldesi, C. T. Butts, A. Markopoulou, "Spectral Graph Forge:\n        Graph Generation Targeting Modularity", IEEE Infocom, \'18.\n        https://arxiv.org/abs/1801.01715\n    ..  [2] M. Newman, "Networks: an introduction", Oxford university press,\n        2010\n\n    Examples\n    --------\n    >>> G = nx.karate_club_graph()\n    >>> H = nx.spectral_graph_forge(G, 0.3)\n    >>>\n    '
    import numpy as np
    import scipy as sp
    available_transformations = ['identity', 'modularity']
    alpha = np.clip(alpha, 0, 1)
    A = nx.to_numpy_array(G)
    n = A.shape[1]
    level = round(n * alpha)
    if transformation not in available_transformations:
        msg = f'{transformation!r} is not a valid transformation. '
        msg += f'Transformations: {available_transformations}'
        raise nx.NetworkXError(msg)
    K = np.ones((1, n)) @ A
    B = A
    if transformation == 'modularity':
        B -= K.T @ K / K.sum()
    (evals, evecs) = np.linalg.eigh(B)
    k = np.argsort(np.abs(evals))[::-1]
    evecs[:, k[np.arange(level, n)]] = 0
    B = evecs @ np.diag(evals) @ evecs.T
    if transformation == 'modularity':
        B += K.T @ K / K.sum()
    B = np.clip(B, 0, 1)
    np.fill_diagonal(B, 0)
    for i in range(n - 1):
        B[i, i + 1:] = sp.stats.bernoulli.rvs(B[i, i + 1:], random_state=seed)
        B[i + 1:, i] = np.transpose(B[i, i + 1:])
    H = nx.from_numpy_array(B)
    return H