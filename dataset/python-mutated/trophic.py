"""Trophic levels"""
import networkx as nx
from networkx.utils import not_implemented_for
__all__ = ['trophic_levels', 'trophic_differences', 'trophic_incoherence_parameter']

@not_implemented_for('undirected')
@nx._dispatch(edge_attrs='weight')
def trophic_levels(G, weight='weight'):
    if False:
        return 10
    'Compute the trophic levels of nodes.\n\n    The trophic level of a node $i$ is\n\n    .. math::\n\n        s_i = 1 + \\frac{1}{k^{in}_i} \\sum_{j} a_{ij} s_j\n\n    where $k^{in}_i$ is the in-degree of i\n\n    .. math::\n\n        k^{in}_i = \\sum_{j} a_{ij}\n\n    and nodes with $k^{in}_i = 0$ have $s_i = 1$ by convention.\n\n    These are calculated using the method outlined in Levine [1]_.\n\n    Parameters\n    ----------\n    G : DiGraph\n        A directed networkx graph\n\n    Returns\n    -------\n    nodes : dict\n        Dictionary of nodes with trophic level as the value.\n\n    References\n    ----------\n    .. [1] Stephen Levine (1980) J. theor. Biol. 83, 195-207\n    '
    import numpy as np
    a = nx.adjacency_matrix(G, weight=weight).T.toarray()
    rowsum = np.sum(a, axis=1)
    p = a[rowsum != 0][:, rowsum != 0]
    p = p / rowsum[rowsum != 0][:, np.newaxis]
    nn = p.shape[0]
    i = np.eye(nn)
    try:
        n = np.linalg.inv(i - p)
    except np.linalg.LinAlgError as err:
        msg = 'Trophic levels are only defined for graphs where every ' + 'node has a path from a basal node (basal nodes are nodes ' + 'with no incoming edges).'
        raise nx.NetworkXError(msg) from err
    y = n.sum(axis=1) + 1
    levels = {}
    zero_node_ids = (node_id for (node_id, degree) in G.in_degree if degree == 0)
    for node_id in zero_node_ids:
        levels[node_id] = 1
    nonzero_node_ids = (node_id for (node_id, degree) in G.in_degree if degree != 0)
    for (i, node_id) in enumerate(nonzero_node_ids):
        levels[node_id] = y[i]
    return levels

@not_implemented_for('undirected')
@nx._dispatch(edge_attrs='weight')
def trophic_differences(G, weight='weight'):
    if False:
        return 10
    'Compute the trophic differences of the edges of a directed graph.\n\n    The trophic difference $x_ij$ for each edge is defined in Johnson et al.\n    [1]_ as:\n\n    .. math::\n        x_ij = s_j - s_i\n\n    Where $s_i$ is the trophic level of node $i$.\n\n    Parameters\n    ----------\n    G : DiGraph\n        A directed networkx graph\n\n    Returns\n    -------\n    diffs : dict\n        Dictionary of edges with trophic differences as the value.\n\n    References\n    ----------\n    .. [1] Samuel Johnson, Virginia Dominguez-Garcia, Luca Donetti, Miguel A.\n        Munoz (2014) PNAS "Trophic coherence determines food-web stability"\n    '
    levels = trophic_levels(G, weight=weight)
    diffs = {}
    for (u, v) in G.edges:
        diffs[u, v] = levels[v] - levels[u]
    return diffs

@not_implemented_for('undirected')
@nx._dispatch(edge_attrs='weight')
def trophic_incoherence_parameter(G, weight='weight', cannibalism=False):
    if False:
        print('Hello World!')
    'Compute the trophic incoherence parameter of a graph.\n\n    Trophic coherence is defined as the homogeneity of the distribution of\n    trophic distances: the more similar, the more coherent. This is measured by\n    the standard deviation of the trophic differences and referred to as the\n    trophic incoherence parameter $q$ by [1].\n\n    Parameters\n    ----------\n    G : DiGraph\n        A directed networkx graph\n\n    cannibalism: Boolean\n        If set to False, self edges are not considered in the calculation\n\n    Returns\n    -------\n    trophic_incoherence_parameter : float\n        The trophic coherence of a graph\n\n    References\n    ----------\n    .. [1] Samuel Johnson, Virginia Dominguez-Garcia, Luca Donetti, Miguel A.\n        Munoz (2014) PNAS "Trophic coherence determines food-web stability"\n    '
    import numpy as np
    if cannibalism:
        diffs = trophic_differences(G, weight=weight)
    else:
        self_loops = list(nx.selfloop_edges(G))
        if self_loops:
            G_2 = G.copy()
            G_2.remove_edges_from(self_loops)
        else:
            G_2 = G
        diffs = trophic_differences(G_2, weight=weight)
    return np.std(list(diffs.values()))