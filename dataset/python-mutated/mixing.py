"""
Mixing matrices for node attributes and degree.
"""
import networkx as nx
from networkx.algorithms.assortativity.pairs import node_attribute_xy, node_degree_xy
from networkx.utils import dict_to_numpy_array
__all__ = ['attribute_mixing_matrix', 'attribute_mixing_dict', 'degree_mixing_matrix', 'degree_mixing_dict', 'mixing_dict']

@nx._dispatch(node_attrs='attribute')
def attribute_mixing_dict(G, attribute, nodes=None, normalized=False):
    if False:
        while True:
            i = 10
    'Returns dictionary representation of mixing matrix for attribute.\n\n    Parameters\n    ----------\n    G : graph\n       NetworkX graph object.\n\n    attribute : string\n       Node attribute key.\n\n    nodes: list or iterable (optional)\n        Unse nodes in container to build the dict. The default is all nodes.\n\n    normalized : bool (default=False)\n       Return counts if False or probabilities if True.\n\n    Examples\n    --------\n    >>> G = nx.Graph()\n    >>> G.add_nodes_from([0, 1], color="red")\n    >>> G.add_nodes_from([2, 3], color="blue")\n    >>> G.add_edge(1, 3)\n    >>> d = nx.attribute_mixing_dict(G, "color")\n    >>> print(d["red"]["blue"])\n    1\n    >>> print(d["blue"]["red"])  # d symmetric for undirected graphs\n    1\n\n    Returns\n    -------\n    d : dictionary\n       Counts or joint probability of occurrence of attribute pairs.\n    '
    xy_iter = node_attribute_xy(G, attribute, nodes)
    return mixing_dict(xy_iter, normalized=normalized)

@nx._dispatch(node_attrs='attribute')
def attribute_mixing_matrix(G, attribute, nodes=None, mapping=None, normalized=True):
    if False:
        return 10
    "Returns mixing matrix for attribute.\n\n    Parameters\n    ----------\n    G : graph\n       NetworkX graph object.\n\n    attribute : string\n       Node attribute key.\n\n    nodes: list or iterable (optional)\n        Use only nodes in container to build the matrix. The default is\n        all nodes.\n\n    mapping : dictionary, optional\n       Mapping from node attribute to integer index in matrix.\n       If not specified, an arbitrary ordering will be used.\n\n    normalized : bool (default=True)\n       Return counts if False or probabilities if True.\n\n    Returns\n    -------\n    m: numpy array\n       Counts or joint probability of occurrence of attribute pairs.\n\n    Notes\n    -----\n    If each node has a unique attribute value, the unnormalized mixing matrix\n    will be equal to the adjacency matrix. To get a denser mixing matrix,\n    the rounding can be performed to form groups of nodes with equal values.\n    For example, the exact height of persons in cm (180.79155222, 163.9080892,\n    163.30095355, 167.99016217, 168.21590163, ...) can be rounded to (180, 163,\n    163, 168, 168, ...).\n\n    Definitions of attribute mixing matrix vary on whether the matrix\n    should include rows for attribute values that don't arise. Here we\n    do not include such empty-rows. But you can force them to appear\n    by inputting a `mapping` that includes those values.\n\n    Examples\n    --------\n    >>> G = nx.path_graph(3)\n    >>> gender = {0: 'male', 1: 'female', 2: 'female'}\n    >>> nx.set_node_attributes(G, gender, 'gender')\n    >>> mapping = {'male': 0, 'female': 1}\n    >>> mix_mat = nx.attribute_mixing_matrix(G, 'gender', mapping=mapping)\n    >>> # mixing from male nodes to female nodes\n    >>> mix_mat[mapping['male'], mapping['female']]\n    0.25\n    "
    d = attribute_mixing_dict(G, attribute, nodes)
    a = dict_to_numpy_array(d, mapping=mapping)
    if normalized:
        a = a / a.sum()
    return a

@nx._dispatch(edge_attrs='weight')
def degree_mixing_dict(G, x='out', y='in', weight=None, nodes=None, normalized=False):
    if False:
        while True:
            i = 10
    "Returns dictionary representation of mixing matrix for degree.\n\n    Parameters\n    ----------\n    G : graph\n        NetworkX graph object.\n\n    x: string ('in','out')\n       The degree type for source node (directed graphs only).\n\n    y: string ('in','out')\n       The degree type for target node (directed graphs only).\n\n    weight: string or None, optional (default=None)\n       The edge attribute that holds the numerical value used\n       as a weight.  If None, then each edge has weight 1.\n       The degree is the sum of the edge weights adjacent to the node.\n\n    normalized : bool (default=False)\n        Return counts if False or probabilities if True.\n\n    Returns\n    -------\n    d: dictionary\n       Counts or joint probability of occurrence of degree pairs.\n    "
    xy_iter = node_degree_xy(G, x=x, y=y, nodes=nodes, weight=weight)
    return mixing_dict(xy_iter, normalized=normalized)

@nx._dispatch(edge_attrs='weight')
def degree_mixing_matrix(G, x='out', y='in', weight=None, nodes=None, normalized=True, mapping=None):
    if False:
        return 10
    "Returns mixing matrix for attribute.\n\n    Parameters\n    ----------\n    G : graph\n       NetworkX graph object.\n\n    x: string ('in','out')\n       The degree type for source node (directed graphs only).\n\n    y: string ('in','out')\n       The degree type for target node (directed graphs only).\n\n    nodes: list or iterable (optional)\n        Build the matrix using only nodes in container.\n        The default is all nodes.\n\n    weight: string or None, optional (default=None)\n       The edge attribute that holds the numerical value used\n       as a weight.  If None, then each edge has weight 1.\n       The degree is the sum of the edge weights adjacent to the node.\n\n    normalized : bool (default=True)\n       Return counts if False or probabilities if True.\n\n    mapping : dictionary, optional\n       Mapping from node degree to integer index in matrix.\n       If not specified, an arbitrary ordering will be used.\n\n    Returns\n    -------\n    m: numpy array\n       Counts, or joint probability, of occurrence of node degree.\n\n    Notes\n    -----\n    Definitions of degree mixing matrix vary on whether the matrix\n    should include rows for degree values that don't arise. Here we\n    do not include such empty-rows. But you can force them to appear\n    by inputting a `mapping` that includes those values. See examples.\n\n    Examples\n    --------\n    >>> G = nx.star_graph(3)\n    >>> mix_mat = nx.degree_mixing_matrix(G)\n    >>> mix_mat[0, 1]  # mixing from node degree 1 to node degree 3\n    0.5\n\n    If you want every possible degree to appear as a row, even if no nodes\n    have that degree, use `mapping` as follows,\n\n    >>> max_degree = max(deg for n, deg in G.degree)\n    >>> mapping = {x: x for x in range(max_degree + 1)} # identity mapping\n    >>> mix_mat = nx.degree_mixing_matrix(G, mapping=mapping)\n    >>> mix_mat[3, 1]  # mixing from node degree 3 to node degree 1\n    0.5\n    "
    d = degree_mixing_dict(G, x=x, y=y, nodes=nodes, weight=weight)
    a = dict_to_numpy_array(d, mapping=mapping)
    if normalized:
        a = a / a.sum()
    return a

def mixing_dict(xy, normalized=False):
    if False:
        return 10
    'Returns a dictionary representation of mixing matrix.\n\n    Parameters\n    ----------\n    xy : list or container of two-tuples\n       Pairs of (x,y) items.\n\n    attribute : string\n       Node attribute key\n\n    normalized : bool (default=False)\n       Return counts if False or probabilities if True.\n\n    Returns\n    -------\n    d: dictionary\n       Counts or Joint probability of occurrence of values in xy.\n    '
    d = {}
    psum = 0.0
    for (x, y) in xy:
        if x not in d:
            d[x] = {}
        if y not in d:
            d[y] = {}
        v = d[x].get(y, 0)
        d[x][y] = v + 1
        psum += 1
    if normalized:
        for (_, jdict) in d.items():
            for j in jdict:
                jdict[j] /= psum
    return d