"""
Graph isomorphism functions.
"""
import networkx as nx
from networkx.exception import NetworkXError
__all__ = ['could_be_isomorphic', 'fast_could_be_isomorphic', 'faster_could_be_isomorphic', 'is_isomorphic']

@nx._dispatch(graphs={'G1': 0, 'G2': 1})
def could_be_isomorphic(G1, G2):
    if False:
        i = 10
        return i + 15
    'Returns False if graphs are definitely not isomorphic.\n    True does NOT guarantee isomorphism.\n\n    Parameters\n    ----------\n    G1, G2 : graphs\n       The two graphs G1 and G2 must be the same type.\n\n    Notes\n    -----\n    Checks for matching degree, triangle, and number of cliques sequences.\n    The triangle sequence contains the number of triangles each node is part of.\n    The clique sequence contains for each node the number of maximal cliques\n    involving that node.\n\n    '
    if G1.order() != G2.order():
        return False
    d1 = G1.degree()
    t1 = nx.triangles(G1)
    clqs_1 = list(nx.find_cliques(G1))
    c1 = {n: sum((1 for c in clqs_1 if n in c)) for n in G1}
    props1 = [[d, t1[v], c1[v]] for (v, d) in d1]
    props1.sort()
    d2 = G2.degree()
    t2 = nx.triangles(G2)
    clqs_2 = list(nx.find_cliques(G2))
    c2 = {n: sum((1 for c in clqs_2 if n in c)) for n in G2}
    props2 = [[d, t2[v], c2[v]] for (v, d) in d2]
    props2.sort()
    if props1 != props2:
        return False
    return True
graph_could_be_isomorphic = could_be_isomorphic

@nx._dispatch(graphs={'G1': 0, 'G2': 1})
def fast_could_be_isomorphic(G1, G2):
    if False:
        while True:
            i = 10
    'Returns False if graphs are definitely not isomorphic.\n\n    True does NOT guarantee isomorphism.\n\n    Parameters\n    ----------\n    G1, G2 : graphs\n       The two graphs G1 and G2 must be the same type.\n\n    Notes\n    -----\n    Checks for matching degree and triangle sequences. The triangle\n    sequence contains the number of triangles each node is part of.\n    '
    if G1.order() != G2.order():
        return False
    d1 = G1.degree()
    t1 = nx.triangles(G1)
    props1 = [[d, t1[v]] for (v, d) in d1]
    props1.sort()
    d2 = G2.degree()
    t2 = nx.triangles(G2)
    props2 = [[d, t2[v]] for (v, d) in d2]
    props2.sort()
    if props1 != props2:
        return False
    return True
fast_graph_could_be_isomorphic = fast_could_be_isomorphic

@nx._dispatch(graphs={'G1': 0, 'G2': 1})
def faster_could_be_isomorphic(G1, G2):
    if False:
        while True:
            i = 10
    'Returns False if graphs are definitely not isomorphic.\n\n    True does NOT guarantee isomorphism.\n\n    Parameters\n    ----------\n    G1, G2 : graphs\n       The two graphs G1 and G2 must be the same type.\n\n    Notes\n    -----\n    Checks for matching degree sequences.\n    '
    if G1.order() != G2.order():
        return False
    d1 = sorted((d for (n, d) in G1.degree()))
    d2 = sorted((d for (n, d) in G2.degree()))
    if d1 != d2:
        return False
    return True
faster_graph_could_be_isomorphic = faster_could_be_isomorphic

@nx._dispatch(graphs={'G1': 0, 'G2': 1}, preserve_edge_attrs='edge_match', preserve_node_attrs='node_match')
def is_isomorphic(G1, G2, node_match=None, edge_match=None):
    if False:
        print('Hello World!')
    'Returns True if the graphs G1 and G2 are isomorphic and False otherwise.\n\n    Parameters\n    ----------\n    G1, G2: graphs\n        The two graphs G1 and G2 must be the same type.\n\n    node_match : callable\n        A function that returns True if node n1 in G1 and n2 in G2 should\n        be considered equal during the isomorphism test.\n        If node_match is not specified then node attributes are not considered.\n\n        The function will be called like\n\n           node_match(G1.nodes[n1], G2.nodes[n2]).\n\n        That is, the function will receive the node attribute dictionaries\n        for n1 and n2 as inputs.\n\n    edge_match : callable\n        A function that returns True if the edge attribute dictionary\n        for the pair of nodes (u1, v1) in G1 and (u2, v2) in G2 should\n        be considered equal during the isomorphism test.  If edge_match is\n        not specified then edge attributes are not considered.\n\n        The function will be called like\n\n           edge_match(G1[u1][v1], G2[u2][v2]).\n\n        That is, the function will receive the edge attribute dictionaries\n        of the edges under consideration.\n\n    Notes\n    -----\n    Uses the vf2 algorithm [1]_.\n\n    Examples\n    --------\n    >>> import networkx.algorithms.isomorphism as iso\n\n    For digraphs G1 and G2, using \'weight\' edge attribute (default: 1)\n\n    >>> G1 = nx.DiGraph()\n    >>> G2 = nx.DiGraph()\n    >>> nx.add_path(G1, [1, 2, 3, 4], weight=1)\n    >>> nx.add_path(G2, [10, 20, 30, 40], weight=2)\n    >>> em = iso.numerical_edge_match("weight", 1)\n    >>> nx.is_isomorphic(G1, G2)  # no weights considered\n    True\n    >>> nx.is_isomorphic(G1, G2, edge_match=em)  # match weights\n    False\n\n    For multidigraphs G1 and G2, using \'fill\' node attribute (default: \'\')\n\n    >>> G1 = nx.MultiDiGraph()\n    >>> G2 = nx.MultiDiGraph()\n    >>> G1.add_nodes_from([1, 2, 3], fill="red")\n    >>> G2.add_nodes_from([10, 20, 30, 40], fill="red")\n    >>> nx.add_path(G1, [1, 2, 3, 4], weight=3, linewidth=2.5)\n    >>> nx.add_path(G2, [10, 20, 30, 40], weight=3)\n    >>> nm = iso.categorical_node_match("fill", "red")\n    >>> nx.is_isomorphic(G1, G2, node_match=nm)\n    True\n\n    For multidigraphs G1 and G2, using \'weight\' edge attribute (default: 7)\n\n    >>> G1.add_edge(1, 2, weight=7)\n    1\n    >>> G2.add_edge(10, 20)\n    1\n    >>> em = iso.numerical_multiedge_match("weight", 7, rtol=1e-6)\n    >>> nx.is_isomorphic(G1, G2, edge_match=em)\n    True\n\n    For multigraphs G1 and G2, using \'weight\' and \'linewidth\' edge attributes\n    with default values 7 and 2.5. Also using \'fill\' node attribute with\n    default value \'red\'.\n\n    >>> em = iso.numerical_multiedge_match(["weight", "linewidth"], [7, 2.5])\n    >>> nm = iso.categorical_node_match("fill", "red")\n    >>> nx.is_isomorphic(G1, G2, edge_match=em, node_match=nm)\n    True\n\n    See Also\n    --------\n    numerical_node_match, numerical_edge_match, numerical_multiedge_match\n    categorical_node_match, categorical_edge_match, categorical_multiedge_match\n\n    References\n    ----------\n    .. [1]  L. P. Cordella, P. Foggia, C. Sansone, M. Vento,\n       "An Improved Algorithm for Matching Large Graphs",\n       3rd IAPR-TC15 Workshop  on Graph-based Representations in\n       Pattern Recognition, Cuen, pp. 149-159, 2001.\n       https://www.researchgate.net/publication/200034365_An_Improved_Algorithm_for_Matching_Large_Graphs\n    '
    if G1.is_directed() and G2.is_directed():
        GM = nx.algorithms.isomorphism.DiGraphMatcher
    elif not G1.is_directed() and (not G2.is_directed()):
        GM = nx.algorithms.isomorphism.GraphMatcher
    else:
        raise NetworkXError('Graphs G1 and G2 are not of the same type.')
    gm = GM(G1, G2, node_match=node_match, edge_match=edge_match)
    return gm.is_isomorphic()