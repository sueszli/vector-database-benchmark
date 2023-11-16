"""
=======================
Distance-regular graphs
=======================
"""
import networkx as nx
from networkx.utils import not_implemented_for
from .distance_measures import diameter
__all__ = ['is_distance_regular', 'is_strongly_regular', 'intersection_array', 'global_parameters']

@nx._dispatch
def is_distance_regular(G):
    if False:
        while True:
            i = 10
    'Returns True if the graph is distance regular, False otherwise.\n\n    A connected graph G is distance-regular if for any nodes x,y\n    and any integers i,j=0,1,...,d (where d is the graph\n    diameter), the number of vertices at distance i from x and\n    distance j from y depends only on i,j and the graph distance\n    between x and y, independently of the choice of x and y.\n\n    Parameters\n    ----------\n    G: Networkx graph (undirected)\n\n    Returns\n    -------\n    bool\n      True if the graph is Distance Regular, False otherwise\n\n    Examples\n    --------\n    >>> G = nx.hypercube_graph(6)\n    >>> nx.is_distance_regular(G)\n    True\n\n    See Also\n    --------\n    intersection_array, global_parameters\n\n    Notes\n    -----\n    For undirected and simple graphs only\n\n    References\n    ----------\n    .. [1] Brouwer, A. E.; Cohen, A. M.; and Neumaier, A.\n        Distance-Regular Graphs. New York: Springer-Verlag, 1989.\n    .. [2] Weisstein, Eric W. "Distance-Regular Graph."\n        http://mathworld.wolfram.com/Distance-RegularGraph.html\n\n    '
    try:
        intersection_array(G)
        return True
    except nx.NetworkXError:
        return False

def global_parameters(b, c):
    if False:
        i = 10
        return i + 15
    'Returns global parameters for a given intersection array.\n\n    Given a distance-regular graph G with integers b_i, c_i,i = 0,....,d\n    such that for any 2 vertices x,y in G at a distance i=d(x,y), there\n    are exactly c_i neighbors of y at a distance of i-1 from x and b_i\n    neighbors of y at a distance of i+1 from x.\n\n    Thus, a distance regular graph has the global parameters,\n    [[c_0,a_0,b_0],[c_1,a_1,b_1],......,[c_d,a_d,b_d]] for the\n    intersection array  [b_0,b_1,.....b_{d-1};c_1,c_2,.....c_d]\n    where a_i+b_i+c_i=k , k= degree of every vertex.\n\n    Parameters\n    ----------\n    b : list\n\n    c : list\n\n    Returns\n    -------\n    iterable\n       An iterable over three tuples.\n\n    Examples\n    --------\n    >>> G = nx.dodecahedral_graph()\n    >>> b, c = nx.intersection_array(G)\n    >>> list(nx.global_parameters(b, c))\n    [(0, 0, 3), (1, 0, 2), (1, 1, 1), (1, 1, 1), (2, 0, 1), (3, 0, 0)]\n\n    References\n    ----------\n    .. [1] Weisstein, Eric W. "Global Parameters."\n       From MathWorld--A Wolfram Web Resource.\n       http://mathworld.wolfram.com/GlobalParameters.html\n\n    See Also\n    --------\n    intersection_array\n    '
    return ((y, b[0] - x - y, x) for (x, y) in zip(b + [0], [0] + c))

@not_implemented_for('directed', 'multigraph')
@nx._dispatch
def intersection_array(G):
    if False:
        print('Hello World!')
    'Returns the intersection array of a distance-regular graph.\n\n    Given a distance-regular graph G with integers b_i, c_i,i = 0,....,d\n    such that for any 2 vertices x,y in G at a distance i=d(x,y), there\n    are exactly c_i neighbors of y at a distance of i-1 from x and b_i\n    neighbors of y at a distance of i+1 from x.\n\n    A distance regular graph\'s intersection array is given by,\n    [b_0,b_1,.....b_{d-1};c_1,c_2,.....c_d]\n\n    Parameters\n    ----------\n    G: Networkx graph (undirected)\n\n    Returns\n    -------\n    b,c: tuple of lists\n\n    Examples\n    --------\n    >>> G = nx.icosahedral_graph()\n    >>> nx.intersection_array(G)\n    ([5, 2, 1], [1, 2, 5])\n\n    References\n    ----------\n    .. [1] Weisstein, Eric W. "Intersection Array."\n       From MathWorld--A Wolfram Web Resource.\n       http://mathworld.wolfram.com/IntersectionArray.html\n\n    See Also\n    --------\n    global_parameters\n    '
    degree = iter(G.degree())
    (_, k) = next(degree)
    for (_, knext) in degree:
        if knext != k:
            raise nx.NetworkXError('Graph is not distance regular.')
        k = knext
    path_length = dict(nx.all_pairs_shortest_path_length(G))
    diameter = max((max(path_length[n].values()) for n in path_length))
    bint = {}
    cint = {}
    for u in G:
        for v in G:
            try:
                i = path_length[u][v]
            except KeyError as err:
                raise nx.NetworkXError('Graph is not distance regular.') from err
            c = len([n for n in G[v] if path_length[n][u] == i - 1])
            b = len([n for n in G[v] if path_length[n][u] == i + 1])
            if cint.get(i, c) != c or bint.get(i, b) != b:
                raise nx.NetworkXError('Graph is not distance regular')
            bint[i] = b
            cint[i] = c
    return ([bint.get(j, 0) for j in range(diameter)], [cint.get(j + 1, 0) for j in range(diameter)])

@not_implemented_for('directed', 'multigraph')
@nx._dispatch
def is_strongly_regular(G):
    if False:
        for i in range(10):
            print('nop')
    'Returns True if and only if the given graph is strongly\n    regular.\n\n    An undirected graph is *strongly regular* if\n\n    * it is regular,\n    * each pair of adjacent vertices has the same number of neighbors in\n      common,\n    * each pair of nonadjacent vertices has the same number of neighbors\n      in common.\n\n    Each strongly regular graph is a distance-regular graph.\n    Conversely, if a distance-regular graph has diameter two, then it is\n    a strongly regular graph. For more information on distance-regular\n    graphs, see :func:`is_distance_regular`.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n        An undirected graph.\n\n    Returns\n    -------\n    bool\n        Whether `G` is strongly regular.\n\n    Examples\n    --------\n\n    The cycle graph on five vertices is strongly regular. It is\n    two-regular, each pair of adjacent vertices has no shared neighbors,\n    and each pair of nonadjacent vertices has one shared neighbor::\n\n        >>> G = nx.cycle_graph(5)\n        >>> nx.is_strongly_regular(G)\n        True\n\n    '
    return is_distance_regular(G) and diameter(G) == 2