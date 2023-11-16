"""
    Functions for constructing matrix-like objects from graph attributes.
"""
import networkx as nx
__all__ = ['attr_matrix', 'attr_sparse_matrix']

def _node_value(G, node_attr):
    if False:
        print('Hello World!')
    'Returns a function that returns a value from G.nodes[u].\n\n    We return a function expecting a node as its sole argument. Then, in the\n    simplest scenario, the returned function will return G.nodes[u][node_attr].\n    However, we also handle the case when `node_attr` is None or when it is a\n    function itself.\n\n    Parameters\n    ----------\n    G : graph\n        A NetworkX graph\n\n    node_attr : {None, str, callable}\n        Specification of how the value of the node attribute should be obtained\n        from the node attribute dictionary.\n\n    Returns\n    -------\n    value : function\n        A function expecting a node as its sole argument. The function will\n        returns a value from G.nodes[u] that depends on `edge_attr`.\n\n    '
    if node_attr is None:

        def value(u):
            if False:
                return 10
            return u
    elif not callable(node_attr):

        def value(u):
            if False:
                print('Hello World!')
            return G.nodes[u][node_attr]
    else:
        value = node_attr
    return value

def _edge_value(G, edge_attr):
    if False:
        while True:
            i = 10
    'Returns a function that returns a value from G[u][v].\n\n    Suppose there exists an edge between u and v.  Then we return a function\n    expecting u and v as arguments.  For Graph and DiGraph, G[u][v] is\n    the edge attribute dictionary, and the function (essentially) returns\n    G[u][v][edge_attr].  However, we also handle cases when `edge_attr` is None\n    and when it is a function itself. For MultiGraph and MultiDiGraph, G[u][v]\n    is a dictionary of all edges between u and v.  In this case, the returned\n    function sums the value of `edge_attr` for every edge between u and v.\n\n    Parameters\n    ----------\n    G : graph\n       A NetworkX graph\n\n    edge_attr : {None, str, callable}\n        Specification of how the value of the edge attribute should be obtained\n        from the edge attribute dictionary, G[u][v].  For multigraphs, G[u][v]\n        is a dictionary of all the edges between u and v.  This allows for\n        special treatment of multiedges.\n\n    Returns\n    -------\n    value : function\n        A function expecting two nodes as parameters. The nodes should\n        represent the from- and to- node of an edge. The function will\n        return a value from G[u][v] that depends on `edge_attr`.\n\n    '
    if edge_attr is None:
        if G.is_multigraph():

            def value(u, v):
                if False:
                    return 10
                return len(G[u][v])
        else:

            def value(u, v):
                if False:
                    while True:
                        i = 10
                return 1
    elif not callable(edge_attr):
        if edge_attr == 'weight':
            if G.is_multigraph():

                def value(u, v):
                    if False:
                        return 10
                    return sum((d.get(edge_attr, 1) for d in G[u][v].values()))
            else:

                def value(u, v):
                    if False:
                        while True:
                            i = 10
                    return G[u][v].get(edge_attr, 1)
        elif G.is_multigraph():

            def value(u, v):
                if False:
                    print('Hello World!')
                return sum((d[edge_attr] for d in G[u][v].values()))
        else:

            def value(u, v):
                if False:
                    while True:
                        i = 10
                return G[u][v][edge_attr]
    else:
        value = edge_attr
    return value

@nx._dispatch(edge_attrs={'edge_attr': None}, node_attrs='node_attr')
def attr_matrix(G, edge_attr=None, node_attr=None, normalized=False, rc_order=None, dtype=None, order=None):
    if False:
        return 10
    'Returns the attribute matrix using attributes from `G` as a numpy array.\n\n    If only `G` is passed in, then the adjacency matrix is constructed.\n\n    Let A be a discrete set of values for the node attribute `node_attr`. Then\n    the elements of A represent the rows and columns of the constructed matrix.\n    Now, iterate through every edge e=(u,v) in `G` and consider the value\n    of the edge attribute `edge_attr`.  If ua and va are the values of the\n    node attribute `node_attr` for u and v, respectively, then the value of\n    the edge attribute is added to the matrix element at (ua, va).\n\n    Parameters\n    ----------\n    G : graph\n        The NetworkX graph used to construct the attribute matrix.\n\n    edge_attr : str, optional\n        Each element of the matrix represents a running total of the\n        specified edge attribute for edges whose node attributes correspond\n        to the rows/cols of the matrix. The attribute must be present for\n        all edges in the graph. If no attribute is specified, then we\n        just count the number of edges whose node attributes correspond\n        to the matrix element.\n\n    node_attr : str, optional\n        Each row and column in the matrix represents a particular value\n        of the node attribute.  The attribute must be present for all nodes\n        in the graph. Note, the values of this attribute should be reliably\n        hashable. So, float values are not recommended. If no attribute is\n        specified, then the rows and columns will be the nodes of the graph.\n\n    normalized : bool, optional\n        If True, then each row is normalized by the summation of its values.\n\n    rc_order : list, optional\n        A list of the node attribute values. This list specifies the ordering\n        of rows and columns of the array. If no ordering is provided, then\n        the ordering will be random (and also, a return value).\n\n    Other Parameters\n    ----------------\n    dtype : NumPy data-type, optional\n        A valid NumPy dtype used to initialize the array. Keep in mind certain\n        dtypes can yield unexpected results if the array is to be normalized.\n        The parameter is passed to numpy.zeros(). If unspecified, the NumPy\n        default is used.\n\n    order : {\'C\', \'F\'}, optional\n        Whether to store multidimensional data in C- or Fortran-contiguous\n        (row- or column-wise) order in memory. This parameter is passed to\n        numpy.zeros(). If unspecified, the NumPy default is used.\n\n    Returns\n    -------\n    M : 2D NumPy ndarray\n        The attribute matrix.\n\n    ordering : list\n        If `rc_order` was specified, then only the attribute matrix is returned.\n        However, if `rc_order` was None, then the ordering used to construct\n        the matrix is returned as well.\n\n    Examples\n    --------\n    Construct an adjacency matrix:\n\n    >>> G = nx.Graph()\n    >>> G.add_edge(0, 1, thickness=1, weight=3)\n    >>> G.add_edge(0, 2, thickness=2)\n    >>> G.add_edge(1, 2, thickness=3)\n    >>> nx.attr_matrix(G, rc_order=[0, 1, 2])\n    array([[0., 1., 1.],\n           [1., 0., 1.],\n           [1., 1., 0.]])\n\n    Alternatively, we can obtain the matrix describing edge thickness.\n\n    >>> nx.attr_matrix(G, edge_attr="thickness", rc_order=[0, 1, 2])\n    array([[0., 1., 2.],\n           [1., 0., 3.],\n           [2., 3., 0.]])\n\n    We can also color the nodes and ask for the probability distribution over\n    all edges (u,v) describing:\n\n        Pr(v has color Y | u has color X)\n\n    >>> G.nodes[0]["color"] = "red"\n    >>> G.nodes[1]["color"] = "red"\n    >>> G.nodes[2]["color"] = "blue"\n    >>> rc = ["red", "blue"]\n    >>> nx.attr_matrix(G, node_attr="color", normalized=True, rc_order=rc)\n    array([[0.33333333, 0.66666667],\n           [1.        , 0.        ]])\n\n    For example, the above tells us that for all edges (u,v):\n\n        Pr( v is red  | u is red)  = 1/3\n        Pr( v is blue | u is red)  = 2/3\n\n        Pr( v is red  | u is blue) = 1\n        Pr( v is blue | u is blue) = 0\n\n    Finally, we can obtain the total weights listed by the node colors.\n\n    >>> nx.attr_matrix(G, edge_attr="weight", node_attr="color", rc_order=rc)\n    array([[3., 2.],\n           [2., 0.]])\n\n    Thus, the total weight over all edges (u,v) with u and v having colors:\n\n        (red, red)   is 3   # the sole contribution is from edge (0,1)\n        (red, blue)  is 2   # contributions from edges (0,2) and (1,2)\n        (blue, red)  is 2   # same as (red, blue) since graph is undirected\n        (blue, blue) is 0   # there are no edges with blue endpoints\n\n    '
    import numpy as np
    edge_value = _edge_value(G, edge_attr)
    node_value = _node_value(G, node_attr)
    if rc_order is None:
        ordering = list({node_value(n) for n in G})
    else:
        ordering = rc_order
    N = len(ordering)
    undirected = not G.is_directed()
    index = dict(zip(ordering, range(N)))
    M = np.zeros((N, N), dtype=dtype, order=order)
    seen = set()
    for (u, nbrdict) in G.adjacency():
        for v in nbrdict:
            (i, j) = (index[node_value(u)], index[node_value(v)])
            if v not in seen:
                M[i, j] += edge_value(u, v)
                if undirected:
                    M[j, i] = M[i, j]
        if undirected:
            seen.add(u)
    if normalized:
        M /= M.sum(axis=1).reshape((N, 1))
    if rc_order is None:
        return (M, ordering)
    else:
        return M

@nx._dispatch(edge_attrs={'edge_attr': None}, node_attrs='node_attr')
def attr_sparse_matrix(G, edge_attr=None, node_attr=None, normalized=False, rc_order=None, dtype=None):
    if False:
        print('Hello World!')
    'Returns a SciPy sparse array using attributes from G.\n\n    If only `G` is passed in, then the adjacency matrix is constructed.\n\n    Let A be a discrete set of values for the node attribute `node_attr`. Then\n    the elements of A represent the rows and columns of the constructed matrix.\n    Now, iterate through every edge e=(u,v) in `G` and consider the value\n    of the edge attribute `edge_attr`.  If ua and va are the values of the\n    node attribute `node_attr` for u and v, respectively, then the value of\n    the edge attribute is added to the matrix element at (ua, va).\n\n    Parameters\n    ----------\n    G : graph\n        The NetworkX graph used to construct the NumPy matrix.\n\n    edge_attr : str, optional\n        Each element of the matrix represents a running total of the\n        specified edge attribute for edges whose node attributes correspond\n        to the rows/cols of the matrix. The attribute must be present for\n        all edges in the graph. If no attribute is specified, then we\n        just count the number of edges whose node attributes correspond\n        to the matrix element.\n\n    node_attr : str, optional\n        Each row and column in the matrix represents a particular value\n        of the node attribute.  The attribute must be present for all nodes\n        in the graph. Note, the values of this attribute should be reliably\n        hashable. So, float values are not recommended. If no attribute is\n        specified, then the rows and columns will be the nodes of the graph.\n\n    normalized : bool, optional\n        If True, then each row is normalized by the summation of its values.\n\n    rc_order : list, optional\n        A list of the node attribute values. This list specifies the ordering\n        of rows and columns of the array. If no ordering is provided, then\n        the ordering will be random (and also, a return value).\n\n    Other Parameters\n    ----------------\n    dtype : NumPy data-type, optional\n        A valid NumPy dtype used to initialize the array. Keep in mind certain\n        dtypes can yield unexpected results if the array is to be normalized.\n        The parameter is passed to numpy.zeros(). If unspecified, the NumPy\n        default is used.\n\n    Returns\n    -------\n    M : SciPy sparse array\n        The attribute matrix.\n\n    ordering : list\n        If `rc_order` was specified, then only the matrix is returned.\n        However, if `rc_order` was None, then the ordering used to construct\n        the matrix is returned as well.\n\n    Examples\n    --------\n    Construct an adjacency matrix:\n\n    >>> G = nx.Graph()\n    >>> G.add_edge(0, 1, thickness=1, weight=3)\n    >>> G.add_edge(0, 2, thickness=2)\n    >>> G.add_edge(1, 2, thickness=3)\n    >>> M = nx.attr_sparse_matrix(G, rc_order=[0, 1, 2])\n    >>> M.toarray()\n    array([[0., 1., 1.],\n           [1., 0., 1.],\n           [1., 1., 0.]])\n\n    Alternatively, we can obtain the matrix describing edge thickness.\n\n    >>> M = nx.attr_sparse_matrix(G, edge_attr="thickness", rc_order=[0, 1, 2])\n    >>> M.toarray()\n    array([[0., 1., 2.],\n           [1., 0., 3.],\n           [2., 3., 0.]])\n\n    We can also color the nodes and ask for the probability distribution over\n    all edges (u,v) describing:\n\n        Pr(v has color Y | u has color X)\n\n    >>> G.nodes[0]["color"] = "red"\n    >>> G.nodes[1]["color"] = "red"\n    >>> G.nodes[2]["color"] = "blue"\n    >>> rc = ["red", "blue"]\n    >>> M = nx.attr_sparse_matrix(G, node_attr="color", normalized=True, rc_order=rc)\n    >>> M.toarray()\n    array([[0.33333333, 0.66666667],\n           [1.        , 0.        ]])\n\n    For example, the above tells us that for all edges (u,v):\n\n        Pr( v is red  | u is red)  = 1/3\n        Pr( v is blue | u is red)  = 2/3\n\n        Pr( v is red  | u is blue) = 1\n        Pr( v is blue | u is blue) = 0\n\n    Finally, we can obtain the total weights listed by the node colors.\n\n    >>> M = nx.attr_sparse_matrix(G, edge_attr="weight", node_attr="color", rc_order=rc)\n    >>> M.toarray()\n    array([[3., 2.],\n           [2., 0.]])\n\n    Thus, the total weight over all edges (u,v) with u and v having colors:\n\n        (red, red)   is 3   # the sole contribution is from edge (0,1)\n        (red, blue)  is 2   # contributions from edges (0,2) and (1,2)\n        (blue, red)  is 2   # same as (red, blue) since graph is undirected\n        (blue, blue) is 0   # there are no edges with blue endpoints\n\n    '
    import numpy as np
    import scipy as sp
    edge_value = _edge_value(G, edge_attr)
    node_value = _node_value(G, node_attr)
    if rc_order is None:
        ordering = list({node_value(n) for n in G})
    else:
        ordering = rc_order
    N = len(ordering)
    undirected = not G.is_directed()
    index = dict(zip(ordering, range(N)))
    M = sp.sparse.lil_array((N, N), dtype=dtype)
    seen = set()
    for (u, nbrdict) in G.adjacency():
        for v in nbrdict:
            (i, j) = (index[node_value(u)], index[node_value(v)])
            if v not in seen:
                M[i, j] += edge_value(u, v)
                if undirected:
                    M[j, i] = M[i, j]
        if undirected:
            seen.add(u)
    if normalized:
        M *= 1 / M.sum(axis=1)[:, np.newaxis]
    if rc_order is None:
        return (M, ordering)
    else:
        return M