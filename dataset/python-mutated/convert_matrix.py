"""Functions to convert NetworkX graphs to and from common data containers
like numpy arrays, scipy sparse arrays, and pandas DataFrames.

The preferred way of converting data to a NetworkX graph is through the
graph constructor.  The constructor calls the `~networkx.convert.to_networkx_graph`
function which attempts to guess the input type and convert it automatically.

Examples
--------
Create a 10 node random graph from a numpy array

>>> import numpy as np
>>> rng = np.random.default_rng()
>>> a = rng.integers(low=0, high=2, size=(10, 10))
>>> DG = nx.from_numpy_array(a, create_using=nx.DiGraph)

or equivalently:

>>> DG = nx.DiGraph(a)

which calls `from_numpy_array` internally based on the type of ``a``.

See Also
--------
nx_agraph, nx_pydot
"""
import itertools
from collections import defaultdict
import networkx as nx
from networkx.utils import not_implemented_for
__all__ = ['from_pandas_adjacency', 'to_pandas_adjacency', 'from_pandas_edgelist', 'to_pandas_edgelist', 'from_scipy_sparse_array', 'to_scipy_sparse_array', 'from_numpy_array', 'to_numpy_array']

@nx._dispatch(edge_attrs='weight')
def to_pandas_adjacency(G, nodelist=None, dtype=None, order=None, multigraph_weight=sum, weight='weight', nonedge=0.0):
    if False:
        print('Hello World!')
    "Returns the graph adjacency matrix as a Pandas DataFrame.\n\n    Parameters\n    ----------\n    G : graph\n        The NetworkX graph used to construct the Pandas DataFrame.\n\n    nodelist : list, optional\n       The rows and columns are ordered according to the nodes in `nodelist`.\n       If `nodelist` is None, then the ordering is produced by G.nodes().\n\n    multigraph_weight : {sum, min, max}, optional\n        An operator that determines how weights in multigraphs are handled.\n        The default is to sum the weights of the multiple edges.\n\n    weight : string or None, optional\n        The edge attribute that holds the numerical value used for\n        the edge weight.  If an edge does not have that attribute, then the\n        value 1 is used instead.\n\n    nonedge : float, optional\n        The matrix values corresponding to nonedges are typically set to zero.\n        However, this could be undesirable if there are matrix values\n        corresponding to actual edges that also have the value zero. If so,\n        one might prefer nonedges to have some other value, such as nan.\n\n    Returns\n    -------\n    df : Pandas DataFrame\n       Graph adjacency matrix\n\n    Notes\n    -----\n    For directed graphs, entry i,j corresponds to an edge from i to j.\n\n    The DataFrame entries are assigned to the weight edge attribute. When\n    an edge does not have a weight attribute, the value of the entry is set to\n    the number 1.  For multiple (parallel) edges, the values of the entries\n    are determined by the 'multigraph_weight' parameter.  The default is to\n    sum the weight attributes for each of the parallel edges.\n\n    When `nodelist` does not contain every node in `G`, the matrix is built\n    from the subgraph of `G` that is induced by the nodes in `nodelist`.\n\n    The convention used for self-loop edges in graphs is to assign the\n    diagonal matrix entry value to the weight attribute of the edge\n    (or the number 1 if the edge has no weight attribute).  If the\n    alternate convention of doubling the edge weight is desired the\n    resulting Pandas DataFrame can be modified as follows:\n\n    >>> import pandas as pd\n    >>> pd.options.display.max_columns = 20\n    >>> import numpy as np\n    >>> G = nx.Graph([(1, 1)])\n    >>> df = nx.to_pandas_adjacency(G, dtype=int)\n    >>> df\n       1\n    1  1\n    >>> df.values[np.diag_indices_from(df)] *= 2\n    >>> df\n       1\n    1  2\n\n    Examples\n    --------\n    >>> G = nx.MultiDiGraph()\n    >>> G.add_edge(0, 1, weight=2)\n    0\n    >>> G.add_edge(1, 0)\n    0\n    >>> G.add_edge(2, 2, weight=3)\n    0\n    >>> G.add_edge(2, 2)\n    1\n    >>> nx.to_pandas_adjacency(G, nodelist=[0, 1, 2], dtype=int)\n       0  1  2\n    0  0  2  0\n    1  1  0  0\n    2  0  0  4\n\n    "
    import pandas as pd
    M = to_numpy_array(G, nodelist=nodelist, dtype=dtype, order=order, multigraph_weight=multigraph_weight, weight=weight, nonedge=nonedge)
    if nodelist is None:
        nodelist = list(G)
    return pd.DataFrame(data=M, index=nodelist, columns=nodelist)

@nx._dispatch(graphs=None)
def from_pandas_adjacency(df, create_using=None):
    if False:
        for i in range(10):
            print('nop')
    'Returns a graph from Pandas DataFrame.\n\n    The Pandas DataFrame is interpreted as an adjacency matrix for the graph.\n\n    Parameters\n    ----------\n    df : Pandas DataFrame\n      An adjacency matrix representation of a graph\n\n    create_using : NetworkX graph constructor, optional (default=nx.Graph)\n       Graph type to create. If graph instance, then cleared before populated.\n\n    Notes\n    -----\n    For directed graphs, explicitly mention create_using=nx.DiGraph,\n    and entry i,j of df corresponds to an edge from i to j.\n\n    If `df` has a single data type for each entry it will be converted to an\n    appropriate Python data type.\n\n    If you have node attributes stored in a separate dataframe `df_nodes`,\n    you can load those attributes to the graph `G` using the following code:\n\n    ```\n    df_nodes = pd.DataFrame({"node_id": [1, 2, 3], "attribute1": ["A", "B", "C"]})\n    G.add_nodes_from((n, dict(d)) for n, d in df_nodes.iterrows())\n    ```\n\n    If `df` has a user-specified compound data type the names\n    of the data fields will be used as attribute keys in the resulting\n    NetworkX graph.\n\n    See Also\n    --------\n    to_pandas_adjacency\n\n    Examples\n    --------\n    Simple integer weights on edges:\n\n    >>> import pandas as pd\n    >>> pd.options.display.max_columns = 20\n    >>> df = pd.DataFrame([[1, 1], [2, 1]])\n    >>> df\n       0  1\n    0  1  1\n    1  2  1\n    >>> G = nx.from_pandas_adjacency(df)\n    >>> G.name = "Graph from pandas adjacency matrix"\n    >>> print(G)\n    Graph named \'Graph from pandas adjacency matrix\' with 2 nodes and 3 edges\n    '
    try:
        df = df[df.index]
    except Exception as err:
        missing = list(set(df.index).difference(set(df.columns)))
        msg = f'{missing} not in columns'
        raise nx.NetworkXError('Columns must match Indices.', msg) from err
    A = df.values
    G = from_numpy_array(A, create_using=create_using)
    nx.relabel.relabel_nodes(G, dict(enumerate(df.columns)), copy=False)
    return G

@nx._dispatch(preserve_edge_attrs=True)
def to_pandas_edgelist(G, source='source', target='target', nodelist=None, dtype=None, edge_key=None):
    if False:
        return 10
    'Returns the graph edge list as a Pandas DataFrame.\n\n    Parameters\n    ----------\n    G : graph\n        The NetworkX graph used to construct the Pandas DataFrame.\n\n    source : str or int, optional\n        A valid column name (string or integer) for the source nodes (for the\n        directed case).\n\n    target : str or int, optional\n        A valid column name (string or integer) for the target nodes (for the\n        directed case).\n\n    nodelist : list, optional\n       Use only nodes specified in nodelist\n\n    dtype : dtype, default None\n        Use to create the DataFrame. Data type to force.\n        Only a single dtype is allowed. If None, infer.\n\n    edge_key : str or int or None, optional (default=None)\n        A valid column name (string or integer) for the edge keys (for the\n        multigraph case). If None, edge keys are not stored in the DataFrame.\n\n    Returns\n    -------\n    df : Pandas DataFrame\n       Graph edge list\n\n    Examples\n    --------\n    >>> G = nx.Graph(\n    ...     [\n    ...         ("A", "B", {"cost": 1, "weight": 7}),\n    ...         ("C", "E", {"cost": 9, "weight": 10}),\n    ...     ]\n    ... )\n    >>> df = nx.to_pandas_edgelist(G, nodelist=["A", "C"])\n    >>> df[["source", "target", "cost", "weight"]]\n      source target  cost  weight\n    0      A      B     1       7\n    1      C      E     9      10\n\n    >>> G = nx.MultiGraph([(\'A\', \'B\', {\'cost\': 1}), (\'A\', \'B\', {\'cost\': 9})])\n    >>> df = nx.to_pandas_edgelist(G, nodelist=[\'A\', \'C\'], edge_key=\'ekey\')\n    >>> df[[\'source\', \'target\', \'cost\', \'ekey\']]\n      source target  cost  ekey\n    0      A      B     1     0\n    1      A      B     9     1\n\n    '
    import pandas as pd
    if nodelist is None:
        edgelist = G.edges(data=True)
    else:
        edgelist = G.edges(nodelist, data=True)
    source_nodes = [s for (s, _, _) in edgelist]
    target_nodes = [t for (_, t, _) in edgelist]
    all_attrs = set().union(*(d.keys() for (_, _, d) in edgelist))
    if source in all_attrs:
        raise nx.NetworkXError(f'Source name {source!r} is an edge attr name')
    if target in all_attrs:
        raise nx.NetworkXError(f'Target name {target!r} is an edge attr name')
    nan = float('nan')
    edge_attr = {k: [d.get(k, nan) for (_, _, d) in edgelist] for k in all_attrs}
    if G.is_multigraph() and edge_key is not None:
        if edge_key in all_attrs:
            raise nx.NetworkXError(f'Edge key name {edge_key!r} is an edge attr name')
        edge_keys = [k for (_, _, k) in G.edges(keys=True)]
        edgelistdict = {source: source_nodes, target: target_nodes, edge_key: edge_keys}
    else:
        edgelistdict = {source: source_nodes, target: target_nodes}
    edgelistdict.update(edge_attr)
    return pd.DataFrame(edgelistdict, dtype=dtype)

@nx._dispatch(graphs=None)
def from_pandas_edgelist(df, source='source', target='target', edge_attr=None, create_using=None, edge_key=None):
    if False:
        for i in range(10):
            print('nop')
    'Returns a graph from Pandas DataFrame containing an edge list.\n\n    The Pandas DataFrame should contain at least two columns of node names and\n    zero or more columns of edge attributes. Each row will be processed as one\n    edge instance.\n\n    Note: This function iterates over DataFrame.values, which is not\n    guaranteed to retain the data type across columns in the row. This is only\n    a problem if your row is entirely numeric and a mix of ints and floats. In\n    that case, all values will be returned as floats. See the\n    DataFrame.iterrows documentation for an example.\n\n    Parameters\n    ----------\n    df : Pandas DataFrame\n        An edge list representation of a graph\n\n    source : str or int\n        A valid column name (string or integer) for the source nodes (for the\n        directed case).\n\n    target : str or int\n        A valid column name (string or integer) for the target nodes (for the\n        directed case).\n\n    edge_attr : str or int, iterable, True, or None\n        A valid column name (str or int) or iterable of column names that are\n        used to retrieve items and add them to the graph as edge attributes.\n        If `True`, all of the remaining columns will be added.\n        If `None`, no edge attributes are added to the graph.\n\n    create_using : NetworkX graph constructor, optional (default=nx.Graph)\n        Graph type to create. If graph instance, then cleared before populated.\n\n    edge_key : str or None, optional (default=None)\n        A valid column name for the edge keys (for a MultiGraph). The values in\n        this column are used for the edge keys when adding edges if create_using\n        is a multigraph.\n\n    If you have node attributes stored in a separate dataframe `df_nodes`,\n    you can load those attributes to the graph `G` using the following code:\n\n    ```\n    df_nodes = pd.DataFrame({"node_id": [1, 2, 3], "attribute1": ["A", "B", "C"]})\n    G.add_nodes_from((n, dict(d)) for n, d in df_nodes.iterrows())\n    ```\n\n    See Also\n    --------\n    to_pandas_edgelist\n\n    Examples\n    --------\n    Simple integer weights on edges:\n\n    >>> import pandas as pd\n    >>> pd.options.display.max_columns = 20\n    >>> import numpy as np\n    >>> rng = np.random.RandomState(seed=5)\n    >>> ints = rng.randint(1, 11, size=(3, 2))\n    >>> a = ["A", "B", "C"]\n    >>> b = ["D", "A", "E"]\n    >>> df = pd.DataFrame(ints, columns=["weight", "cost"])\n    >>> df[0] = a\n    >>> df["b"] = b\n    >>> df[["weight", "cost", 0, "b"]]\n       weight  cost  0  b\n    0       4     7  A  D\n    1       7     1  B  A\n    2      10     9  C  E\n    >>> G = nx.from_pandas_edgelist(df, 0, "b", ["weight", "cost"])\n    >>> G["E"]["C"]["weight"]\n    10\n    >>> G["E"]["C"]["cost"]\n    9\n    >>> edges = pd.DataFrame(\n    ...     {\n    ...         "source": [0, 1, 2],\n    ...         "target": [2, 2, 3],\n    ...         "weight": [3, 4, 5],\n    ...         "color": ["red", "blue", "blue"],\n    ...     }\n    ... )\n    >>> G = nx.from_pandas_edgelist(edges, edge_attr=True)\n    >>> G[0][2]["color"]\n    \'red\'\n\n    Build multigraph with custom keys:\n\n    >>> edges = pd.DataFrame(\n    ...     {\n    ...         "source": [0, 1, 2, 0],\n    ...         "target": [2, 2, 3, 2],\n    ...         "my_edge_key": ["A", "B", "C", "D"],\n    ...         "weight": [3, 4, 5, 6],\n    ...         "color": ["red", "blue", "blue", "blue"],\n    ...     }\n    ... )\n    >>> G = nx.from_pandas_edgelist(\n    ...     edges,\n    ...     edge_key="my_edge_key",\n    ...     edge_attr=["weight", "color"],\n    ...     create_using=nx.MultiGraph(),\n    ... )\n    >>> G[0][2]\n    AtlasView({\'A\': {\'weight\': 3, \'color\': \'red\'}, \'D\': {\'weight\': 6, \'color\': \'blue\'}})\n\n\n    '
    g = nx.empty_graph(0, create_using)
    if edge_attr is None:
        g.add_edges_from(zip(df[source], df[target]))
        return g
    reserved_columns = [source, target]
    attr_col_headings = []
    attribute_data = []
    if edge_attr is True:
        attr_col_headings = [c for c in df.columns if c not in reserved_columns]
    elif isinstance(edge_attr, list | tuple):
        attr_col_headings = edge_attr
    else:
        attr_col_headings = [edge_attr]
    if len(attr_col_headings) == 0:
        raise nx.NetworkXError(f'Invalid edge_attr argument: No columns found with name: {attr_col_headings}')
    try:
        attribute_data = zip(*[df[col] for col in attr_col_headings])
    except (KeyError, TypeError) as err:
        msg = f'Invalid edge_attr argument: {edge_attr}'
        raise nx.NetworkXError(msg) from err
    if g.is_multigraph():
        if edge_key is not None:
            try:
                multigraph_edge_keys = df[edge_key]
                attribute_data = zip(attribute_data, multigraph_edge_keys)
            except (KeyError, TypeError) as err:
                msg = f'Invalid edge_key argument: {edge_key}'
                raise nx.NetworkXError(msg) from err
        for (s, t, attrs) in zip(df[source], df[target], attribute_data):
            if edge_key is not None:
                (attrs, multigraph_edge_key) = attrs
                key = g.add_edge(s, t, key=multigraph_edge_key)
            else:
                key = g.add_edge(s, t)
            g[s][t][key].update(zip(attr_col_headings, attrs))
    else:
        for (s, t, attrs) in zip(df[source], df[target], attribute_data):
            g.add_edge(s, t)
            g[s][t].update(zip(attr_col_headings, attrs))
    return g

@nx._dispatch(edge_attrs='weight')
def to_scipy_sparse_array(G, nodelist=None, dtype=None, weight='weight', format='csr'):
    if False:
        while True:
            i = 10
    'Returns the graph adjacency matrix as a SciPy sparse array.\n\n    Parameters\n    ----------\n    G : graph\n        The NetworkX graph used to construct the sparse matrix.\n\n    nodelist : list, optional\n       The rows and columns are ordered according to the nodes in `nodelist`.\n       If `nodelist` is None, then the ordering is produced by G.nodes().\n\n    dtype : NumPy data-type, optional\n        A valid NumPy dtype used to initialize the array. If None, then the\n        NumPy default is used.\n\n    weight : string or None   optional (default=\'weight\')\n        The edge attribute that holds the numerical value used for\n        the edge weight.  If None then all edge weights are 1.\n\n    format : str in {\'bsr\', \'csr\', \'csc\', \'coo\', \'lil\', \'dia\', \'dok\'}\n        The type of the matrix to be returned (default \'csr\').  For\n        some algorithms different implementations of sparse matrices\n        can perform better.  See [1]_ for details.\n\n    Returns\n    -------\n    A : SciPy sparse array\n       Graph adjacency matrix.\n\n    Notes\n    -----\n    For directed graphs, matrix entry i,j corresponds to an edge from i to j.\n\n    The matrix entries are populated using the edge attribute held in\n    parameter weight. When an edge does not have that attribute, the\n    value of the entry is 1.\n\n    For multiple edges the matrix values are the sums of the edge weights.\n\n    When `nodelist` does not contain every node in `G`, the adjacency matrix\n    is built from the subgraph of `G` that is induced by the nodes in\n    `nodelist`.\n\n    The convention used for self-loop edges in graphs is to assign the\n    diagonal matrix entry value to the weight attribute of the edge\n    (or the number 1 if the edge has no weight attribute).  If the\n    alternate convention of doubling the edge weight is desired the\n    resulting SciPy sparse array can be modified as follows:\n\n    >>> G = nx.Graph([(1, 1)])\n    >>> A = nx.to_scipy_sparse_array(G)\n    >>> print(A.todense())\n    [[1]]\n    >>> A.setdiag(A.diagonal() * 2)\n    >>> print(A.toarray())\n    [[2]]\n\n    Examples\n    --------\n    >>> G = nx.MultiDiGraph()\n    >>> G.add_edge(0, 1, weight=2)\n    0\n    >>> G.add_edge(1, 0)\n    0\n    >>> G.add_edge(2, 2, weight=3)\n    0\n    >>> G.add_edge(2, 2)\n    1\n    >>> S = nx.to_scipy_sparse_array(G, nodelist=[0, 1, 2])\n    >>> print(S.toarray())\n    [[0 2 0]\n     [1 0 0]\n     [0 0 4]]\n\n    References\n    ----------\n    .. [1] Scipy Dev. References, "Sparse Matrices",\n       https://docs.scipy.org/doc/scipy/reference/sparse.html\n    '
    import scipy as sp
    if len(G) == 0:
        raise nx.NetworkXError('Graph has no nodes or edges')
    if nodelist is None:
        nodelist = list(G)
        nlen = len(G)
    else:
        nlen = len(nodelist)
        if nlen == 0:
            raise nx.NetworkXError('nodelist has no nodes')
        nodeset = set(G.nbunch_iter(nodelist))
        if nlen != len(nodeset):
            for n in nodelist:
                if n not in G:
                    raise nx.NetworkXError(f'Node {n} in nodelist is not in G')
            raise nx.NetworkXError('nodelist contains duplicates.')
        if nlen < len(G):
            G = G.subgraph(nodelist)
    index = dict(zip(nodelist, range(nlen)))
    coefficients = zip(*((index[u], index[v], wt) for (u, v, wt) in G.edges(data=weight, default=1)))
    try:
        (row, col, data) = coefficients
    except ValueError:
        (row, col, data) = ([], [], [])
    if G.is_directed():
        A = sp.sparse.coo_array((data, (row, col)), shape=(nlen, nlen), dtype=dtype)
    else:
        d = data + data
        r = row + col
        c = col + row
        selfloops = list(nx.selfloop_edges(G, data=weight, default=1))
        if selfloops:
            (diag_index, diag_data) = zip(*((index[u], -wt) for (u, v, wt) in selfloops))
            d += diag_data
            r += diag_index
            c += diag_index
        A = sp.sparse.coo_array((d, (r, c)), shape=(nlen, nlen), dtype=dtype)
    try:
        return A.asformat(format)
    except ValueError as err:
        raise nx.NetworkXError(f'Unknown sparse matrix format: {format}') from err

def _csr_gen_triples(A):
    if False:
        while True:
            i = 10
    'Converts a SciPy sparse array in **Compressed Sparse Row** format to\n    an iterable of weighted edge triples.\n\n    '
    nrows = A.shape[0]
    (data, indices, indptr) = (A.data, A.indices, A.indptr)
    for i in range(nrows):
        for j in range(indptr[i], indptr[i + 1]):
            yield (i, int(indices[j]), data[j])

def _csc_gen_triples(A):
    if False:
        i = 10
        return i + 15
    'Converts a SciPy sparse array in **Compressed Sparse Column** format to\n    an iterable of weighted edge triples.\n\n    '
    ncols = A.shape[1]
    (data, indices, indptr) = (A.data, A.indices, A.indptr)
    for i in range(ncols):
        for j in range(indptr[i], indptr[i + 1]):
            yield (int(indices[j]), i, data[j])

def _coo_gen_triples(A):
    if False:
        for i in range(10):
            print('nop')
    'Converts a SciPy sparse array in **Coordinate** format to an iterable\n    of weighted edge triples.\n\n    '
    return ((int(i), int(j), d) for (i, j, d) in zip(A.row, A.col, A.data))

def _dok_gen_triples(A):
    if False:
        return 10
    'Converts a SciPy sparse array in **Dictionary of Keys** format to an\n    iterable of weighted edge triples.\n\n    '
    for ((r, c), v) in A.items():
        yield (r, c, v)

def _generate_weighted_edges(A):
    if False:
        i = 10
        return i + 15
    'Returns an iterable over (u, v, w) triples, where u and v are adjacent\n    vertices and w is the weight of the edge joining u and v.\n\n    `A` is a SciPy sparse array (in any format).\n\n    '
    if A.format == 'csr':
        return _csr_gen_triples(A)
    if A.format == 'csc':
        return _csc_gen_triples(A)
    if A.format == 'dok':
        return _dok_gen_triples(A)
    return _coo_gen_triples(A.tocoo())

@nx._dispatch(graphs=None)
def from_scipy_sparse_array(A, parallel_edges=False, create_using=None, edge_attribute='weight'):
    if False:
        return 10
    "Creates a new graph from an adjacency matrix given as a SciPy sparse\n    array.\n\n    Parameters\n    ----------\n    A: scipy.sparse array\n      An adjacency matrix representation of a graph\n\n    parallel_edges : Boolean\n      If this is True, `create_using` is a multigraph, and `A` is an\n      integer matrix, then entry *(i, j)* in the matrix is interpreted as the\n      number of parallel edges joining vertices *i* and *j* in the graph.\n      If it is False, then the entries in the matrix are interpreted as\n      the weight of a single edge joining the vertices.\n\n    create_using : NetworkX graph constructor, optional (default=nx.Graph)\n       Graph type to create. If graph instance, then cleared before populated.\n\n    edge_attribute: string\n       Name of edge attribute to store matrix numeric value. The data will\n       have the same type as the matrix entry (int, float, (real,imag)).\n\n    Notes\n    -----\n    For directed graphs, explicitly mention create_using=nx.DiGraph,\n    and entry i,j of A corresponds to an edge from i to j.\n\n    If `create_using` is :class:`networkx.MultiGraph` or\n    :class:`networkx.MultiDiGraph`, `parallel_edges` is True, and the\n    entries of `A` are of type :class:`int`, then this function returns a\n    multigraph (constructed from `create_using`) with parallel edges.\n    In this case, `edge_attribute` will be ignored.\n\n    If `create_using` indicates an undirected multigraph, then only the edges\n    indicated by the upper triangle of the matrix `A` will be added to the\n    graph.\n\n    Examples\n    --------\n    >>> import scipy as sp\n    >>> A = sp.sparse.eye(2, 2, 1)\n    >>> G = nx.from_scipy_sparse_array(A)\n\n    If `create_using` indicates a multigraph and the matrix has only integer\n    entries and `parallel_edges` is False, then the entries will be treated\n    as weights for edges joining the nodes (without creating parallel edges):\n\n    >>> A = sp.sparse.csr_array([[1, 1], [1, 2]])\n    >>> G = nx.from_scipy_sparse_array(A, create_using=nx.MultiGraph)\n    >>> G[1][1]\n    AtlasView({0: {'weight': 2}})\n\n    If `create_using` indicates a multigraph and the matrix has only integer\n    entries and `parallel_edges` is True, then the entries will be treated\n    as the number of parallel edges joining those two vertices:\n\n    >>> A = sp.sparse.csr_array([[1, 1], [1, 2]])\n    >>> G = nx.from_scipy_sparse_array(\n    ...     A, parallel_edges=True, create_using=nx.MultiGraph\n    ... )\n    >>> G[1][1]\n    AtlasView({0: {'weight': 1}, 1: {'weight': 1}})\n\n    "
    G = nx.empty_graph(0, create_using)
    (n, m) = A.shape
    if n != m:
        raise nx.NetworkXError(f'Adjacency matrix not square: nx,ny={A.shape}')
    G.add_nodes_from(range(n))
    triples = _generate_weighted_edges(A)
    if A.dtype.kind in ('i', 'u') and G.is_multigraph() and parallel_edges:
        chain = itertools.chain.from_iterable
        triples = chain((((u, v, 1) for d in range(w)) for (u, v, w) in triples))
    if G.is_multigraph() and (not G.is_directed()):
        triples = ((u, v, d) for (u, v, d) in triples if u <= v)
    G.add_weighted_edges_from(triples, weight=edge_attribute)
    return G

@nx._dispatch(edge_attrs='weight')
def to_numpy_array(G, nodelist=None, dtype=None, order=None, multigraph_weight=sum, weight='weight', nonedge=0.0):
    if False:
        return 10
    'Returns the graph adjacency matrix as a NumPy array.\n\n    Parameters\n    ----------\n    G : graph\n        The NetworkX graph used to construct the NumPy array.\n\n    nodelist : list, optional\n        The rows and columns are ordered according to the nodes in `nodelist`.\n        If `nodelist` is ``None``, then the ordering is produced by ``G.nodes()``.\n\n    dtype : NumPy data type, optional\n        A NumPy data type used to initialize the array. If None, then the NumPy\n        default is used. The dtype can be structured if `weight=None`, in which\n        case the dtype field names are used to look up edge attributes. The\n        result is a structured array where each named field in the dtype\n        corresponds to the adjacency for that edge attribute. See examples for\n        details.\n\n    order : {\'C\', \'F\'}, optional\n        Whether to store multidimensional data in C- or Fortran-contiguous\n        (row- or column-wise) order in memory. If None, then the NumPy default\n        is used.\n\n    multigraph_weight : callable, optional\n        An function that determines how weights in multigraphs are handled.\n        The function should accept a sequence of weights and return a single\n        value. The default is to sum the weights of the multiple edges.\n\n    weight : string or None optional (default = \'weight\')\n        The edge attribute that holds the numerical value used for\n        the edge weight. If an edge does not have that attribute, then the\n        value 1 is used instead. `weight` must be ``None`` if a structured\n        dtype is used.\n\n    nonedge : array_like (default = 0.0)\n        The value used to represent non-edges in the adjacency matrix.\n        The array values corresponding to nonedges are typically set to zero.\n        However, this could be undesirable if there are array values\n        corresponding to actual edges that also have the value zero. If so,\n        one might prefer nonedges to have some other value, such as ``nan``.\n\n    Returns\n    -------\n    A : NumPy ndarray\n        Graph adjacency matrix\n\n    Raises\n    ------\n    NetworkXError\n        If `dtype` is a structured dtype and `G` is a multigraph\n    ValueError\n        If `dtype` is a structured dtype and `weight` is not `None`\n\n    See Also\n    --------\n    from_numpy_array\n\n    Notes\n    -----\n    For directed graphs, entry ``i, j`` corresponds to an edge from ``i`` to ``j``.\n\n    Entries in the adjacency matrix are given by the `weight` edge attribute.\n    When an edge does not have a weight attribute, the value of the entry is\n    set to the number 1.  For multiple (parallel) edges, the values of the\n    entries are determined by the `multigraph_weight` parameter. The default is\n    to sum the weight attributes for each of the parallel edges.\n\n    When `nodelist` does not contain every node in `G`, the adjacency matrix is\n    built from the subgraph of `G` that is induced by the nodes in `nodelist`.\n\n    The convention used for self-loop edges in graphs is to assign the\n    diagonal array entry value to the weight attribute of the edge\n    (or the number 1 if the edge has no weight attribute). If the\n    alternate convention of doubling the edge weight is desired the\n    resulting NumPy array can be modified as follows:\n\n    >>> import numpy as np\n    >>> G = nx.Graph([(1, 1)])\n    >>> A = nx.to_numpy_array(G)\n    >>> A\n    array([[1.]])\n    >>> A[np.diag_indices_from(A)] *= 2\n    >>> A\n    array([[2.]])\n\n    Examples\n    --------\n    >>> G = nx.MultiDiGraph()\n    >>> G.add_edge(0, 1, weight=2)\n    0\n    >>> G.add_edge(1, 0)\n    0\n    >>> G.add_edge(2, 2, weight=3)\n    0\n    >>> G.add_edge(2, 2)\n    1\n    >>> nx.to_numpy_array(G, nodelist=[0, 1, 2])\n    array([[0., 2., 0.],\n           [1., 0., 0.],\n           [0., 0., 4.]])\n\n    When `nodelist` argument is used, nodes of `G` which do not appear in the `nodelist`\n    and their edges are not included in the adjacency matrix. Here is an example:\n\n    >>> G = nx.Graph()\n    >>> G.add_edge(3, 1)\n    >>> G.add_edge(2, 0)\n    >>> G.add_edge(2, 1)\n    >>> G.add_edge(3, 0)\n    >>> nx.to_numpy_array(G, nodelist=[1, 2, 3])\n    array([[0., 1., 1.],\n           [1., 0., 0.],\n           [1., 0., 0.]])\n\n    This function can also be used to create adjacency matrices for multiple\n    edge attributes with structured dtypes:\n\n    >>> G = nx.Graph()\n    >>> G.add_edge(0, 1, weight=10)\n    >>> G.add_edge(1, 2, cost=5)\n    >>> G.add_edge(2, 3, weight=3, cost=-4.0)\n    >>> dtype = np.dtype([("weight", int), ("cost", float)])\n    >>> A = nx.to_numpy_array(G, dtype=dtype, weight=None)\n    >>> A["weight"]\n    array([[ 0, 10,  0,  0],\n           [10,  0,  1,  0],\n           [ 0,  1,  0,  3],\n           [ 0,  0,  3,  0]])\n    >>> A["cost"]\n    array([[ 0.,  1.,  0.,  0.],\n           [ 1.,  0.,  5.,  0.],\n           [ 0.,  5.,  0., -4.],\n           [ 0.,  0., -4.,  0.]])\n\n    As stated above, the argument "nonedge" is useful especially when there are\n    actually edges with weight 0 in the graph. Setting a nonedge value different than 0,\n    makes it much clearer to differentiate such 0-weighted edges and actual nonedge values.\n\n    >>> G = nx.Graph()\n    >>> G.add_edge(3, 1, weight=2)\n    >>> G.add_edge(2, 0, weight=0)\n    >>> G.add_edge(2, 1, weight=0)\n    >>> G.add_edge(3, 0, weight=1)\n    >>> nx.to_numpy_array(G, nonedge=-1.)\n    array([[-1.,  2., -1.,  1.],\n           [ 2., -1.,  0., -1.],\n           [-1.,  0., -1.,  0.],\n           [ 1., -1.,  0., -1.]])\n    '
    import numpy as np
    if nodelist is None:
        nodelist = list(G)
    nlen = len(nodelist)
    nodeset = set(nodelist)
    if nodeset - set(G):
        raise nx.NetworkXError(f'Nodes {nodeset - set(G)} in nodelist is not in G')
    if len(nodeset) < nlen:
        raise nx.NetworkXError('nodelist contains duplicates.')
    A = np.full((nlen, nlen), fill_value=nonedge, dtype=dtype, order=order)
    if nlen == 0 or G.number_of_edges() == 0:
        return A
    edge_attrs = None
    if A.dtype.names:
        if weight is None:
            edge_attrs = dtype.names
        else:
            raise ValueError('Specifying `weight` not supported for structured dtypes\n.To create adjacency matrices from structured dtypes, use `weight=None`.')
    idx = dict(zip(nodelist, range(nlen)))
    if len(nodelist) < len(G):
        G = G.subgraph(nodelist).copy()
    if G.is_multigraph():
        if edge_attrs:
            raise nx.NetworkXError('Structured arrays are not supported for MultiGraphs')
        d = defaultdict(list)
        for (u, v, wt) in G.edges(data=weight, default=1.0):
            d[idx[u], idx[v]].append(wt)
        (i, j) = np.array(list(d.keys())).T
        wts = [multigraph_weight(ws) for ws in d.values()]
    else:
        (i, j, wts) = ([], [], [])
        if edge_attrs:
            for (u, v, data) in G.edges(data=True):
                i.append(idx[u])
                j.append(idx[v])
                wts.append(data)
            for attr in edge_attrs:
                attr_data = [wt.get(attr, 1.0) for wt in wts]
                A[attr][i, j] = attr_data
                if not G.is_directed():
                    A[attr][j, i] = attr_data
            return A
        for (u, v, wt) in G.edges(data=weight, default=1.0):
            i.append(idx[u])
            j.append(idx[v])
            wts.append(wt)
    A[i, j] = wts
    if not G.is_directed():
        A[j, i] = wts
    return A

@nx._dispatch(graphs=None)
def from_numpy_array(A, parallel_edges=False, create_using=None, edge_attr='weight'):
    if False:
        print('Hello World!')
    'Returns a graph from a 2D NumPy array.\n\n    The 2D NumPy array is interpreted as an adjacency matrix for the graph.\n\n    Parameters\n    ----------\n    A : a 2D numpy.ndarray\n        An adjacency matrix representation of a graph\n\n    parallel_edges : Boolean\n        If this is True, `create_using` is a multigraph, and `A` is an\n        integer array, then entry *(i, j)* in the array is interpreted as the\n        number of parallel edges joining vertices *i* and *j* in the graph.\n        If it is False, then the entries in the array are interpreted as\n        the weight of a single edge joining the vertices.\n\n    create_using : NetworkX graph constructor, optional (default=nx.Graph)\n       Graph type to create. If graph instance, then cleared before populated.\n\n    edge_attr : String, optional (default="weight")\n        The attribute to which the array values are assigned on each edge. If\n        it is None, edge attributes will not be assigned.\n\n    Notes\n    -----\n    For directed graphs, explicitly mention create_using=nx.DiGraph,\n    and entry i,j of A corresponds to an edge from i to j.\n\n    If `create_using` is :class:`networkx.MultiGraph` or\n    :class:`networkx.MultiDiGraph`, `parallel_edges` is True, and the\n    entries of `A` are of type :class:`int`, then this function returns a\n    multigraph (of the same type as `create_using`) with parallel edges.\n\n    If `create_using` indicates an undirected multigraph, then only the edges\n    indicated by the upper triangle of the array `A` will be added to the\n    graph.\n\n    If `edge_attr` is Falsy (False or None), edge attributes will not be\n    assigned, and the array data will be treated like a binary mask of\n    edge presence or absence. Otherwise, the attributes will be assigned\n    as follows:\n\n    If the NumPy array has a single data type for each array entry it\n    will be converted to an appropriate Python data type.\n\n    If the NumPy array has a user-specified compound data type the names\n    of the data fields will be used as attribute keys in the resulting\n    NetworkX graph.\n\n    See Also\n    --------\n    to_numpy_array\n\n    Examples\n    --------\n    Simple integer weights on edges:\n\n    >>> import numpy as np\n    >>> A = np.array([[1, 1], [2, 1]])\n    >>> G = nx.from_numpy_array(A)\n    >>> G.edges(data=True)\n    EdgeDataView([(0, 0, {\'weight\': 1}), (0, 1, {\'weight\': 2}), (1, 1, {\'weight\': 1})])\n\n    If `create_using` indicates a multigraph and the array has only integer\n    entries and `parallel_edges` is False, then the entries will be treated\n    as weights for edges joining the nodes (without creating parallel edges):\n\n    >>> A = np.array([[1, 1], [1, 2]])\n    >>> G = nx.from_numpy_array(A, create_using=nx.MultiGraph)\n    >>> G[1][1]\n    AtlasView({0: {\'weight\': 2}})\n\n    If `create_using` indicates a multigraph and the array has only integer\n    entries and `parallel_edges` is True, then the entries will be treated\n    as the number of parallel edges joining those two vertices:\n\n    >>> A = np.array([[1, 1], [1, 2]])\n    >>> temp = nx.MultiGraph()\n    >>> G = nx.from_numpy_array(A, parallel_edges=True, create_using=temp)\n    >>> G[1][1]\n    AtlasView({0: {\'weight\': 1}, 1: {\'weight\': 1}})\n\n    User defined compound data type on edges:\n\n    >>> dt = [("weight", float), ("cost", int)]\n    >>> A = np.array([[(1.0, 2)]], dtype=dt)\n    >>> G = nx.from_numpy_array(A)\n    >>> G.edges()\n    EdgeView([(0, 0)])\n    >>> G[0][0]["cost"]\n    2\n    >>> G[0][0]["weight"]\n    1.0\n\n    '
    kind_to_python_type = {'f': float, 'i': int, 'u': int, 'b': bool, 'c': complex, 'S': str, 'U': str, 'V': 'void'}
    G = nx.empty_graph(0, create_using)
    if A.ndim != 2:
        raise nx.NetworkXError(f'Input array must be 2D, not {A.ndim}')
    (n, m) = A.shape
    if n != m:
        raise nx.NetworkXError(f'Adjacency matrix not square: nx,ny={A.shape}')
    dt = A.dtype
    try:
        python_type = kind_to_python_type[dt.kind]
    except Exception as err:
        raise TypeError(f'Unknown numpy data type: {dt}') from err
    G.add_nodes_from(range(n))
    edges = ((int(e[0]), int(e[1])) for e in zip(*A.nonzero()))
    if python_type == 'void':
        fields = sorted(((offset, dtype, name) for (name, (dtype, offset)) in A.dtype.fields.items()))
        triples = ((u, v, {} if edge_attr in [False, None] else {name: kind_to_python_type[dtype.kind](val) for ((_, dtype, name), val) in zip(fields, A[u, v])}) for (u, v) in edges)
    elif python_type is int and G.is_multigraph() and parallel_edges:
        chain = itertools.chain.from_iterable
        if edge_attr in [False, None]:
            triples = chain((((u, v, {}) for d in range(A[u, v])) for (u, v) in edges))
        else:
            triples = chain((((u, v, {edge_attr: 1}) for d in range(A[u, v])) for (u, v) in edges))
    elif edge_attr in [False, None]:
        triples = ((u, v, {}) for (u, v) in edges)
    else:
        triples = ((u, v, {edge_attr: python_type(A[u, v])}) for (u, v) in edges)
    if G.is_multigraph() and (not G.is_directed()):
        triples = ((u, v, d) for (u, v, d) in triples if u <= v)
    G.add_edges_from(triples)
    return G