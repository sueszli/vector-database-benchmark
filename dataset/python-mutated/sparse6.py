"""Functions for reading and writing graphs in the *sparse6* format.

The *sparse6* file format is a space-efficient format for large sparse
graphs. For small graphs or large dense graphs, use the *graph6* file
format.

For more information, see the `sparse6`_ homepage.

.. _sparse6: https://users.cecs.anu.edu.au/~bdm/data/formats.html

"""
import networkx as nx
from networkx.exception import NetworkXError
from networkx.readwrite.graph6 import data_to_n, n_to_data
from networkx.utils import not_implemented_for, open_file
__all__ = ['from_sparse6_bytes', 'read_sparse6', 'to_sparse6_bytes', 'write_sparse6']

def _generate_sparse6_bytes(G, nodes, header):
    if False:
        return 10
    "Yield bytes in the sparse6 encoding of a graph.\n\n    `G` is an undirected simple graph. `nodes` is the list of nodes for\n    which the node-induced subgraph will be encoded; if `nodes` is the\n    list of all nodes in the graph, the entire graph will be\n    encoded. `header` is a Boolean that specifies whether to generate\n    the header ``b'>>sparse6<<'`` before the remaining data.\n\n    This function generates `bytes` objects in the following order:\n\n    1. the header (if requested),\n    2. the encoding of the number of nodes,\n    3. each character, one-at-a-time, in the encoding of the requested\n       node-induced subgraph,\n    4. a newline character.\n\n    This function raises :exc:`ValueError` if the graph is too large for\n    the graph6 format (that is, greater than ``2 ** 36`` nodes).\n\n    "
    n = len(G)
    if n >= 2 ** 36:
        raise ValueError('sparse6 is only defined if number of nodes is less than 2 ** 36')
    if header:
        yield b'>>sparse6<<'
    yield b':'
    for d in n_to_data(n):
        yield str.encode(chr(d + 63))
    k = 1
    while 1 << k < n:
        k += 1

    def enc(x):
        if False:
            for i in range(10):
                print('nop')
        'Big endian k-bit encoding of x'
        return [1 if x & 1 << k - 1 - i else 0 for i in range(k)]
    edges = sorted(((max(u, v), min(u, v)) for (u, v) in G.edges()))
    bits = []
    curv = 0
    for (v, u) in edges:
        if v == curv:
            bits.append(0)
            bits.extend(enc(u))
        elif v == curv + 1:
            curv += 1
            bits.append(1)
            bits.extend(enc(u))
        else:
            curv = v
            bits.append(1)
            bits.extend(enc(v))
            bits.append(0)
            bits.extend(enc(u))
    if k < 6 and n == 1 << k and (-len(bits) % 6 >= k) and (curv < n - 1):
        bits.append(0)
        bits.extend([1] * (-len(bits) % 6))
    else:
        bits.extend([1] * (-len(bits) % 6))
    data = [(bits[i + 0] << 5) + (bits[i + 1] << 4) + (bits[i + 2] << 3) + (bits[i + 3] << 2) + (bits[i + 4] << 1) + (bits[i + 5] << 0) for i in range(0, len(bits), 6)]
    for d in data:
        yield str.encode(chr(d + 63))
    yield b'\n'

@nx._dispatch(graphs=None)
def from_sparse6_bytes(string):
    if False:
        while True:
            i = 10
    'Read an undirected graph in sparse6 format from string.\n\n    Parameters\n    ----------\n    string : string\n       Data in sparse6 format\n\n    Returns\n    -------\n    G : Graph\n\n    Raises\n    ------\n    NetworkXError\n        If the string is unable to be parsed in sparse6 format\n\n    Examples\n    --------\n    >>> G = nx.from_sparse6_bytes(b":A_")\n    >>> sorted(G.edges())\n    [(0, 1), (0, 1), (0, 1)]\n\n    See Also\n    --------\n    read_sparse6, write_sparse6\n\n    References\n    ----------\n    .. [1] Sparse6 specification\n           <https://users.cecs.anu.edu.au/~bdm/data/formats.html>\n\n    '
    if string.startswith(b'>>sparse6<<'):
        string = string[11:]
    if not string.startswith(b':'):
        raise NetworkXError('Expected leading colon in sparse6')
    chars = [c - 63 for c in string[1:]]
    (n, data) = data_to_n(chars)
    k = 1
    while 1 << k < n:
        k += 1

    def parseData():
        if False:
            for i in range(10):
                print('nop')
        'Returns stream of pairs b[i], x[i] for sparse6 format.'
        chunks = iter(data)
        d = None
        dLen = 0
        while 1:
            if dLen < 1:
                try:
                    d = next(chunks)
                except StopIteration:
                    return
                dLen = 6
            dLen -= 1
            b = d >> dLen & 1
            x = d & (1 << dLen) - 1
            xLen = dLen
            while xLen < k:
                try:
                    d = next(chunks)
                except StopIteration:
                    return
                dLen = 6
                x = (x << 6) + d
                xLen += 6
            x = x >> xLen - k
            dLen = xLen - k
            yield (b, x)
    v = 0
    G = nx.MultiGraph()
    G.add_nodes_from(range(n))
    multigraph = False
    for (b, x) in parseData():
        if b == 1:
            v += 1
        if x >= n or v >= n:
            break
        elif x > v:
            v = x
        else:
            if G.has_edge(x, v):
                multigraph = True
            G.add_edge(x, v)
    if not multigraph:
        G = nx.Graph(G)
    return G

def to_sparse6_bytes(G, nodes=None, header=True):
    if False:
        while True:
            i = 10
    "Convert an undirected graph to bytes in sparse6 format.\n\n    Parameters\n    ----------\n    G : Graph (undirected)\n\n    nodes: list or iterable\n       Nodes are labeled 0...n-1 in the order provided.  If None the ordering\n       given by ``G.nodes()`` is used.\n\n    header: bool\n       If True add '>>sparse6<<' bytes to head of data.\n\n    Raises\n    ------\n    NetworkXNotImplemented\n        If the graph is directed.\n\n    ValueError\n        If the graph has at least ``2 ** 36`` nodes; the sparse6 format\n        is only defined for graphs of order less than ``2 ** 36``.\n\n    Examples\n    --------\n    >>> nx.to_sparse6_bytes(nx.path_graph(2))\n    b'>>sparse6<<:An\\n'\n\n    See Also\n    --------\n    to_sparse6_bytes, read_sparse6, write_sparse6_bytes\n\n    Notes\n    -----\n    The returned bytes end with a newline character.\n\n    The format does not support edge or node labels.\n\n    References\n    ----------\n    .. [1] Graph6 specification\n           <https://users.cecs.anu.edu.au/~bdm/data/formats.html>\n\n    "
    if nodes is not None:
        G = G.subgraph(nodes)
    G = nx.convert_node_labels_to_integers(G, ordering='sorted')
    return b''.join(_generate_sparse6_bytes(G, nodes, header))

@open_file(0, mode='rb')
@nx._dispatch(graphs=None)
def read_sparse6(path):
    if False:
        return 10
    'Read an undirected graph in sparse6 format from path.\n\n    Parameters\n    ----------\n    path : file or string\n       File or filename to write.\n\n    Returns\n    -------\n    G : Graph/Multigraph or list of Graphs/MultiGraphs\n       If the file contains multiple lines then a list of graphs is returned\n\n    Raises\n    ------\n    NetworkXError\n        If the string is unable to be parsed in sparse6 format\n\n    Examples\n    --------\n    You can read a sparse6 file by giving the path to the file::\n\n        >>> import tempfile\n        >>> with tempfile.NamedTemporaryFile(delete=False) as f:\n        ...     _ = f.write(b">>sparse6<<:An\\n")\n        ...     _ = f.seek(0)\n        ...     G = nx.read_sparse6(f.name)\n        >>> list(G.edges())\n        [(0, 1)]\n\n    You can also read a sparse6 file by giving an open file-like object::\n\n        >>> import tempfile\n        >>> with tempfile.NamedTemporaryFile() as f:\n        ...     _ = f.write(b">>sparse6<<:An\\n")\n        ...     _ = f.seek(0)\n        ...     G = nx.read_sparse6(f)\n        >>> list(G.edges())\n        [(0, 1)]\n\n    See Also\n    --------\n    read_sparse6, from_sparse6_bytes\n\n    References\n    ----------\n    .. [1] Sparse6 specification\n           <https://users.cecs.anu.edu.au/~bdm/data/formats.html>\n\n    '
    glist = []
    for line in path:
        line = line.strip()
        if not len(line):
            continue
        glist.append(from_sparse6_bytes(line))
    if len(glist) == 1:
        return glist[0]
    else:
        return glist

@not_implemented_for('directed')
@open_file(1, mode='wb')
def write_sparse6(G, path, nodes=None, header=True):
    if False:
        for i in range(10):
            print('nop')
    "Write graph G to given path in sparse6 format.\n\n    Parameters\n    ----------\n    G : Graph (undirected)\n\n    path : file or string\n       File or filename to write\n\n    nodes: list or iterable\n       Nodes are labeled 0...n-1 in the order provided.  If None the ordering\n       given by G.nodes() is used.\n\n    header: bool\n       If True add '>>sparse6<<' string to head of data\n\n    Raises\n    ------\n    NetworkXError\n        If the graph is directed\n\n    Examples\n    --------\n    You can write a sparse6 file by giving the path to the file::\n\n        >>> import tempfile\n        >>> with tempfile.NamedTemporaryFile(delete=False) as f:\n        ...     nx.write_sparse6(nx.path_graph(2), f.name)\n        ...     print(f.read())\n        b'>>sparse6<<:An\\n'\n\n    You can also write a sparse6 file by giving an open file-like object::\n\n        >>> with tempfile.NamedTemporaryFile() as f:\n        ...     nx.write_sparse6(nx.path_graph(2), f)\n        ...     _ = f.seek(0)\n        ...     print(f.read())\n        b'>>sparse6<<:An\\n'\n\n    See Also\n    --------\n    read_sparse6, from_sparse6_bytes\n\n    Notes\n    -----\n    The format does not support edge or node labels.\n\n    References\n    ----------\n    .. [1] Sparse6 specification\n           <https://users.cecs.anu.edu.au/~bdm/data/formats.html>\n\n    "
    if nodes is not None:
        G = G.subgraph(nodes)
    G = nx.convert_node_labels_to_integers(G, ordering='sorted')
    for b in _generate_sparse6_bytes(G, nodes, header):
        path.write(b)