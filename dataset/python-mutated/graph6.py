"""Functions for reading and writing graphs in the *graph6* format.

The *graph6* file format is suitable for small graphs or large dense
graphs. For large sparse graphs, use the *sparse6* format.

For more information, see the `graph6`_ homepage.

.. _graph6: http://users.cecs.anu.edu.au/~bdm/data/formats.html

"""
from itertools import islice
import networkx as nx
from networkx.exception import NetworkXError
from networkx.utils import not_implemented_for, open_file
__all__ = ['from_graph6_bytes', 'read_graph6', 'to_graph6_bytes', 'write_graph6']

def _generate_graph6_bytes(G, nodes, header):
    if False:
        return 10
    "Yield bytes in the graph6 encoding of a graph.\n\n    `G` is an undirected simple graph. `nodes` is the list of nodes for\n    which the node-induced subgraph will be encoded; if `nodes` is the\n    list of all nodes in the graph, the entire graph will be\n    encoded. `header` is a Boolean that specifies whether to generate\n    the header ``b'>>graph6<<'`` before the remaining data.\n\n    This function generates `bytes` objects in the following order:\n\n    1. the header (if requested),\n    2. the encoding of the number of nodes,\n    3. each character, one-at-a-time, in the encoding of the requested\n       node-induced subgraph,\n    4. a newline character.\n\n    This function raises :exc:`ValueError` if the graph is too large for\n    the graph6 format (that is, greater than ``2 ** 36`` nodes).\n\n    "
    n = len(G)
    if n >= 2 ** 36:
        raise ValueError('graph6 is only defined if number of nodes is less than 2 ** 36')
    if header:
        yield b'>>graph6<<'
    for d in n_to_data(n):
        yield str.encode(chr(d + 63))
    bits = (nodes[j] in G[nodes[i]] for j in range(1, n) for i in range(j))
    chunk = list(islice(bits, 6))
    while chunk:
        d = sum((b << 5 - i for (i, b) in enumerate(chunk)))
        yield str.encode(chr(d + 63))
        chunk = list(islice(bits, 6))
    yield b'\n'

@nx._dispatch(graphs=None)
def from_graph6_bytes(bytes_in):
    if False:
        print('Hello World!')
    'Read a simple undirected graph in graph6 format from bytes.\n\n    Parameters\n    ----------\n    bytes_in : bytes\n       Data in graph6 format, without a trailing newline.\n\n    Returns\n    -------\n    G : Graph\n\n    Raises\n    ------\n    NetworkXError\n        If bytes_in is unable to be parsed in graph6 format\n\n    ValueError\n        If any character ``c`` in bytes_in does not satisfy\n        ``63 <= ord(c) < 127``.\n\n    Examples\n    --------\n    >>> G = nx.from_graph6_bytes(b"A_")\n    >>> sorted(G.edges())\n    [(0, 1)]\n\n    See Also\n    --------\n    read_graph6, write_graph6\n\n    References\n    ----------\n    .. [1] Graph6 specification\n           <http://users.cecs.anu.edu.au/~bdm/data/formats.html>\n\n    '

    def bits():
        if False:
            for i in range(10):
                print('nop')
        'Returns sequence of individual bits from 6-bit-per-value\n        list of data values.'
        for d in data:
            for i in [5, 4, 3, 2, 1, 0]:
                yield (d >> i & 1)
    if bytes_in.startswith(b'>>graph6<<'):
        bytes_in = bytes_in[10:]
    data = [c - 63 for c in bytes_in]
    if any((c > 63 for c in data)):
        raise ValueError('each input character must be in range(63, 127)')
    (n, data) = data_to_n(data)
    nd = (n * (n - 1) // 2 + 5) // 6
    if len(data) != nd:
        raise NetworkXError(f'Expected {n * (n - 1) // 2} bits but got {len(data) * 6} in graph6')
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for ((i, j), b) in zip(((i, j) for j in range(1, n) for i in range(j)), bits()):
        if b:
            G.add_edge(i, j)
    return G

@not_implemented_for('directed')
@not_implemented_for('multigraph')
def to_graph6_bytes(G, nodes=None, header=True):
    if False:
        i = 10
        return i + 15
    "Convert a simple undirected graph to bytes in graph6 format.\n\n    Parameters\n    ----------\n    G : Graph (undirected)\n\n    nodes: list or iterable\n       Nodes are labeled 0...n-1 in the order provided.  If None the ordering\n       given by ``G.nodes()`` is used.\n\n    header: bool\n       If True add '>>graph6<<' bytes to head of data.\n\n    Raises\n    ------\n    NetworkXNotImplemented\n        If the graph is directed or is a multigraph.\n\n    ValueError\n        If the graph has at least ``2 ** 36`` nodes; the graph6 format\n        is only defined for graphs of order less than ``2 ** 36``.\n\n    Examples\n    --------\n    >>> nx.to_graph6_bytes(nx.path_graph(2))\n    b'>>graph6<<A_\\n'\n\n    See Also\n    --------\n    from_graph6_bytes, read_graph6, write_graph6_bytes\n\n    Notes\n    -----\n    The returned bytes end with a newline character.\n\n    The format does not support edge or node labels, parallel edges or\n    self loops. If self loops are present they are silently ignored.\n\n    References\n    ----------\n    .. [1] Graph6 specification\n           <http://users.cecs.anu.edu.au/~bdm/data/formats.html>\n\n    "
    if nodes is not None:
        G = G.subgraph(nodes)
    H = nx.convert_node_labels_to_integers(G)
    nodes = sorted(H.nodes())
    return b''.join(_generate_graph6_bytes(H, nodes, header))

@open_file(0, mode='rb')
@nx._dispatch(graphs=None)
def read_graph6(path):
    if False:
        print('Hello World!')
    'Read simple undirected graphs in graph6 format from path.\n\n    Parameters\n    ----------\n    path : file or string\n       File or filename to write.\n\n    Returns\n    -------\n    G : Graph or list of Graphs\n       If the file contains multiple lines then a list of graphs is returned\n\n    Raises\n    ------\n    NetworkXError\n        If the string is unable to be parsed in graph6 format\n\n    Examples\n    --------\n    You can read a graph6 file by giving the path to the file::\n\n        >>> import tempfile\n        >>> with tempfile.NamedTemporaryFile(delete=False) as f:\n        ...     _ = f.write(b">>graph6<<A_\\n")\n        ...     _ = f.seek(0)\n        ...     G = nx.read_graph6(f.name)\n        >>> list(G.edges())\n        [(0, 1)]\n\n    You can also read a graph6 file by giving an open file-like object::\n\n        >>> import tempfile\n        >>> with tempfile.NamedTemporaryFile() as f:\n        ...     _ = f.write(b">>graph6<<A_\\n")\n        ...     _ = f.seek(0)\n        ...     G = nx.read_graph6(f)\n        >>> list(G.edges())\n        [(0, 1)]\n\n    See Also\n    --------\n    from_graph6_bytes, write_graph6\n\n    References\n    ----------\n    .. [1] Graph6 specification\n           <http://users.cecs.anu.edu.au/~bdm/data/formats.html>\n\n    '
    glist = []
    for line in path:
        line = line.strip()
        if not len(line):
            continue
        glist.append(from_graph6_bytes(line))
    if len(glist) == 1:
        return glist[0]
    else:
        return glist

@not_implemented_for('directed')
@not_implemented_for('multigraph')
@open_file(1, mode='wb')
def write_graph6(G, path, nodes=None, header=True):
    if False:
        return 10
    "Write a simple undirected graph to a path in graph6 format.\n\n    Parameters\n    ----------\n    G : Graph (undirected)\n\n    path : str\n       The path naming the file to which to write the graph.\n\n    nodes: list or iterable\n       Nodes are labeled 0...n-1 in the order provided.  If None the ordering\n       given by ``G.nodes()`` is used.\n\n    header: bool\n       If True add '>>graph6<<' string to head of data\n\n    Raises\n    ------\n    NetworkXNotImplemented\n        If the graph is directed or is a multigraph.\n\n    ValueError\n        If the graph has at least ``2 ** 36`` nodes; the graph6 format\n        is only defined for graphs of order less than ``2 ** 36``.\n\n    Examples\n    --------\n    You can write a graph6 file by giving the path to a file::\n\n        >>> import tempfile\n        >>> with tempfile.NamedTemporaryFile(delete=False) as f:\n        ...     nx.write_graph6(nx.path_graph(2), f.name)\n        ...     _ = f.seek(0)\n        ...     print(f.read())\n        b'>>graph6<<A_\\n'\n\n    See Also\n    --------\n    from_graph6_bytes, read_graph6\n\n    Notes\n    -----\n    The function writes a newline character after writing the encoding\n    of the graph.\n\n    The format does not support edge or node labels, parallel edges or\n    self loops.  If self loops are present they are silently ignored.\n\n    References\n    ----------\n    .. [1] Graph6 specification\n           <http://users.cecs.anu.edu.au/~bdm/data/formats.html>\n\n    "
    return write_graph6_file(G, path, nodes=nodes, header=header)

@not_implemented_for('directed')
@not_implemented_for('multigraph')
def write_graph6_file(G, f, nodes=None, header=True):
    if False:
        for i in range(10):
            print('nop')
    "Write a simple undirected graph to a file-like object in graph6 format.\n\n    Parameters\n    ----------\n    G : Graph (undirected)\n\n    f : file-like object\n       The file to write.\n\n    nodes: list or iterable\n       Nodes are labeled 0...n-1 in the order provided.  If None the ordering\n       given by ``G.nodes()`` is used.\n\n    header: bool\n       If True add '>>graph6<<' string to head of data\n\n    Raises\n    ------\n    NetworkXNotImplemented\n        If the graph is directed or is a multigraph.\n\n    ValueError\n        If the graph has at least ``2 ** 36`` nodes; the graph6 format\n        is only defined for graphs of order less than ``2 ** 36``.\n\n    Examples\n    --------\n    You can write a graph6 file by giving an open file-like object::\n\n        >>> import tempfile\n        >>> with tempfile.NamedTemporaryFile() as f:\n        ...     nx.write_graph6(nx.path_graph(2), f)\n        ...     _ = f.seek(0)\n        ...     print(f.read())\n        b'>>graph6<<A_\\n'\n\n    See Also\n    --------\n    from_graph6_bytes, read_graph6\n\n    Notes\n    -----\n    The function writes a newline character after writing the encoding\n    of the graph.\n\n    The format does not support edge or node labels, parallel edges or\n    self loops.  If self loops are present they are silently ignored.\n\n    References\n    ----------\n    .. [1] Graph6 specification\n           <http://users.cecs.anu.edu.au/~bdm/data/formats.html>\n\n    "
    if nodes is not None:
        G = G.subgraph(nodes)
    H = nx.convert_node_labels_to_integers(G)
    nodes = sorted(H.nodes())
    for b in _generate_graph6_bytes(H, nodes, header):
        f.write(b)

def data_to_n(data):
    if False:
        return 10
    'Read initial one-, four- or eight-unit value from graph6\n    integer sequence.\n\n    Return (value, rest of seq.)'
    if data[0] <= 62:
        return (data[0], data[1:])
    if data[1] <= 62:
        return ((data[1] << 12) + (data[2] << 6) + data[3], data[4:])
    return ((data[2] << 30) + (data[3] << 24) + (data[4] << 18) + (data[5] << 12) + (data[6] << 6) + data[7], data[8:])

def n_to_data(n):
    if False:
        return 10
    'Convert an integer to one-, four- or eight-unit graph6 sequence.\n\n    This function is undefined if `n` is not in ``range(2 ** 36)``.\n\n    '
    if n <= 62:
        return [n]
    elif n <= 258047:
        return [63, n >> 12 & 63, n >> 6 & 63, n & 63]
    else:
        return [63, 63, n >> 30 & 63, n >> 24 & 63, n >> 18 & 63, n >> 12 & 63, n >> 6 & 63, n & 63]