"""
********************
Bipartite Edge Lists
********************
Read and write NetworkX graphs as bipartite edge lists.

Format
------
You can read or write three formats of edge lists with these functions.

Node pairs with no data::

 1 2

Python dictionary as data::

 1 2 {'weight':7, 'color':'green'}

Arbitrary data::

 1 2 7 green

For each edge (u, v) the node u is assigned to part 0 and the node v to part 1.
"""
__all__ = ['generate_edgelist', 'write_edgelist', 'parse_edgelist', 'read_edgelist']
import networkx as nx
from networkx.utils import not_implemented_for, open_file

@open_file(1, mode='wb')
def write_edgelist(G, path, comments='#', delimiter=' ', data=True, encoding='utf-8'):
    if False:
        for i in range(10):
            print('nop')
    'Write a bipartite graph as a list of edges.\n\n    Parameters\n    ----------\n    G : Graph\n       A NetworkX bipartite graph\n    path : file or string\n       File or filename to write. If a file is provided, it must be\n       opened in \'wb\' mode. Filenames ending in .gz or .bz2 will be compressed.\n    comments : string, optional\n       The character used to indicate the start of a comment\n    delimiter : string, optional\n       The string used to separate values.  The default is whitespace.\n    data : bool or list, optional\n       If False write no edge data.\n       If True write a string representation of the edge data dictionary..\n       If a list (or other iterable) is provided, write the  keys specified\n       in the list.\n    encoding: string, optional\n       Specify which encoding to use when writing file.\n\n    Examples\n    --------\n    >>> G = nx.path_graph(4)\n    >>> G.add_nodes_from([0, 2], bipartite=0)\n    >>> G.add_nodes_from([1, 3], bipartite=1)\n    >>> nx.write_edgelist(G, "test.edgelist")\n    >>> fh = open("test.edgelist", "wb")\n    >>> nx.write_edgelist(G, fh)\n    >>> nx.write_edgelist(G, "test.edgelist.gz")\n    >>> nx.write_edgelist(G, "test.edgelist.gz", data=False)\n\n    >>> G = nx.Graph()\n    >>> G.add_edge(1, 2, weight=7, color="red")\n    >>> nx.write_edgelist(G, "test.edgelist", data=False)\n    >>> nx.write_edgelist(G, "test.edgelist", data=["color"])\n    >>> nx.write_edgelist(G, "test.edgelist", data=["color", "weight"])\n\n    See Also\n    --------\n    write_edgelist\n    generate_edgelist\n    '
    for line in generate_edgelist(G, delimiter, data):
        line += '\n'
        path.write(line.encode(encoding))

@not_implemented_for('directed')
def generate_edgelist(G, delimiter=' ', data=True):
    if False:
        return 10
    'Generate a single line of the bipartite graph G in edge list format.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n       The graph is assumed to have node attribute `part` set to 0,1 representing\n       the two graph parts\n\n    delimiter : string, optional\n       Separator for node labels\n\n    data : bool or list of keys\n       If False generate no edge data.  If True use a dictionary\n       representation of edge data.  If a list of keys use a list of data\n       values corresponding to the keys.\n\n    Returns\n    -------\n    lines : string\n        Lines of data in adjlist format.\n\n    Examples\n    --------\n    >>> from networkx.algorithms import bipartite\n    >>> G = nx.path_graph(4)\n    >>> G.add_nodes_from([0, 2], bipartite=0)\n    >>> G.add_nodes_from([1, 3], bipartite=1)\n    >>> G[1][2]["weight"] = 3\n    >>> G[2][3]["capacity"] = 12\n    >>> for line in bipartite.generate_edgelist(G, data=False):\n    ...     print(line)\n    0 1\n    2 1\n    2 3\n\n    >>> for line in bipartite.generate_edgelist(G):\n    ...     print(line)\n    0 1 {}\n    2 1 {\'weight\': 3}\n    2 3 {\'capacity\': 12}\n\n    >>> for line in bipartite.generate_edgelist(G, data=["weight"]):\n    ...     print(line)\n    0 1\n    2 1 3\n    2 3\n    '
    try:
        part0 = [n for (n, d) in G.nodes.items() if d['bipartite'] == 0]
    except BaseException as err:
        raise AttributeError('Missing node attribute `bipartite`') from err
    if data is True or data is False:
        for n in part0:
            for edge in G.edges(n, data=data):
                yield delimiter.join(map(str, edge))
    else:
        for n in part0:
            for (u, v, d) in G.edges(n, data=True):
                edge = [u, v]
                try:
                    edge.extend((d[k] for k in data))
                except KeyError:
                    pass
                yield delimiter.join(map(str, edge))

@nx._dispatch(name='bipartite_parse_edgelist', graphs=None)
def parse_edgelist(lines, comments='#', delimiter=None, create_using=None, nodetype=None, data=True):
    if False:
        for i in range(10):
            print('nop')
    'Parse lines of an edge list representation of a bipartite graph.\n\n    Parameters\n    ----------\n    lines : list or iterator of strings\n        Input data in edgelist format\n    comments : string, optional\n       Marker for comment lines\n    delimiter : string, optional\n       Separator for node labels\n    create_using: NetworkX graph container, optional\n       Use given NetworkX graph for holding nodes or edges.\n    nodetype : Python type, optional\n       Convert nodes to this type.\n    data : bool or list of (label,type) tuples\n       If False generate no edge data or if True use a dictionary\n       representation of edge data or a list tuples specifying dictionary\n       key names and types for edge data.\n\n    Returns\n    -------\n    G: NetworkX Graph\n        The bipartite graph corresponding to lines\n\n    Examples\n    --------\n    Edgelist with no data:\n\n    >>> from networkx.algorithms import bipartite\n    >>> lines = ["1 2", "2 3", "3 4"]\n    >>> G = bipartite.parse_edgelist(lines, nodetype=int)\n    >>> sorted(G.nodes())\n    [1, 2, 3, 4]\n    >>> sorted(G.nodes(data=True))\n    [(1, {\'bipartite\': 0}), (2, {\'bipartite\': 0}), (3, {\'bipartite\': 0}), (4, {\'bipartite\': 1})]\n    >>> sorted(G.edges())\n    [(1, 2), (2, 3), (3, 4)]\n\n    Edgelist with data in Python dictionary representation:\n\n    >>> lines = ["1 2 {\'weight\':3}", "2 3 {\'weight\':27}", "3 4 {\'weight\':3.0}"]\n    >>> G = bipartite.parse_edgelist(lines, nodetype=int)\n    >>> sorted(G.nodes())\n    [1, 2, 3, 4]\n    >>> sorted(G.edges(data=True))\n    [(1, 2, {\'weight\': 3}), (2, 3, {\'weight\': 27}), (3, 4, {\'weight\': 3.0})]\n\n    Edgelist with data in a list:\n\n    >>> lines = ["1 2 3", "2 3 27", "3 4 3.0"]\n    >>> G = bipartite.parse_edgelist(lines, nodetype=int, data=(("weight", float),))\n    >>> sorted(G.nodes())\n    [1, 2, 3, 4]\n    >>> sorted(G.edges(data=True))\n    [(1, 2, {\'weight\': 3.0}), (2, 3, {\'weight\': 27.0}), (3, 4, {\'weight\': 3.0})]\n\n    See Also\n    --------\n    '
    from ast import literal_eval
    G = nx.empty_graph(0, create_using)
    for line in lines:
        p = line.find(comments)
        if p >= 0:
            line = line[:p]
        if not len(line):
            continue
        s = line.strip().split(delimiter)
        if len(s) < 2:
            continue
        u = s.pop(0)
        v = s.pop(0)
        d = s
        if nodetype is not None:
            try:
                u = nodetype(u)
                v = nodetype(v)
            except BaseException as err:
                raise TypeError(f'Failed to convert nodes {u},{v} to type {nodetype}.') from err
        if len(d) == 0 or data is False:
            edgedata = {}
        elif data is True:
            try:
                edgedata = dict(literal_eval(' '.join(d)))
            except BaseException as err:
                raise TypeError(f'Failed to convert edge data ({d}) to dictionary.') from err
        else:
            if len(d) != len(data):
                raise IndexError(f'Edge data {d} and data_keys {data} are not the same length')
            edgedata = {}
            for ((edge_key, edge_type), edge_value) in zip(data, d):
                try:
                    edge_value = edge_type(edge_value)
                except BaseException as err:
                    raise TypeError(f'Failed to convert {edge_key} data {edge_value} to type {edge_type}.') from err
                edgedata.update({edge_key: edge_value})
        G.add_node(u, bipartite=0)
        G.add_node(v, bipartite=1)
        G.add_edge(u, v, **edgedata)
    return G

@open_file(0, mode='rb')
@nx._dispatch(name='bipartite_read_edgelist', graphs=None)
def read_edgelist(path, comments='#', delimiter=None, create_using=None, nodetype=None, data=True, edgetype=None, encoding='utf-8'):
    if False:
        i = 10
        return i + 15
    'Read a bipartite graph from a list of edges.\n\n    Parameters\n    ----------\n    path : file or string\n       File or filename to read. If a file is provided, it must be\n       opened in \'rb\' mode.\n       Filenames ending in .gz or .bz2 will be uncompressed.\n    comments : string, optional\n       The character used to indicate the start of a comment.\n    delimiter : string, optional\n       The string used to separate values.  The default is whitespace.\n    create_using : Graph container, optional,\n       Use specified container to build graph.  The default is networkx.Graph,\n       an undirected graph.\n    nodetype : int, float, str, Python type, optional\n       Convert node data from strings to specified type\n    data : bool or list of (label,type) tuples\n       Tuples specifying dictionary key names and types for edge data\n    edgetype : int, float, str, Python type, optional OBSOLETE\n       Convert edge data from strings to specified type and use as \'weight\'\n    encoding: string, optional\n       Specify which encoding to use when reading file.\n\n    Returns\n    -------\n    G : graph\n       A networkx Graph or other type specified with create_using\n\n    Examples\n    --------\n    >>> from networkx.algorithms import bipartite\n    >>> G = nx.path_graph(4)\n    >>> G.add_nodes_from([0, 2], bipartite=0)\n    >>> G.add_nodes_from([1, 3], bipartite=1)\n    >>> bipartite.write_edgelist(G, "test.edgelist")\n    >>> G = bipartite.read_edgelist("test.edgelist")\n\n    >>> fh = open("test.edgelist", "rb")\n    >>> G = bipartite.read_edgelist(fh)\n    >>> fh.close()\n\n    >>> G = bipartite.read_edgelist("test.edgelist", nodetype=int)\n\n    Edgelist with data in a list:\n\n    >>> textline = "1 2 3"\n    >>> fh = open("test.edgelist", "w")\n    >>> d = fh.write(textline)\n    >>> fh.close()\n    >>> G = bipartite.read_edgelist(\n    ...     "test.edgelist", nodetype=int, data=(("weight", float),)\n    ... )\n    >>> list(G)\n    [1, 2]\n    >>> list(G.edges(data=True))\n    [(1, 2, {\'weight\': 3.0})]\n\n    See parse_edgelist() for more examples of formatting.\n\n    See Also\n    --------\n    parse_edgelist\n\n    Notes\n    -----\n    Since nodes must be hashable, the function nodetype must return hashable\n    types (e.g. int, float, str, frozenset - or tuples of those, etc.)\n    '
    lines = (line.decode(encoding) for line in path)
    return parse_edgelist(lines, comments=comments, delimiter=delimiter, create_using=create_using, nodetype=nodetype, data=data)