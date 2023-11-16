"""
**************
Adjacency List
**************
Read and write NetworkX graphs as adjacency lists.

Adjacency list format is useful for graphs without data associated
with nodes or edges and for nodes that can be meaningfully represented
as strings.

Format
------
The adjacency list format consists of lines with node labels.  The
first label in a line is the source node.  Further labels in the line
are considered target nodes and are added to the graph along with an edge
between the source node and target node.

The graph with edges a-b, a-c, d-e can be represented as the following
adjacency list (anything following the # in a line is a comment)::

     a b c # source target target
     d e
"""
__all__ = ['generate_adjlist', 'write_adjlist', 'parse_adjlist', 'read_adjlist']
import networkx as nx
from networkx.utils import open_file

def generate_adjlist(G, delimiter=' '):
    if False:
        while True:
            i = 10
    'Generate a single line of the graph G in adjacency list format.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n\n    delimiter : string, optional\n       Separator for node labels\n\n    Returns\n    -------\n    lines : string\n        Lines of data in adjlist format.\n\n    Examples\n    --------\n    >>> G = nx.lollipop_graph(4, 3)\n    >>> for line in nx.generate_adjlist(G):\n    ...     print(line)\n    0 1 2 3\n    1 2 3\n    2 3\n    3 4\n    4 5\n    5 6\n    6\n\n    See Also\n    --------\n    write_adjlist, read_adjlist\n\n    Notes\n    -----\n    The default `delimiter=" "` will result in unexpected results if node names contain\n    whitespace characters. To avoid this problem, specify an alternate delimiter when spaces are\n    valid in node names.\n\n    NB: This option is not available for data that isn\'t user-generated.\n\n    '
    directed = G.is_directed()
    seen = set()
    for (s, nbrs) in G.adjacency():
        line = str(s) + delimiter
        for (t, data) in nbrs.items():
            if not directed and t in seen:
                continue
            if G.is_multigraph():
                for d in data.values():
                    line += str(t) + delimiter
            else:
                line += str(t) + delimiter
        if not directed:
            seen.add(s)
        yield line[:-len(delimiter)]

@open_file(1, mode='wb')
def write_adjlist(G, path, comments='#', delimiter=' ', encoding='utf-8'):
    if False:
        print('Hello World!')
    'Write graph G in single-line adjacency-list format to path.\n\n\n    Parameters\n    ----------\n    G : NetworkX graph\n\n    path : string or file\n       Filename or file handle for data output.\n       Filenames ending in .gz or .bz2 will be compressed.\n\n    comments : string, optional\n       Marker for comment lines\n\n    delimiter : string, optional\n       Separator for node labels\n\n    encoding : string, optional\n       Text encoding.\n\n    Examples\n    --------\n    >>> G = nx.path_graph(4)\n    >>> nx.write_adjlist(G, "test.adjlist")\n\n    The path can be a filehandle or a string with the name of the file. If a\n    filehandle is provided, it has to be opened in \'wb\' mode.\n\n    >>> fh = open("test.adjlist", "wb")\n    >>> nx.write_adjlist(G, fh)\n\n    Notes\n    -----\n    The default `delimiter=" "` will result in unexpected results if node names contain\n    whitespace characters. To avoid this problem, specify an alternate delimiter when spaces are\n    valid in node names.\n    NB: This option is not available for data that isn\'t user-generated.\n\n    This format does not store graph, node, or edge data.\n\n    See Also\n    --------\n    read_adjlist, generate_adjlist\n    '
    import sys
    import time
    pargs = comments + ' '.join(sys.argv) + '\n'
    header = pargs + comments + f' GMT {time.asctime(time.gmtime())}\n' + comments + f' {G.name}\n'
    path.write(header.encode(encoding))
    for line in generate_adjlist(G, delimiter):
        line += '\n'
        path.write(line.encode(encoding))

@nx._dispatch(graphs=None)
def parse_adjlist(lines, comments='#', delimiter=None, create_using=None, nodetype=None):
    if False:
        for i in range(10):
            print('nop')
    'Parse lines of a graph adjacency list representation.\n\n    Parameters\n    ----------\n    lines : list or iterator of strings\n        Input data in adjlist format\n\n    create_using : NetworkX graph constructor, optional (default=nx.Graph)\n       Graph type to create. If graph instance, then cleared before populated.\n\n    nodetype : Python type, optional\n       Convert nodes to this type.\n\n    comments : string, optional\n       Marker for comment lines\n\n    delimiter : string, optional\n       Separator for node labels.  The default is whitespace.\n\n    Returns\n    -------\n    G: NetworkX graph\n        The graph corresponding to the lines in adjacency list format.\n\n    Examples\n    --------\n    >>> lines = ["1 2 5", "2 3 4", "3 5", "4", "5"]\n    >>> G = nx.parse_adjlist(lines, nodetype=int)\n    >>> nodes = [1, 2, 3, 4, 5]\n    >>> all(node in G for node in nodes)\n    True\n    >>> edges = [(1, 2), (1, 5), (2, 3), (2, 4), (3, 5)]\n    >>> all((u, v) in G.edges() or (v, u) in G.edges() for (u, v) in edges)\n    True\n\n    See Also\n    --------\n    read_adjlist\n\n    '
    G = nx.empty_graph(0, create_using)
    for line in lines:
        p = line.find(comments)
        if p >= 0:
            line = line[:p]
        if not len(line):
            continue
        vlist = line.strip().split(delimiter)
        u = vlist.pop(0)
        if nodetype is not None:
            try:
                u = nodetype(u)
            except BaseException as err:
                raise TypeError(f'Failed to convert node ({u}) to type {nodetype}') from err
        G.add_node(u)
        if nodetype is not None:
            try:
                vlist = list(map(nodetype, vlist))
            except BaseException as err:
                raise TypeError(f"Failed to convert nodes ({','.join(vlist)}) to type {nodetype}") from err
        G.add_edges_from([(u, v) for v in vlist])
    return G

@open_file(0, mode='rb')
@nx._dispatch(graphs=None)
def read_adjlist(path, comments='#', delimiter=None, create_using=None, nodetype=None, encoding='utf-8'):
    if False:
        return 10
    'Read graph in adjacency list format from path.\n\n    Parameters\n    ----------\n    path : string or file\n       Filename or file handle to read.\n       Filenames ending in .gz or .bz2 will be uncompressed.\n\n    create_using : NetworkX graph constructor, optional (default=nx.Graph)\n       Graph type to create. If graph instance, then cleared before populated.\n\n    nodetype : Python type, optional\n       Convert nodes to this type.\n\n    comments : string, optional\n       Marker for comment lines\n\n    delimiter : string, optional\n       Separator for node labels.  The default is whitespace.\n\n    Returns\n    -------\n    G: NetworkX graph\n        The graph corresponding to the lines in adjacency list format.\n\n    Examples\n    --------\n    >>> G = nx.path_graph(4)\n    >>> nx.write_adjlist(G, "test.adjlist")\n    >>> G = nx.read_adjlist("test.adjlist")\n\n    The path can be a filehandle or a string with the name of the file. If a\n    filehandle is provided, it has to be opened in \'rb\' mode.\n\n    >>> fh = open("test.adjlist", "rb")\n    >>> G = nx.read_adjlist(fh)\n\n    Filenames ending in .gz or .bz2 will be compressed.\n\n    >>> nx.write_adjlist(G, "test.adjlist.gz")\n    >>> G = nx.read_adjlist("test.adjlist.gz")\n\n    The optional nodetype is a function to convert node strings to nodetype.\n\n    For example\n\n    >>> G = nx.read_adjlist("test.adjlist", nodetype=int)\n\n    will attempt to convert all nodes to integer type.\n\n    Since nodes must be hashable, the function nodetype must return hashable\n    types (e.g. int, float, str, frozenset - or tuples of those, etc.)\n\n    The optional create_using parameter indicates the type of NetworkX graph\n    created.  The default is `nx.Graph`, an undirected graph.\n    To read the data as a directed graph use\n\n    >>> G = nx.read_adjlist("test.adjlist", create_using=nx.DiGraph)\n\n    Notes\n    -----\n    This format does not store graph or node data.\n\n    See Also\n    --------\n    write_adjlist\n    '
    lines = (line.decode(encoding) for line in path)
    return parse_adjlist(lines, comments=comments, delimiter=delimiter, create_using=create_using, nodetype=nodetype)