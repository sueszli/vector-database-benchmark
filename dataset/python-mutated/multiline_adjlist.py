"""
*************************
Multi-line Adjacency List
*************************
Read and write NetworkX graphs as multi-line adjacency lists.

The multi-line adjacency list format is useful for graphs with
nodes that can be meaningfully represented as strings.  With this format
simple edge data can be stored but node or graph data is not.

Format
------
The first label in a line is the source node label followed by the node degree
d.  The next d lines are target node labels and optional edge data.
That pattern repeats for all nodes in the graph.

The graph with edges a-b, a-c, d-e can be represented as the following
adjacency list (anything following the # in a line is a comment)::

     # example.multiline-adjlist
     a 2
     b
     c
     d 1
     e
"""
__all__ = ['generate_multiline_adjlist', 'write_multiline_adjlist', 'parse_multiline_adjlist', 'read_multiline_adjlist']
import networkx as nx
from networkx.utils import open_file

def generate_multiline_adjlist(G, delimiter=' '):
    if False:
        while True:
            i = 10
    'Generate a single line of the graph G in multiline adjacency list format.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n\n    delimiter : string, optional\n       Separator for node labels\n\n    Returns\n    -------\n    lines : string\n        Lines of data in multiline adjlist format.\n\n    Examples\n    --------\n    >>> G = nx.lollipop_graph(4, 3)\n    >>> for line in nx.generate_multiline_adjlist(G):\n    ...     print(line)\n    0 3\n    1 {}\n    2 {}\n    3 {}\n    1 2\n    2 {}\n    3 {}\n    2 1\n    3 {}\n    3 1\n    4 {}\n    4 1\n    5 {}\n    5 1\n    6 {}\n    6 0\n\n    See Also\n    --------\n    write_multiline_adjlist, read_multiline_adjlist\n    '
    if G.is_directed():
        if G.is_multigraph():
            for (s, nbrs) in G.adjacency():
                nbr_edges = [(u, data) for (u, datadict) in nbrs.items() for (key, data) in datadict.items()]
                deg = len(nbr_edges)
                yield (str(s) + delimiter + str(deg))
                for (u, d) in nbr_edges:
                    if d is None:
                        yield str(u)
                    else:
                        yield (str(u) + delimiter + str(d))
        else:
            for (s, nbrs) in G.adjacency():
                deg = len(nbrs)
                yield (str(s) + delimiter + str(deg))
                for (u, d) in nbrs.items():
                    if d is None:
                        yield str(u)
                    else:
                        yield (str(u) + delimiter + str(d))
    elif G.is_multigraph():
        seen = set()
        for (s, nbrs) in G.adjacency():
            nbr_edges = [(u, data) for (u, datadict) in nbrs.items() if u not in seen for (key, data) in datadict.items()]
            deg = len(nbr_edges)
            yield (str(s) + delimiter + str(deg))
            for (u, d) in nbr_edges:
                if d is None:
                    yield str(u)
                else:
                    yield (str(u) + delimiter + str(d))
            seen.add(s)
    else:
        seen = set()
        for (s, nbrs) in G.adjacency():
            nbr_edges = [(u, d) for (u, d) in nbrs.items() if u not in seen]
            deg = len(nbr_edges)
            yield (str(s) + delimiter + str(deg))
            for (u, d) in nbr_edges:
                if d is None:
                    yield str(u)
                else:
                    yield (str(u) + delimiter + str(d))
            seen.add(s)

@open_file(1, mode='wb')
def write_multiline_adjlist(G, path, delimiter=' ', comments='#', encoding='utf-8'):
    if False:
        i = 10
        return i + 15
    'Write the graph G in multiline adjacency list format to path\n\n    Parameters\n    ----------\n    G : NetworkX graph\n\n    path : string or file\n       Filename or file handle to write to.\n       Filenames ending in .gz or .bz2 will be compressed.\n\n    comments : string, optional\n       Marker for comment lines\n\n    delimiter : string, optional\n       Separator for node labels\n\n    encoding : string, optional\n       Text encoding.\n\n    Examples\n    --------\n    >>> G = nx.path_graph(4)\n    >>> nx.write_multiline_adjlist(G, "test.adjlist")\n\n    The path can be a file handle or a string with the name of the file. If a\n    file handle is provided, it has to be opened in \'wb\' mode.\n\n    >>> fh = open("test.adjlist", "wb")\n    >>> nx.write_multiline_adjlist(G, fh)\n\n    Filenames ending in .gz or .bz2 will be compressed.\n\n    >>> nx.write_multiline_adjlist(G, "test.adjlist.gz")\n\n    See Also\n    --------\n    read_multiline_adjlist\n    '
    import sys
    import time
    pargs = comments + ' '.join(sys.argv)
    header = f'{pargs}\n' + comments + f' GMT {time.asctime(time.gmtime())}\n' + comments + f' {G.name}\n'
    path.write(header.encode(encoding))
    for multiline in generate_multiline_adjlist(G, delimiter):
        multiline += '\n'
        path.write(multiline.encode(encoding))

@nx._dispatch(graphs=None)
def parse_multiline_adjlist(lines, comments='#', delimiter=None, create_using=None, nodetype=None, edgetype=None):
    if False:
        for i in range(10):
            print('nop')
    'Parse lines of a multiline adjacency list representation of a graph.\n\n    Parameters\n    ----------\n    lines : list or iterator of strings\n        Input data in multiline adjlist format\n\n    create_using : NetworkX graph constructor, optional (default=nx.Graph)\n       Graph type to create. If graph instance, then cleared before populated.\n\n    nodetype : Python type, optional\n       Convert nodes to this type.\n\n    edgetype : Python type, optional\n       Convert edges to this type.\n\n    comments : string, optional\n       Marker for comment lines\n\n    delimiter : string, optional\n       Separator for node labels.  The default is whitespace.\n\n    Returns\n    -------\n    G: NetworkX graph\n        The graph corresponding to the lines in multiline adjacency list format.\n\n    Examples\n    --------\n    >>> lines = [\n    ...     "1 2",\n    ...     "2 {\'weight\':3, \'name\': \'Frodo\'}",\n    ...     "3 {}",\n    ...     "2 1",\n    ...     "5 {\'weight\':6, \'name\': \'Saruman\'}",\n    ... ]\n    >>> G = nx.parse_multiline_adjlist(iter(lines), nodetype=int)\n    >>> list(G)\n    [1, 2, 3, 5]\n\n    '
    from ast import literal_eval
    G = nx.empty_graph(0, create_using)
    for line in lines:
        p = line.find(comments)
        if p >= 0:
            line = line[:p]
        if not line:
            continue
        try:
            (u, deg) = line.strip().split(delimiter)
            deg = int(deg)
        except BaseException as err:
            raise TypeError(f'Failed to read node and degree on line ({line})') from err
        if nodetype is not None:
            try:
                u = nodetype(u)
            except BaseException as err:
                raise TypeError(f'Failed to convert node ({u}) to type {nodetype}') from err
        G.add_node(u)
        for i in range(deg):
            while True:
                try:
                    line = next(lines)
                except StopIteration as err:
                    msg = f'Failed to find neighbor for node ({u})'
                    raise TypeError(msg) from err
                p = line.find(comments)
                if p >= 0:
                    line = line[:p]
                if line:
                    break
            vlist = line.strip().split(delimiter)
            numb = len(vlist)
            if numb < 1:
                continue
            v = vlist.pop(0)
            data = ''.join(vlist)
            if nodetype is not None:
                try:
                    v = nodetype(v)
                except BaseException as err:
                    raise TypeError(f'Failed to convert node ({v}) to type {nodetype}') from err
            if edgetype is not None:
                try:
                    edgedata = {'weight': edgetype(data)}
                except BaseException as err:
                    raise TypeError(f'Failed to convert edge data ({data}) to type {edgetype}') from err
            else:
                try:
                    edgedata = literal_eval(data)
                except:
                    edgedata = {}
            G.add_edge(u, v, **edgedata)
    return G

@open_file(0, mode='rb')
@nx._dispatch(graphs=None)
def read_multiline_adjlist(path, comments='#', delimiter=None, create_using=None, nodetype=None, edgetype=None, encoding='utf-8'):
    if False:
        print('Hello World!')
    'Read graph in multi-line adjacency list format from path.\n\n    Parameters\n    ----------\n    path : string or file\n       Filename or file handle to read.\n       Filenames ending in .gz or .bz2 will be uncompressed.\n\n    create_using : NetworkX graph constructor, optional (default=nx.Graph)\n       Graph type to create. If graph instance, then cleared before populated.\n\n    nodetype : Python type, optional\n       Convert nodes to this type.\n\n    edgetype : Python type, optional\n       Convert edge data to this type.\n\n    comments : string, optional\n       Marker for comment lines\n\n    delimiter : string, optional\n       Separator for node labels.  The default is whitespace.\n\n    Returns\n    -------\n    G: NetworkX graph\n\n    Examples\n    --------\n    >>> G = nx.path_graph(4)\n    >>> nx.write_multiline_adjlist(G, "test.adjlist")\n    >>> G = nx.read_multiline_adjlist("test.adjlist")\n\n    The path can be a file or a string with the name of the file. If a\n    file s provided, it has to be opened in \'rb\' mode.\n\n    >>> fh = open("test.adjlist", "rb")\n    >>> G = nx.read_multiline_adjlist(fh)\n\n    Filenames ending in .gz or .bz2 will be compressed.\n\n    >>> nx.write_multiline_adjlist(G, "test.adjlist.gz")\n    >>> G = nx.read_multiline_adjlist("test.adjlist.gz")\n\n    The optional nodetype is a function to convert node strings to nodetype.\n\n    For example\n\n    >>> G = nx.read_multiline_adjlist("test.adjlist", nodetype=int)\n\n    will attempt to convert all nodes to integer type.\n\n    The optional edgetype is a function to convert edge data strings to\n    edgetype.\n\n    >>> G = nx.read_multiline_adjlist("test.adjlist")\n\n    The optional create_using parameter is a NetworkX graph container.\n    The default is Graph(), an undirected graph.  To read the data as\n    a directed graph use\n\n    >>> G = nx.read_multiline_adjlist("test.adjlist", create_using=nx.DiGraph)\n\n    Notes\n    -----\n    This format does not store graph, node, or edge data.\n\n    See Also\n    --------\n    write_multiline_adjlist\n    '
    lines = (line.decode(encoding) for line in path)
    return parse_multiline_adjlist(lines, comments=comments, delimiter=delimiter, create_using=create_using, nodetype=nodetype, edgetype=edgetype)