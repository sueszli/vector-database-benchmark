"""
Read graphs in LEDA format.

LEDA is a C++ class library for efficient data types and algorithms.

Format
------
See http://www.algorithmic-solutions.info/leda_guide/graphs/leda_native_graph_fileformat.html

"""
__all__ = ['read_leda', 'parse_leda']
import networkx as nx
from networkx.exception import NetworkXError
from networkx.utils import open_file

@open_file(0, mode='rb')
@nx._dispatch(graphs=None)
def read_leda(path, encoding='UTF-8'):
    if False:
        return 10
    "Read graph in LEDA format from path.\n\n    Parameters\n    ----------\n    path : file or string\n       File or filename to read.  Filenames ending in .gz or .bz2  will be\n       uncompressed.\n\n    Returns\n    -------\n    G : NetworkX graph\n\n    Examples\n    --------\n    G=nx.read_leda('file.leda')\n\n    References\n    ----------\n    .. [1] http://www.algorithmic-solutions.info/leda_guide/graphs/leda_native_graph_fileformat.html\n    "
    lines = (line.decode(encoding) for line in path)
    G = parse_leda(lines)
    return G

@nx._dispatch(graphs=None)
def parse_leda(lines):
    if False:
        for i in range(10):
            print('nop')
    'Read graph in LEDA format from string or iterable.\n\n    Parameters\n    ----------\n    lines : string or iterable\n       Data in LEDA format.\n\n    Returns\n    -------\n    G : NetworkX graph\n\n    Examples\n    --------\n    G=nx.parse_leda(string)\n\n    References\n    ----------\n    .. [1] http://www.algorithmic-solutions.info/leda_guide/graphs/leda_native_graph_fileformat.html\n    '
    if isinstance(lines, str):
        lines = iter(lines.split('\n'))
    lines = iter([line.rstrip('\n') for line in lines if not (line.startswith(('#', '\n')) or line == '')])
    for i in range(3):
        next(lines)
    du = int(next(lines))
    if du == -1:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    n = int(next(lines))
    node = {}
    for i in range(1, n + 1):
        symbol = next(lines).rstrip().strip('|{}|  ')
        if symbol == '':
            symbol = str(i)
        node[i] = symbol
    G.add_nodes_from([s for (i, s) in node.items()])
    m = int(next(lines))
    for i in range(m):
        try:
            (s, t, reversal, label) = next(lines).split()
        except BaseException as err:
            raise NetworkXError(f'Too few fields in LEDA.GRAPH edge {i + 1}') from err
        G.add_edge(node[int(s)], node[int(t)], label=label[2:-2])
    return G