"""
*****
Pajek
*****
Read graphs in Pajek format.

This implementation handles directed and undirected graphs including
those with self loops and parallel edges.

Format
------
See http://vlado.fmf.uni-lj.si/pub/networks/pajek/doc/draweps.htm
for format information.

"""
import warnings
import networkx as nx
from networkx.utils import open_file
__all__ = ['read_pajek', 'parse_pajek', 'generate_pajek', 'write_pajek']

def generate_pajek(G):
    if False:
        print('Hello World!')
    'Generate lines in Pajek graph format.\n\n    Parameters\n    ----------\n    G : graph\n       A Networkx graph\n\n    References\n    ----------\n    See http://vlado.fmf.uni-lj.si/pub/networks/pajek/doc/draweps.htm\n    for format information.\n    '
    if G.name == '':
        name = 'NetworkX'
    else:
        name = G.name
    yield f'*vertices {G.order()}'
    nodes = list(G)
    nodenumber = dict(zip(nodes, range(1, len(nodes) + 1)))
    for n in nodes:
        na = G.nodes.get(n, {}).copy()
        x = na.pop('x', 0.0)
        y = na.pop('y', 0.0)
        try:
            id = int(na.pop('id', nodenumber[n]))
        except ValueError as err:
            err.args += ("Pajek format requires 'id' to be an int(). Refer to the 'Relabeling nodes' section.",)
            raise
        nodenumber[n] = id
        shape = na.pop('shape', 'ellipse')
        s = ' '.join(map(make_qstr, (id, n, x, y, shape)))
        for (k, v) in na.items():
            if isinstance(v, str) and v.strip() != '':
                s += f' {make_qstr(k)} {make_qstr(v)}'
            else:
                warnings.warn(f"Node attribute {k} is not processed. {('Empty attribute' if isinstance(v, str) else 'Non-string attribute')}.")
        yield s
    if G.is_directed():
        yield '*arcs'
    else:
        yield '*edges'
    for (u, v, edgedata) in G.edges(data=True):
        d = edgedata.copy()
        value = d.pop('weight', 1.0)
        s = ' '.join(map(make_qstr, (nodenumber[u], nodenumber[v], value)))
        for (k, v) in d.items():
            if isinstance(v, str) and v.strip() != '':
                s += f' {make_qstr(k)} {make_qstr(v)}'
            else:
                warnings.warn(f"Edge attribute {k} is not processed. {('Empty attribute' if isinstance(v, str) else 'Non-string attribute')}.")
        yield s

@open_file(1, mode='wb')
def write_pajek(G, path, encoding='UTF-8'):
    if False:
        return 10
    'Write graph in Pajek format to path.\n\n    Parameters\n    ----------\n    G : graph\n       A Networkx graph\n    path : file or string\n       File or filename to write.\n       Filenames ending in .gz or .bz2 will be compressed.\n\n    Examples\n    --------\n    >>> G = nx.path_graph(4)\n    >>> nx.write_pajek(G, "test.net")\n\n    Warnings\n    --------\n    Optional node attributes and edge attributes must be non-empty strings.\n    Otherwise it will not be written into the file. You will need to\n    convert those attributes to strings if you want to keep them.\n\n    References\n    ----------\n    See http://vlado.fmf.uni-lj.si/pub/networks/pajek/doc/draweps.htm\n    for format information.\n    '
    for line in generate_pajek(G):
        line += '\n'
        path.write(line.encode(encoding))

@open_file(0, mode='rb')
@nx._dispatch(graphs=None)
def read_pajek(path, encoding='UTF-8'):
    if False:
        for i in range(10):
            print('nop')
    'Read graph in Pajek format from path.\n\n    Parameters\n    ----------\n    path : file or string\n       File or filename to write.\n       Filenames ending in .gz or .bz2 will be uncompressed.\n\n    Returns\n    -------\n    G : NetworkX MultiGraph or MultiDiGraph.\n\n    Examples\n    --------\n    >>> G = nx.path_graph(4)\n    >>> nx.write_pajek(G, "test.net")\n    >>> G = nx.read_pajek("test.net")\n\n    To create a Graph instead of a MultiGraph use\n\n    >>> G1 = nx.Graph(G)\n\n    References\n    ----------\n    See http://vlado.fmf.uni-lj.si/pub/networks/pajek/doc/draweps.htm\n    for format information.\n    '
    lines = (line.decode(encoding) for line in path)
    return parse_pajek(lines)

@nx._dispatch(graphs=None)
def parse_pajek(lines):
    if False:
        print('Hello World!')
    'Parse Pajek format graph from string or iterable.\n\n    Parameters\n    ----------\n    lines : string or iterable\n       Data in Pajek format.\n\n    Returns\n    -------\n    G : NetworkX graph\n\n    See Also\n    --------\n    read_pajek\n\n    '
    import shlex
    if isinstance(lines, str):
        lines = iter(lines.split('\n'))
    lines = iter([line.rstrip('\n') for line in lines])
    G = nx.MultiDiGraph()
    labels = []
    while lines:
        try:
            l = next(lines)
        except:
            break
        if l.lower().startswith('*network'):
            try:
                (label, name) = l.split(None, 1)
            except ValueError:
                pass
            else:
                G.graph['name'] = name
        elif l.lower().startswith('*vertices'):
            nodelabels = {}
            (l, nnodes) = l.split()
            for i in range(int(nnodes)):
                l = next(lines)
                try:
                    splitline = [x.decode('utf-8') for x in shlex.split(str(l).encode('utf-8'))]
                except AttributeError:
                    splitline = shlex.split(str(l))
                (id, label) = splitline[0:2]
                labels.append(label)
                G.add_node(label)
                nodelabels[id] = label
                G.nodes[label]['id'] = id
                try:
                    (x, y, shape) = splitline[2:5]
                    G.nodes[label].update({'x': float(x), 'y': float(y), 'shape': shape})
                except:
                    pass
                extra_attr = zip(splitline[5::2], splitline[6::2])
                G.nodes[label].update(extra_attr)
        elif l.lower().startswith('*edges') or l.lower().startswith('*arcs'):
            if l.lower().startswith('*edge'):
                G = nx.MultiGraph(G)
            if l.lower().startswith('*arcs'):
                G = G.to_directed()
            for l in lines:
                try:
                    splitline = [x.decode('utf-8') for x in shlex.split(str(l).encode('utf-8'))]
                except AttributeError:
                    splitline = shlex.split(str(l))
                if len(splitline) < 2:
                    continue
                (ui, vi) = splitline[0:2]
                u = nodelabels.get(ui, ui)
                v = nodelabels.get(vi, vi)
                edge_data = {}
                try:
                    w = splitline[2:3]
                    edge_data.update({'weight': float(w[0])})
                except:
                    pass
                extra_attr = zip(splitline[3::2], splitline[4::2])
                edge_data.update(extra_attr)
                G.add_edge(u, v, **edge_data)
        elif l.lower().startswith('*matrix'):
            G = nx.DiGraph(G)
            adj_list = ((labels[row], labels[col], {'weight': int(data)}) for (row, line) in enumerate(lines) for (col, data) in enumerate(line.split()) if int(data) != 0)
            G.add_edges_from(adj_list)
    return G

def make_qstr(t):
    if False:
        i = 10
        return i + 15
    'Returns the string representation of t.\n    Add outer double-quotes if the string has a space.\n    '
    if not isinstance(t, str):
        t = str(t)
    if ' ' in t:
        t = f'"{t}"'
    return t