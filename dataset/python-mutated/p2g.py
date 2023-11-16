"""
This module provides the following: read and write of p2g format
used in metabolic pathway studies.

See https://web.archive.org/web/20080626113807/http://www.cs.purdue.edu/homes/koyuturk/pathway/ for a description.

The summary is included here:

A file that describes a uniquely labeled graph (with extension ".gr")
format looks like the following:


name
3 4
a
1 2
b

c
0 2

"name" is simply a description of what the graph corresponds to. The
second line displays the number of nodes and number of edges,
respectively. This sample graph contains three nodes labeled "a", "b",
and "c". The rest of the graph contains two lines for each node. The
first line for a node contains the node label. After the declaration
of the node label, the out-edges of that node in the graph are
provided. For instance, "a" is linked to nodes 1 and 2, which are
labeled "b" and "c", while the node labeled "b" has no outgoing
edges. Observe that node labeled "c" has an outgoing edge to
itself. Indeed, self-loops are allowed. Node index starts from 0.

"""
import networkx as nx
from networkx.utils import open_file

@open_file(1, mode='w')
def write_p2g(G, path, encoding='utf-8'):
    if False:
        for i in range(10):
            print('nop')
    'Write NetworkX graph in p2g format.\n\n    Notes\n    -----\n    This format is meant to be used with directed graphs with\n    possible self loops.\n    '
    path.write(f'{G.name}\n'.encode(encoding))
    path.write(f'{G.order()} {G.size()}\n'.encode(encoding))
    nodes = list(G)
    nodenumber = dict(zip(nodes, range(len(nodes))))
    for n in nodes:
        path.write(f'{n}\n'.encode(encoding))
        for nbr in G.neighbors(n):
            path.write(f'{nodenumber[nbr]} '.encode(encoding))
        path.write('\n'.encode(encoding))

@open_file(0, mode='r')
@nx._dispatch(graphs=None)
def read_p2g(path, encoding='utf-8'):
    if False:
        while True:
            i = 10
    'Read graph in p2g format from path.\n\n    Returns\n    -------\n    MultiDiGraph\n\n    Notes\n    -----\n    If you want a DiGraph (with no self loops allowed and no edge data)\n    use D=nx.DiGraph(read_p2g(path))\n    '
    lines = (line.decode(encoding) for line in path)
    G = parse_p2g(lines)
    return G

@nx._dispatch(graphs=None)
def parse_p2g(lines):
    if False:
        i = 10
        return i + 15
    'Parse p2g format graph from string or iterable.\n\n    Returns\n    -------\n    MultiDiGraph\n    '
    description = next(lines).strip()
    G = nx.MultiDiGraph(name=description, selfloops=True)
    (nnodes, nedges) = map(int, next(lines).split())
    nodelabel = {}
    nbrs = {}
    for i in range(nnodes):
        n = next(lines).strip()
        nodelabel[i] = n
        G.add_node(n)
        nbrs[n] = map(int, next(lines).split())
    for n in G:
        for nbr in nbrs[n]:
            G.add_edge(n, nodelabel[nbr])
    return G