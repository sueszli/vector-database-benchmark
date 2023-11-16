"""
Utilities for creating dot output from a MachOGraph
"""
from collections import deque
try:
    from itertools import imap
except ImportError:
    imap = map
__all__ = ['itergraphreport']

def itergraphreport(nodes, describe_edge, name='G'):
    if False:
        while True:
            i = 10
    edges = deque()
    nodetoident = {}

    def nodevisitor(node, data, outgoing, incoming):
        if False:
            i = 10
            return i + 15
        return {'label': str(node)}

    def edgevisitor(edge, data, head, tail):
        if False:
            for i in range(10):
                print('nop')
        return {}
    yield ('digraph %s {\n' % (name,))
    attr = {'rankdir': 'LR', 'concentrate': 'true'}
    cpatt = '%s="%s"'
    for item in attr.items():
        yield ('\t%s;\n' % (cpatt % item,))
    for (node, data, _outgoing, _incoming) in nodes:
        nodetoident[node] = getattr(data, 'identifier', node)
    for (node, data, outgoing, incoming) in nodes:
        for edge in imap(describe_edge, outgoing):
            edges.append(edge)
        yield ('\t"%s" [%s];\n' % (node, ','.join([cpatt % item for item in nodevisitor(node, data, outgoing, incoming).items()])))
    graph = []
    while edges:
        (edge, data, head, tail) = edges.popleft()
        if data in ('run_file', 'load_dylib'):
            graph.append((edge, data, head, tail))

    def do_graph(edges, tabs):
        if False:
            while True:
                i = 10
        edgestr = tabs + '"%s" -> "%s" [%s];\n'
        for (edge, data, head, tail) in edges:
            attribs = edgevisitor(edge, data, head, tail)
            yield (edgestr % (head, tail, ','.join([cpatt % item for item in attribs.items()])))
    for s in do_graph(graph, '\t'):
        yield s
    yield '}\n'