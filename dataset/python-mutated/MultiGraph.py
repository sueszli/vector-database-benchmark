"""get/set abstraction for multi-graph representation."""
from functools import reduce

class MultiGraph:
    """A directed multigraph abstraction with labeled edges."""

    def __init__(self, nodes=()):
        if False:
            print('Hello World!')
        'Initialize a new MultiGraph object.'
        self._adjacency_list = {}
        for n in nodes:
            self._adjacency_list[n] = set()
        self._label_map = {}

    def __eq__(self, g):
        if False:
            i = 10
            return i + 15
        'Return true if g is equal to this graph.'
        return isinstance(g, MultiGraph) and self._adjacency_list == g._adjacency_list and (self._label_map == g._label_map)

    def __repr__(self):
        if False:
            print('Hello World!')
        'Return a unique string representation of this graph.'
        s = '<MultiGraph: '
        for key in sorted(self._adjacency_list):
            values = sorted(self._adjacency_list[key])
            s += f"({key!r}: {','.join((repr(v) for v in values))})"
        return s + '>'

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        'Return a concise string description of this graph.'
        nodenum = len(self._adjacency_list)
        edgenum = reduce(lambda x, y: x + y, [len(v) for v in self._adjacency_list.values()])
        labelnum = len(self._label_map)
        return '<MultiGraph: ' + str(nodenum) + ' node(s), ' + str(edgenum) + ' edge(s), ' + str(labelnum) + ' unique label(s)>'

    def add_node(self, node):
        if False:
            for i in range(10):
                print('nop')
        'Add a node to this graph.'
        if node not in self._adjacency_list:
            self._adjacency_list[node] = set()

    def add_edge(self, source, to, label=None):
        if False:
            print('Hello World!')
        'Add an edge to this graph.'
        if source not in self._adjacency_list:
            raise ValueError('Unknown <from> node: ' + str(source))
        if to not in self._adjacency_list:
            raise ValueError('Unknown <to> node: ' + str(to))
        edge = (to, label)
        self._adjacency_list[source].add(edge)
        if label not in self._label_map:
            self._label_map[label] = set()
        self._label_map[label].add((source, to))

    def child_edges(self, parent):
        if False:
            return 10
        'Return a list of (child, label) pairs for parent.'
        if parent not in self._adjacency_list:
            raise ValueError('Unknown <parent> node: ' + str(parent))
        return sorted(self._adjacency_list[parent])

    def children(self, parent):
        if False:
            while True:
                i = 10
        'Return a list of unique children for parent.'
        return sorted({x[0] for x in self.child_edges(parent)})

    def edges(self, label):
        if False:
            return 10
        'Return a list of all the edges with this label.'
        if label not in self._label_map:
            raise ValueError('Unknown label: ' + str(label))
        return sorted(self._label_map[label])

    def labels(self):
        if False:
            i = 10
            return i + 15
        'Return a list of all the edge labels in this graph.'
        return sorted(self._label_map.keys())

    def nodes(self):
        if False:
            for i in range(10):
                print('nop')
        'Return a list of the nodes in this graph.'
        return list(self._adjacency_list.keys())

    def parent_edges(self, child):
        if False:
            for i in range(10):
                print('nop')
        'Return a list of (parent, label) pairs for child.'
        if child not in self._adjacency_list:
            raise ValueError('Unknown <child> node: ' + str(child))
        parents = []
        for (parent, children) in self._adjacency_list.items():
            for x in children:
                if x[0] == child:
                    parents.append((parent, x[1]))
        return sorted(parents)

    def parents(self, child):
        if False:
            i = 10
            return i + 15
        'Return a list of unique parents for child.'
        return sorted({x[0] for x in self.parent_edges(child)})

    def remove_node(self, node):
        if False:
            print('Hello World!')
        'Remove node and all edges connected to it.'
        if node not in self._adjacency_list:
            raise ValueError('Unknown node: ' + str(node))
        del self._adjacency_list[node]
        for n in self._adjacency_list:
            self._adjacency_list[n] = {x for x in self._adjacency_list[n] if x[0] != node}
        for label in list(self._label_map.keys()):
            lm = {x for x in self._label_map[label] if x[0] != node and x[1] != node}
            if lm:
                self._label_map[label] = lm
            else:
                del self._label_map[label]

    def remove_edge(self, parent, child, label):
        if False:
            while True:
                i = 10
        'Remove edge (NOT IMPLEMENTED).'
        raise NotImplementedError('remove_edge is not yet implemented')

def df_search(graph, root=None):
    if False:
        i = 10
        return i + 15
    'Depth first search of g.\n\n    Returns a list of all nodes that can be reached from the root node\n    in depth-first order.\n\n    If root is not given, the search will be rooted at an arbitrary node.\n    '
    seen = {}
    search = []
    if len(graph.nodes()) < 1:
        return search
    if root is None:
        root = graph.nodes()[0]
    seen[root] = 1
    search.append(root)
    current = graph.children(root)
    while len(current) > 0:
        node = current[0]
        current = current[1:]
        if node not in seen:
            search.append(node)
            seen[node] = 1
            current = graph.children(node) + current
    return search

def bf_search(graph, root=None):
    if False:
        i = 10
        return i + 15
    'Breadth first search of g.\n\n    Returns a list of all nodes that can be reached from the root node\n    in breadth-first order.\n\n    If root is not given, the search will be rooted at an arbitrary node.\n    '
    seen = {}
    search = []
    if len(graph.nodes()) < 1:
        return search
    if root is None:
        root = graph.nodes()[0]
    seen[root] = 1
    search.append(root)
    current = graph.children(root)
    while len(current) > 0:
        node = current[0]
        current = current[1:]
        if node not in seen:
            search.append(node)
            seen[node] = 1
            current.extend(graph.children(node))
    return search