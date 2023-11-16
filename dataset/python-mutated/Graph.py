"""get/set abstraction for graph representation."""
from functools import reduce

class Graph:
    """A directed graph abstraction with labeled edges."""

    def __init__(self, nodes=()):
        if False:
            return 10
        'Initialize a new Graph object.'
        self._adjacency_list = {}
        for n in nodes:
            self._adjacency_list[n] = set()
        self._label_map = {}
        self._edge_map = {}

    def __eq__(self, g):
        if False:
            return 10
        'Return true if g is equal to this graph.'
        return isinstance(g, Graph) and self._adjacency_list == g._adjacency_list and (self._label_map == g._label_map) and (self._edge_map == g._edge_map)

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        'Return a unique string representation of this graph.'
        s = '<Graph: '
        for key in sorted(self._adjacency_list):
            values = sorted(((x, self._edge_map[key, x]) for x in list(self._adjacency_list[key])))
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
        return '<Graph: ' + str(nodenum) + ' node(s), ' + str(edgenum) + ' edge(s), ' + str(labelnum) + ' unique label(s)>'

    def add_node(self, node):
        if False:
            i = 10
            return i + 15
        'Add a node to this graph.'
        if node not in self._adjacency_list:
            self._adjacency_list[node] = set()

    def add_edge(self, source, to, label=None):
        if False:
            for i in range(10):
                print('nop')
        'Add an edge to this graph.'
        if source not in self._adjacency_list:
            raise ValueError('Unknown <from> node: ' + str(source))
        if to not in self._adjacency_list:
            raise ValueError('Unknown <to> node: ' + str(to))
        if (source, to) in self._edge_map:
            raise ValueError(str(source) + ' -> ' + str(to) + ' exists')
        self._adjacency_list[source].add(to)
        if label not in self._label_map:
            self._label_map[label] = set()
        self._label_map[label].add((source, to))
        self._edge_map[source, to] = label

    def child_edges(self, parent):
        if False:
            return 10
        'Return a list of (child, label) pairs for parent.'
        if parent not in self._adjacency_list:
            raise ValueError('Unknown <parent> node: ' + str(parent))
        return [(x, self._edge_map[parent, x]) for x in sorted(self._adjacency_list[parent])]

    def children(self, parent):
        if False:
            return 10
        'Return a list of unique children for parent.'
        return sorted(self._adjacency_list[parent])

    def edges(self, label):
        if False:
            while True:
                i = 10
        'Return a list of all the edges with this label.'
        if label not in self._label_map:
            raise ValueError('Unknown label: ' + str(label))
        return sorted(self._label_map[label])

    def labels(self):
        if False:
            return 10
        'Return a list of all the edge labels in this graph.'
        return sorted(self._label_map.keys())

    def nodes(self):
        if False:
            print('Hello World!')
        'Return a list of the nodes in this graph.'
        return list(self._adjacency_list.keys())

    def parent_edges(self, child):
        if False:
            while True:
                i = 10
        'Return a list of (parent, label) pairs for child.'
        if child not in self._adjacency_list:
            raise ValueError('Unknown <child> node: ' + str(child))
        parents = []
        for (parent, children) in self._adjacency_list.items():
            for x in children:
                if x == child:
                    parents.append((parent, self._edge_map[parent, child]))
        return sorted(parents)

    def parents(self, child):
        if False:
            return 10
        'Return a list of unique parents for child.'
        return sorted({x[0] for x in self.parent_edges(child)})

    def remove_node(self, node):
        if False:
            return 10
        'Remove node and all edges connected to it.'
        if node not in self._adjacency_list:
            raise ValueError('Unknown node: ' + str(node))
        del self._adjacency_list[node]
        for n in self._adjacency_list.keys():
            self._adjacency_list[n] = {x for x in self._adjacency_list[n] if x != node}
        for label in list(self._label_map.keys()):
            lm = {x for x in self._label_map[label] if x[0] != node and x[1] != node}
            if lm:
                self._label_map[label] = lm
            else:
                del self._label_map[label]
        for edge in list(self._edge_map.keys()):
            if edge[0] == node or edge[1] == node:
                del self._edge_map[edge]

    def remove_edge(self, parent, child, label):
        if False:
            return 10
        'Remove edge (NOT IMPLEMENTED).'
        raise NotImplementedError('remove_edge is not yet implemented')