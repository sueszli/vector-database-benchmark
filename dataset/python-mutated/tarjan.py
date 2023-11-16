"""
Implements Tarjan's algorithm for finding strongly connected components
in a graph.
https://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm
"""
from algorithms.graph.graph import DirectedGraph

class Tarjan:
    """
    A directed graph used for finding strongly connected components
    """

    def __init__(self, dict_graph):
        if False:
            while True:
                i = 10
        self.graph = DirectedGraph(dict_graph)
        self.index = 0
        self.stack = []
        for vertex in self.graph.nodes:
            vertex.index = None
        self.sccs = []
        for vertex in self.graph.nodes:
            if vertex.index is None:
                self.strongconnect(vertex, self.sccs)

    def strongconnect(self, vertex, sccs):
        if False:
            while True:
                i = 10
        '\n        Given a vertex, adds all successors of the given vertex to the same connected component\n        '
        vertex.index = self.index
        vertex.lowlink = self.index
        self.index += 1
        self.stack.append(vertex)
        vertex.on_stack = True
        for adjacent in self.graph.adjacency_list[vertex]:
            if adjacent.index is None:
                self.strongconnect(adjacent, sccs)
                vertex.lowlink = min(vertex.lowlink, adjacent.lowlink)
            elif adjacent.on_stack:
                vertex.lowlink = min(vertex.lowlink, adjacent.index)
        if vertex.lowlink == vertex.index:
            scc = []
            while True:
                adjacent = self.stack.pop()
                adjacent.on_stack = False
                scc.append(adjacent)
                if adjacent == vertex:
                    break
            scc.sort()
            sccs.append(scc)