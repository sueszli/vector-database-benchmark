"""
In a directed graph, a strongly connected component is a set of vertices such
that for any pairs of vertices u and v there exists a path (u-...-v) that
connects them. A graph is strongly connected if it is a single strongly
connected component.
"""
from collections import defaultdict

class Graph:
    """
    A directed graph where edges are one-way (a two-way edge can be represented by using two edges).
    """

    def __init__(self, vertex_count):
        if False:
            return 10
        '\n        Create a new graph with vertex_count vertices.\n        '
        self.vertex_count = vertex_count
        self.graph = defaultdict(list)

    def add_edge(self, source, target):
        if False:
            while True:
                i = 10
        '\n        Add an edge going from source to target\n        '
        self.graph[source].append(target)

    def dfs(self):
        if False:
            i = 10
            return i + 15
        '\n        Determine if all nodes are reachable from node 0\n        '
        visited = [False] * self.vertex_count
        self.dfs_util(0, visited)
        if visited == [True] * self.vertex_count:
            return True
        return False

    def dfs_util(self, source, visited):
        if False:
            for i in range(10):
                print('nop')
        '\n        Determine if all nodes are reachable from the given node\n        '
        visited[source] = True
        for adjacent in self.graph[source]:
            if not visited[adjacent]:
                self.dfs_util(adjacent, visited)

    def reverse_graph(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Create a new graph where every edge a->b is replaced with an edge b->a\n        '
        reverse_graph = Graph(self.vertex_count)
        for (source, adjacent) in self.graph.items():
            for target in adjacent:
                reverse_graph.add_edge(target, source)
        return reverse_graph

    def is_strongly_connected(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Determine if the graph is strongly connected.\n        '
        if self.dfs():
            reversed_graph = self.reverse_graph()
            if reversed_graph.dfs():
                return True
        return False