"""
Determine if there is a path between nodes in a graph
"""
from collections import defaultdict

class Graph:
    """
    A directed graph
    """

    def __init__(self, vertex_count):
        if False:
            print('Hello World!')
        self.vertex_count = vertex_count
        self.graph = defaultdict(list)
        self.has_path = False

    def add_edge(self, source, target):
        if False:
            return 10
        '\n        Add a new directed edge to the graph\n        '
        self.graph[source].append(target)

    def dfs(self, source, target):
        if False:
            i = 10
            return i + 15
        '\n        Determine if there is a path from source to target using a depth first search\n        '
        visited = [False] * self.vertex_count
        self.dfsutil(visited, source, target)

    def dfsutil(self, visited, source, target):
        if False:
            for i in range(10):
                print('nop')
        '\n        Determine if there is a path from source to target using a depth first search.\n        :param: visited should be an array of booleans determining if the\n        corresponding vertex has been visited already\n        '
        visited[source] = True
        for i in self.graph[source]:
            if target in self.graph[source]:
                self.has_path = True
                return
            if not visited[i]:
                self.dfsutil(visited, source, i)

    def is_reachable(self, source, target):
        if False:
            while True:
                i = 10
        '\n        Determine if there is a path from source to target\n        '
        self.has_path = False
        self.dfs(source, target)
        return self.has_path