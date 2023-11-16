"""
Finds the transitive closure of a graph.

reference: https://en.wikipedia.org/wiki/Transitive_closure#In_graph_theory
"""

class Graph:
    """
    This class represents a directed graph using adjacency lists
    """

    def __init__(self, vertices):
        if False:
            for i in range(10):
                print('nop')
        self.vertex_count = vertices
        self.graph = {}
        self.closure = [[0 for j in range(vertices)] for i in range(vertices)]

    def add_edge(self, source, target):
        if False:
            print('Hello World!')
        '\n        Adds a directed edge to the graph\n        '
        if source in self.graph:
            self.graph[source].append(target)
        else:
            self.graph[source] = [target]

    def dfs_util(self, source, target):
        if False:
            print('Hello World!')
        '\n        A recursive DFS traversal function that finds\n        all reachable vertices for source\n        '
        self.closure[source][target] = 1
        for adjacent in self.graph[target]:
            if self.closure[source][adjacent] == 0:
                self.dfs_util(source, adjacent)

    def transitive_closure(self):
        if False:
            i = 10
            return i + 15
        '\n        The function to find transitive closure. It uses\n        recursive dfs_util()\n        '
        for i in range(self.vertex_count):
            self.dfs_util(i, i)
        return self.closure