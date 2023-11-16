class EdgeNode:

    def __init__(self, vi, vj, val):
        if False:
            while True:
                i = 10
        self.vi = vi
        self.vj = vj
        self.val = val

class Graph:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.edges = []

    def creatGraph(self, edges=[]):
        if False:
            return 10
        for (vi, vj, val) in edges:
            self.add_edge(vi, vj, val)

    def add_edge(self, vi, vj, val):
        if False:
            for i in range(10):
                print('nop')
        edge = EdgeNode(vi, vj, val)
        self.edges.append(edge)

    def get_edge(self, vi, vj):
        if False:
            while True:
                i = 10
        for edge in self.edges:
            if vi == edge.vi and vj == edge.vj:
                val = edge.val
                return val
        return None

    def printGraph(self):
        if False:
            print('Hello World!')
        for edge in self.edges:
            print(str(edge.vi) + ' - ' + str(edge.vj) + ' : ' + str(edge.val))
graph = Graph()
edges = [[1, 2, 5], [1, 5, 6], [2, 4, 7], [4, 3, 9], [3, 1, 2], [5, 6, 8], [6, 4, 3]]
graph.creatGraph(edges)
print(graph.get_edge(3, 4))
graph.printGraph()