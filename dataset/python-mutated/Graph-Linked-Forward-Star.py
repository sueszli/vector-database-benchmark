class EdgeNode:

    def __init__(self, vj, val):
        if False:
            i = 10
            return i + 15
        self.vj = vj
        self.val = val
        self.next = None

class Graph:

    def __init__(self, ver_count, edge_count):
        if False:
            return 10
        self.ver_count = ver_count
        self.edge_count = edge_count
        self.head = [-1 for _ in range(ver_count)]
        self.edges = []

    def __valid(self, v):
        if False:
            print('Hello World!')
        return 0 <= v <= self.ver_count

    def creatGraph(self, edges=[]):
        if False:
            i = 10
            return i + 15
        for i in range(len(edges)):
            (vi, vj, val) = edges[i]
            self.add_edge(i, vi, vj, val)

    def add_edge(self, index, vi, vj, val):
        if False:
            while True:
                i = 10
        if not self.__valid(vi) or not self.__valid(vj):
            raise ValueError(str(vi) + ' or ' + str(vj) + ' is not a valid vertex.')
        edge = EdgeNode(vj, val)
        edge.next = self.head[vi]
        self.edges.append(edge)
        self.head[vi] = index

    def get_edge(self, vi, vj):
        if False:
            i = 10
            return i + 15
        if not self.__valid(vi) or not self.__valid(vj):
            raise ValueError(str(vi) + ' or ' + str(vj) + ' is not a valid vertex.')
        index = self.head[vi]
        while index != -1:
            if vj == self.edges[index].vj:
                return self.edges[index].val
            index = self.edges[index].next
        return None

    def printGraph(self):
        if False:
            print('Hello World!')
        for vi in range(self.ver_count):
            index = self.head[vi]
            while index != -1:
                print(str(vi) + ' - ' + str(self.edges[index].vj) + ' : ' + str(self.edges[index].val))
                index = self.edges[index].next
graph = Graph(7, 7)
edges = [[1, 2, 5], [1, 5, 6], [2, 4, 7], [4, 3, 9], [3, 1, 2], [5, 6, 8], [6, 4, 3]]
graph.creatGraph(edges)
print(graph.get_edge(4, 3))
print(graph.get_edge(4, 5))
graph.printGraph()