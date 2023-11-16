class EdgeNode:

    def __init__(self, vj, val):
        if False:
            i = 10
            return i + 15
        self.vj = vj
        self.val = val
        self.next = None

class VertexNode:

    def __init__(self, vi):
        if False:
            print('Hello World!')
        self.vi = vi
        self.head = None

class Graph:

    def __init__(self, ver_count):
        if False:
            return 10
        self.ver_count = ver_count
        self.vertices = []
        for vi in range(ver_count):
            vertex = VertexNode(vi)
            self.vertices.append(vertex)

    def __valid(self, v):
        if False:
            print('Hello World!')
        return 0 <= v <= self.ver_count

    def creatGraph(self, edges=[]):
        if False:
            i = 10
            return i + 15
        for (vi, vj, val) in edges:
            self.add_edge(vi, vj, val)

    def add_edge(self, vi, vj, val):
        if False:
            print('Hello World!')
        if not self.__valid(vi) or not self.__valid(vj):
            raise ValueError(str(vi) + ' or ' + str(vj) + ' is not a valid vertex.')
        vertex = self.vertices[vi]
        edge = EdgeNode(vj, val)
        edge.next = vertex.head
        vertex.head = edge

    def get_edge(self, vi, vj):
        if False:
            while True:
                i = 10
        if not self.__valid(vi) or not self.__valid(vj):
            raise ValueError(str(vi) + ' or ' + str(vj) + ' is not a valid vertex.')
        vertex = self.vertices[vi]
        cur_edge = vertex.head
        while cur_edge:
            if cur_edge.vj == vj:
                return cur_edge.val
            cur_edge = cur_edge.next
        return None

    def printGraph(self):
        if False:
            return 10
        for vertex in self.vertices:
            cur_edge = vertex.head
            while cur_edge:
                print(str(vertex.vi) + ' - ' + str(cur_edge.vj) + ' : ' + str(cur_edge.val))
                cur_edge = cur_edge.next
graph = Graph(7)
edges = [[1, 2, 5], [1, 5, 6], [2, 4, 7], [4, 3, 9], [3, 1, 2], [5, 6, 8], [6, 4, 3]]
graph.creatGraph(edges)
print(graph.get_edge(3, 4))
graph.printGraph()