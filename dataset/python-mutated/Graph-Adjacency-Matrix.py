class Graph:

    def __init__(self, ver_count):
        if False:
            return 10
        self.ver_count = ver_count
        self.adj_matrix = [[None for _ in range(ver_count)] for _ in range(ver_count)]

    def __valid(self, v):
        if False:
            return 10
        return 0 <= v <= self.ver_count

    def creatGraph(self, edges=[]):
        if False:
            return 10
        for (vi, vj, val) in edges:
            self.add_edge(vi, vj, val)

    def add_edge(self, vi, vj, val):
        if False:
            return 10
        if not self.__valid(vi) or not self.__valid(vj):
            raise ValueError(str(vi) + ' or ' + str(vj) + ' is not a valid vertex.')
        self.adj_matrix[vi][vj] = val

    def get_edge(self, vi, vj):
        if False:
            for i in range(10):
                print('nop')
        if not self.__valid(vi) or not self.__valid(vj):
            raise ValueError(str(vi) + ' or ' + str(vj) + ' is not a valid vertex.')
        return self.adj_matrix[vi][vj]

    def printGraph(self):
        if False:
            print('Hello World!')
        for vi in range(self.ver_count):
            for vj in range(self.ver_count):
                val = self.get_edge(vi, vj)
                if val:
                    print(str(vi) + ' - ' + str(vj) + ' : ' + str(val))
graph = Graph(5)
edges = [[1, 2, 5], [2, 1, 5], [1, 3, 30], [3, 1, 30], [2, 3, 14], [3, 2, 14], [2, 4, 26], [4, 2, 26]]
graph.creatGraph(edges)
print(graph.get_edge(3, 4))
graph.printGraph()