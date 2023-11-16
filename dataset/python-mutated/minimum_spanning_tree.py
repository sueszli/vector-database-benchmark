"""
Minimum spanning tree (MST) is going to use an undirected graph
"""
import sys

class Edge:
    """
    An edge of an undirected graph
    """

    def __init__(self, source, target, weight):
        if False:
            print('Hello World!')
        self.source = source
        self.target = target
        self.weight = weight

class DisjointSet:
    """
    The disjoint set is represented with an list <n> of integers where
    <n[i]> is the parent of the node at position <i>.
    If <n[i]> = <i>, <i> it's a root, or a head, of a set
    """

    def __init__(self, size):
        if False:
            while True:
                i = 10
        '\n        Args:\n            n (int): Number of vertices in the graph\n        '
        self.parent = [None] * size
        self.size = [1] * size
        for i in range(size):
            self.parent[i] = i

    def merge_set(self, node1, node2):
        if False:
            for i in range(10):
                print('nop')
        '\n        Args:\n            node1, node2 (int): Indexes of nodes whose sets will be merged.\n        '
        node1 = self.find_set(node1)
        node2 = self.find_set(node2)
        if self.size[node1] < self.size[node2]:
            self.parent[node1] = node2
            self.size[node2] += self.size[node1]
        else:
            self.parent[node2] = node1
            self.size[node1] += self.size[node2]

    def find_set(self, node):
        if False:
            for i in range(10):
                print('nop')
        '\n        Get the root element of the set containing <a>\n        '
        if self.parent[node] != node:
            self.parent[node] = self.find_set(self.parent[node])
        return self.parent[node]

def kruskal(vertex_count, edges, forest):
    if False:
        return 10
    "\n    Args:\n        vertex_count (int): Number of vertices in the graph\n        edges (list of Edge): Edges of the graph\n        forest (DisjointSet): DisjointSet of the vertices\n    Returns:\n        int: sum of weights of the minnimum spanning tree\n\n    Kruskal algorithm:\n        This algorithm will find the optimal graph with less edges and less\n        total weight to connect all vertices (MST), the MST will always contain\n        n-1 edges because it's the minimum required to connect n vertices.\n\n    Procedure:\n        Sort the edges (criteria: less weight).\n        Only take edges of nodes in different sets.\n        If we take a edge, we need to merge the sets to discard these.\n        After repeat this until select n-1 edges, we will have the complete MST.\n    "
    edges.sort(key=lambda edge: edge.weight)
    mst = []
    for edge in edges:
        set_u = forest.find_set(edge.u)
        set_v = forest.find_set(edge.v)
        if set_u != set_v:
            forest.merge_set(set_u, set_v)
            mst.append(edge)
            if len(mst) == vertex_count - 1:
                break
    return sum([edge.weight for edge in mst])

def main():
    if False:
        while True:
            i = 10
    '\n    Test. How input works:\n    Input consists of different weighted, connected, undirected graphs.\n    line 1:\n      integers n, m\n    lines 2..m+2:\n      edge with the format -> node index u, node index v, integer weight\n\n    Samples of input:\n\n    5 6\n    1 2 3\n    1 3 8\n    2 4 5\n    3 4 2\n    3 5 4\n    4 5 6\n\n    3 3\n    2 1 20\n    3 1 20\n    2 3 100\n\n    Sum of weights of the optimal paths:\n    14, 40\n    '
    for size in sys.stdin:
        (vertex_count, edge_count) = map(int, size.split())
        forest = DisjointSet(edge_count)
        edges = [None] * edge_count
        for i in range(edge_count):
            (source, target, weight) = map(int, input().split())
            source -= 1
            target -= 1
            edges[i] = Edge(source, target, weight)
        print('MST weights sum:', kruskal(vertex_count, edges, forest))
if __name__ == '__main__':
    main()