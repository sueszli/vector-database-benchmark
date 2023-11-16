class UnionFind(object):

    def __init__(self, n):
        if False:
            i = 10
            return i + 15
        self.set = range(n)

    def find_set(self, x):
        if False:
            while True:
                i = 10
        if self.set[x] != x:
            self.set[x] = self.find_set(self.set[x])
        return self.set[x]

    def union_set(self, x, y):
        if False:
            for i in range(10):
                print('nop')
        (x_root, y_root) = map(self.find_set, (x, y))
        if x_root == y_root:
            return False
        self.set[min(x_root, y_root)] = max(x_root, y_root)
        return True

class Solution(object):

    def findRedundantConnection(self, edges):
        if False:
            while True:
                i = 10
        '\n        :type edges: List[List[int]]\n        :rtype: List[int]\n        '
        union_find = UnionFind(len(edges) + 1)
        for edge in edges:
            if not union_find.union_set(*edge):
                return edge
        return []