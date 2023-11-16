class UnionFind(object):

    def __init__(self, n):
        if False:
            print('Hello World!')
        self.set = range(n)
        self.count = n

    def find_set(self, x):
        if False:
            print('Hello World!')
        if self.set[x] != x:
            self.set[x] = self.find_set(self.set[x])
        return self.set[x]

    def union_set(self, x, y):
        if False:
            return 10
        (x_root, y_root) = map(self.find_set, (x, y))
        if x_root != y_root:
            self.set[min(x_root, y_root)] = max(x_root, y_root)
            self.count -= 1

class Solution(object):

    def countComponents(self, n, edges):
        if False:
            print('Hello World!')
        '\n        :type n: int\n        :type edges: List[List[int]]\n        :rtype: int\n        '
        union_find = UnionFind(n)
        for (i, j) in edges:
            union_find.union_set(i, j)
        return union_find.count