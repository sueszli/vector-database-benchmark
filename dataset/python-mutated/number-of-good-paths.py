import collections

class UnionFind(object):

    def __init__(self, vals):
        if False:
            for i in range(10):
                print('nop')
        self.set = range(len(vals))
        self.rank = [0] * len(vals)
        self.cnt = [collections.Counter({v: 1}) for v in vals]

    def find_set(self, x):
        if False:
            i = 10
            return i + 15
        stk = []
        while self.set[x] != x:
            stk.append(x)
            x = self.set[x]
        while stk:
            self.set[stk.pop()] = x
        return x

    def union_set(self, x, y, v):
        if False:
            i = 10
            return i + 15
        (x, y) = (self.find_set(x), self.find_set(y))
        if x == y:
            return 0
        if self.rank[x] > self.rank[y]:
            (x, y) = (y, x)
        self.set[x] = self.set[y]
        if self.rank[x] == self.rank[y]:
            self.rank[y] += 1
        (cx, cy) = (self.cnt[x][v], self.cnt[y][v])
        self.cnt[y] = collections.Counter({v: cx + cy})
        return cx * cy

class Solution(object):

    def numberOfGoodPaths(self, vals, edges):
        if False:
            i = 10
            return i + 15
        '\n        :type vals: List[int]\n        :type edges: List[List[int]]\n        :rtype: int\n        '
        edges.sort(key=lambda x: max(vals[x[0]], vals[x[1]]))
        uf = UnionFind(vals)
        return len(vals) + sum((uf.union_set(i, j, max(vals[i], vals[j])) for (i, j) in edges))