class UnionFind(object):

    def __init__(self, n):
        if False:
            while True:
                i = 10
        self.set = range(n)
        self.count = n

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
        self.set[max(x_root, y_root)] = min(x_root, y_root)
        self.count -= 1
        return True

class Solution(object):

    def minCostToSupplyWater(self, n, wells, pipes):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type n: int\n        :type wells: List[int]\n        :type pipes: List[List[int]]\n        :rtype: int\n        '
        w = [[c, 0, i] for (i, c) in enumerate(wells, 1)]
        p = [[c, i, j] for (i, j, c) in pipes]
        result = 0
        union_find = UnionFind(n + 1)
        for (c, x, y) in sorted(w + p):
            if not union_find.union_set(x, y):
                continue
            result += c
            if union_find.count == 1:
                break
        return result