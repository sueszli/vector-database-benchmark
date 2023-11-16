class UnionFind(object):

    def __init__(self, n):
        if False:
            while True:
                i = 10
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
            i = 10
            return i + 15
        (x_root, y_root) = map(self.find_set, (x, y))
        if x_root == y_root:
            return False
        self.set[min(x_root, y_root)] = max(x_root, y_root)
        return True

class Solution(object):

    def removeStones(self, stones):
        if False:
            i = 10
            return i + 15
        '\n        :type stones: List[List[int]]\n        :rtype: int\n        '
        MAX_ROW = 10000
        union_find = UnionFind(2 * MAX_ROW)
        for (r, c) in stones:
            union_find.union_set(r, c + MAX_ROW)
        return len(stones) - len({union_find.find_set(r) for (r, _) in stones})