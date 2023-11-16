class UnionFind(object):

    def __init__(self, n):
        if False:
            return 10
        self.set = range(n)
        self.count = n

    def find_set(self, x):
        if False:
            for i in range(10):
                print('nop')
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

    def regionsBySlashes(self, grid):
        if False:
            while True:
                i = 10
        '\n        :type grid: List[str]\n        :rtype: int\n        '

        def index(n, i, j, k):
            if False:
                return 10
            return (i * n + j) * 4 + k
        union_find = UnionFind(len(grid) ** 2 * 4)
        (N, E, S, W) = range(4)
        for i in xrange(len(grid)):
            for j in xrange(len(grid)):
                if i:
                    union_find.union_set(index(len(grid), i - 1, j, S), index(len(grid), i, j, N))
                if j:
                    union_find.union_set(index(len(grid), i, j - 1, E), index(len(grid), i, j, W))
                if grid[i][j] != '/':
                    union_find.union_set(index(len(grid), i, j, N), index(len(grid), i, j, E))
                    union_find.union_set(index(len(grid), i, j, S), index(len(grid), i, j, W))
                if grid[i][j] != '\\':
                    union_find.union_set(index(len(grid), i, j, W), index(len(grid), i, j, N))
                    union_find.union_set(index(len(grid), i, j, E), index(len(grid), i, j, S))
        return union_find.count