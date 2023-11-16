class UnionFind(object):

    def __init__(self, n):
        if False:
            while True:
                i = 10
        self.set = range(n)

    def find_set(self, x):
        if False:
            i = 10
            return i + 15
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

    def swimInWater(self, grid):
        if False:
            return 10
        '\n        :type grid: List[List[int]]\n        :rtype: int\n        '
        n = len(grid)
        positions = [None] * n ** 2
        for i in xrange(n):
            for j in xrange(n):
                positions[grid[i][j]] = (i, j)
        directions = ((-1, 0), (1, 0), (0, -1), (0, 1))
        union_find = UnionFind(n ** 2)
        for elevation in xrange(n ** 2):
            (i, j) = positions[elevation]
            for direction in directions:
                (x, y) = (i + direction[0], j + direction[1])
                if 0 <= x < n and 0 <= y < n and (grid[x][y] <= elevation):
                    union_find.union_set(i * n + j, x * n + y)
                    if union_find.find_set(0) == union_find.find_set(n ** 2 - 1):
                        return elevation
        return n ** 2 - 1