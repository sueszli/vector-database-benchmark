class Solution(object):

    def constructProductMatrix(self, grid):
        if False:
            print('Hello World!')
        '\n        :type grid: List[List[int]]\n        :rtype: List[List[int]]\n        '
        MOD = 12345
        right = [1] * (len(grid) * len(grid[0]) + 1)
        for i in reversed(xrange(len(grid))):
            for j in reversed(xrange(len(grid[0]))):
                right[i * len(grid[0]) + j] = right[i * len(grid[0]) + j + 1] * grid[i][j] % MOD
        left = 1
        for i in xrange(len(grid)):
            for j in xrange(len(grid[0])):
                (grid[i][j], left) = (left * right[i * len(grid[0]) + j + 1] % MOD, left * grid[i][j] % MOD)
        return grid

class Solution2(object):

    def constructProductMatrix(self, grid):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type grid: List[List[int]]\n        :rtype: List[List[int]]\n        '
        MOD = 12345
        left = [1] * (len(grid) * len(grid[0]) + 1)
        for i in xrange(len(grid)):
            for j in xrange(len(grid[0])):
                left[i * len(grid[0]) + j + 1] = left[i * len(grid[0]) + j] * grid[i][j] % MOD
        right = [1] * (len(grid) * len(grid[0]) + 1)
        for i in reversed(xrange(len(grid))):
            for j in reversed(xrange(len(grid[0]))):
                right[i * len(grid[0]) + j] = right[i * len(grid[0]) + j + 1] * grid[i][j] % MOD
        for i in xrange(len(grid)):
            for j in xrange(len(grid[0])):
                grid[i][j] = left[i * len(grid[0]) + j] * right[i * len(grid[0]) + j + 1] % MOD
        return grid