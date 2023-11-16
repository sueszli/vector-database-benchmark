class Solution(object):

    def maxSum(self, grid):
        if False:
            print('Hello World!')
        '\n        :type grid: List[List[int]]\n        :rtype: int\n        '

        def total(i, j):
            if False:
                while True:
                    i = 10
            return grid[i][j] + grid[i][j + 1] + grid[i][j + 2] + grid[i + 1][j + 1] + grid[i + 2][j] + grid[i + 2][j + 1] + grid[i + 2][j + 2]
        return max((total(i, j) for i in xrange(len(grid) - 2) for j in xrange(len(grid[0]) - 2)))