class Solution(object):

    def minPathCost(self, grid, moveCost):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type grid: List[List[int]]\n        :type moveCost: List[List[int]]\n        :rtype: int\n        '
        dp = [[0] * len(grid[0]) for _ in xrange(2)]
        dp[0] = [grid[0][j] for j in xrange(len(grid[0]))]
        for i in xrange(len(grid) - 1):
            for j in xrange(len(grid[0])):
                dp[(i + 1) % 2][j] = min((dp[i % 2][k] + moveCost[x][j] for (k, x) in enumerate(grid[i]))) + grid[i + 1][j]
        return min(dp[(len(grid) - 1) % 2])