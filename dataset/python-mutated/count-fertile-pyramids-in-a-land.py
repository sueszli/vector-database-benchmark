class Solution(object):

    def countPyramids(self, grid):
        if False:
            while True:
                i = 10
        '\n        :type grid: List[List[int]]\n        :rtype: int\n        '

        def count(grid, reverse):
            if False:
                i = 10
                return i + 15

            def get_grid(i, j):
                if False:
                    return 10
                return grid[~i][j] if reverse else grid[i][j]
            result = 0
            dp = [0] * len(grid[0])
            for i in xrange(1, len(grid)):
                new_dp = [0] * len(grid[0])
                for j in xrange(1, len(grid[0]) - 1):
                    if get_grid(i, j) == get_grid(i - 1, j - 1) == get_grid(i - 1, j) == get_grid(i - 1, j + 1) == 1:
                        new_dp[j] = min(dp[j - 1], dp[j + 1]) + 1
                dp = new_dp
                result += sum(dp)
            return result
        return count(grid, False) + count(grid, True)

class Solution2(object):

    def countPyramids(self, grid):
        if False:
            while True:
                i = 10
        '\n        :type grid: List[List[int]]\n        :rtype: int\n        '

        def count(grid):
            if False:
                i = 10
                return i + 15
            dp = [[0 for _ in xrange(len(grid[0]))] for _ in xrange(len(grid))]
            for i in xrange(1, len(grid)):
                for j in xrange(1, len(grid[0]) - 1):
                    if grid[i][j] == grid[i - 1][j - 1] == grid[i - 1][j] == grid[i - 1][j + 1] == 1:
                        dp[i][j] = min(dp[i - 1][j - 1], dp[i - 1][j], dp[i - 1][j + 1]) + 1
            return sum((sum(row) for row in dp))
        return count(grid) + count(grid[::-1])