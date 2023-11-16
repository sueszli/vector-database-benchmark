class Solution(object):

    def numberOfPaths(self, grid, k):
        if False:
            print('Hello World!')
        '\n        :type grid: List[List[int]]\n        :type k: int\n        :rtype: int\n        '
        MOD = 10 ** 9 + 7
        dp = [[0 for _ in xrange(k)] for _ in xrange(len(grid[0]))]
        dp[0][0] = 1
        for i in xrange(len(grid)):
            for j in xrange(len(grid[0])):
                dp[j] = [((dp[j - 1][(l - grid[i][j]) % k] if j - 1 >= 0 else 0) + dp[j][(l - grid[i][j]) % k]) % MOD for l in xrange(k)]
        return dp[-1][0]