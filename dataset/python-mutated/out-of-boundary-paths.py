class Solution(object):

    def findPaths(self, m, n, N, x, y):
        if False:
            i = 10
            return i + 15
        '\n        :type m: int\n        :type n: int\n        :type N: int\n        :type x: int\n        :type y: int\n        :rtype: int\n        '
        M = 1000000000 + 7
        dp = [[[0 for _ in xrange(n)] for _ in xrange(m)] for _ in xrange(2)]
        for moves in xrange(N):
            for i in xrange(m):
                for j in xrange(n):
                    dp[(moves + 1) % 2][i][j] = (((1 if i == 0 else dp[moves % 2][i - 1][j]) + (1 if i == m - 1 else dp[moves % 2][i + 1][j])) % M + ((1 if j == 0 else dp[moves % 2][i][j - 1]) + (1 if j == n - 1 else dp[moves % 2][i][j + 1])) % M) % M
        return dp[N % 2][x][y]