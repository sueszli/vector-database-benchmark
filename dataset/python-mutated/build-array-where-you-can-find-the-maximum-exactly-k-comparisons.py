class Solution(object):

    def numOfArrays(self, n, m, k):
        if False:
            print('Hello World!')
        '\n        :type n: int\n        :type m: int\n        :type k: int\n        :rtype: int\n        '
        MOD = 10 ** 9 + 7
        dp = [[[0] * (k + 1) for _ in xrange(m + 1)] for _ in xrange(2)]
        prefix_dp = [[[0] * (k + 1) for _ in xrange(m + 1)] for _ in xrange(2)]
        for i in xrange(1, m + 1):
            dp[1][i][1] = 1
            prefix_dp[1][i][1] = (prefix_dp[1][i - 1][1] + dp[1][i][1]) % MOD
        for l in xrange(2, n + 1):
            for i in xrange(1, m + 1):
                for j in xrange(1, k + 1):
                    dp[l % 2][i][j] = (i * dp[(l - 1) % 2][i][j] % MOD + prefix_dp[(l - 1) % 2][i - 1][j - 1]) % MOD
                    prefix_dp[l % 2][i][j] = (prefix_dp[l % 2][i - 1][j] + dp[l % 2][i][j]) % MOD
        return prefix_dp[n % 2][m][k]