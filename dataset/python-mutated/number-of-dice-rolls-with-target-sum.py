class Solution(object):

    def numRollsToTarget(self, d, f, target):
        if False:
            i = 10
            return i + 15
        '\n        :type d: int\n        :type f: int\n        :type target: int\n        :rtype: int\n        '
        MOD = 10 ** 9 + 7
        dp = [[0 for _ in xrange(target + 1)] for _ in xrange(2)]
        dp[0][0] = 1
        for i in xrange(1, d + 1):
            dp[i % 2] = [0 for _ in xrange(target + 1)]
            for k in xrange(1, f + 1):
                for j in xrange(k, target + 1):
                    dp[i % 2][j] = (dp[i % 2][j] + dp[(i - 1) % 2][j - k]) % MOD
        return dp[d % 2][target] % MOD