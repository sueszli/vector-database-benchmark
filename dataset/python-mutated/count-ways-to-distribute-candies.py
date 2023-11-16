class Solution(object):

    def waysToDistribute(self, n, k):
        if False:
            return 10
        '\n        :type n: int\n        :type k: int\n        :rtype: int\n        '
        MOD = 10 ** 9 + 7
        dp = [1] * k
        for i in xrange(1, n):
            for j in reversed(xrange(1, min(i, k))):
                dp[j] = ((j + 1) * dp[j] + dp[j - 1]) % MOD
        return dp[k - 1]