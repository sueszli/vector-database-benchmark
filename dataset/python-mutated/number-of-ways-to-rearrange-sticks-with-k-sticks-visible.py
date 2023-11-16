class Solution(object):

    def rearrangeSticks(self, n, k):
        if False:
            print('Hello World!')
        '\n        :type n: int\n        :type k: int\n        :rtype: int\n        '
        MOD = 10 ** 9 + 7
        dp = [[0 for _ in xrange(k + 1)] for _ in xrange(2)]
        dp[1][1] = 1
        for i in xrange(2, n + 1):
            for j in xrange(1, min(i, k) + 1):
                dp[i % 2][j] = (dp[(i - 1) % 2][j - 1] + (i - 1) * dp[(i - 1) % 2][j]) % MOD
        return dp[n % 2][k]