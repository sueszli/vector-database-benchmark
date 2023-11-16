class Solution(object):

    def numberOfWays(self, s):
        if False:
            return 10
        '\n        :type s: str\n        :rtype: int\n        '
        K = 3
        dp = [[0] * 2 for _ in xrange(K)]
        for c in s:
            j = ord(c) - ord('0')
            dp[0][j] += 1
            for i in xrange(1, len(dp)):
                dp[i][j] += dp[i - 1][1 ^ j]
        return dp[-1][0] + dp[-1][1]