class Solution(object):

    def getMoneyAmount(self, n):
        if False:
            print('Hello World!')
        '\n        :type n: int\n        :rtype: int\n        '
        dp = [[0] * (n + 1) for _ in xrange(n + 1)]
        for j in xrange(n + 1):
            for i in reversed(xrange(j - 1)):
                dp[i][j] = min((k + 1 + max(dp[i][k], dp[k + 1][j]) for k in xrange(i, j)))
        return dp[0][n]

class Solution2(object):

    def getMoneyAmount(self, n):
        if False:
            print('Hello World!')
        '\n        :type n: int\n        :rtype: int\n        '
        dp = [[0] * (n + 1) for _ in xrange(n + 1)]
        for i in reversed(xrange(n)):
            for j in xrange(i + 2, n + 1):
                dp[i][j] = min((k + 1 + max(dp[i][k], dp[k + 1][j]) for k in xrange(i, j)))
        return dp[0][n]