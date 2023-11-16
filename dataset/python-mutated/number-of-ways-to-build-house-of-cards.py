class Solution(object):

    def houseOfCards(self, n):
        if False:
            i = 10
            return i + 15
        '\n        :type n: int\n        :rtype: int\n        '
        dp = [0] * (n + 1)
        dp[0] = 1
        for t in xrange(1, (n + 1) // 3 + 1):
            for i in reversed(xrange(3 * t - 1, n + 1)):
                dp[i] += dp[i - (3 * t - 1)]
        return dp[n]

class Solution_TLE(object):

    def houseOfCards(self, n):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type n: int\n        :rtype: int\n        '
        dp = [[0] * (n + 1) for _ in xrange((n + 1) // 3 + 1)]
        dp[0][0] = 1
        for t in xrange(1, (n + 1) // 3 + 1):
            for i in xrange(3 * t - 1, n + 1):
                dp[t][i] = sum((dp[j][i - (3 * t - 1)] for j in xrange(t)))
        return sum((dp[t][n] for t in xrange((n + 1) // 3 + 1)))