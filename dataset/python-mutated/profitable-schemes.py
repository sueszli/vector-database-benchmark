import itertools

class Solution(object):

    def profitableSchemes(self, G, P, group, profit):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type G: int\n        :type P: int\n        :type group: List[int]\n        :type profit: List[int]\n        :rtype: int\n        '
        dp = [[0 for _ in xrange(G + 1)] for _ in xrange(P + 1)]
        dp[0][0] = 1
        for (p, g) in itertools.izip(profit, group):
            for i in reversed(xrange(P + 1)):
                for j in reversed(xrange(G - g + 1)):
                    dp[min(i + p, P)][j + g] += dp[i][j]
        return sum(dp[P]) % (10 ** 9 + 7)