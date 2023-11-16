import math

class Solution(object):

    def twoEggDrop(self, n):
        if False:
            while True:
                i = 10
        '\n        :type n: int\n        :rtype: int\n        '
        return int(math.ceil((-1 + (1 + 8 * n) ** 0.5) / 2))

class Solution2(object):

    def twoEggDrop(self, n):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type n: int\n        :rtype: int\n        '
        K = 2
        dp = [[float('inf') for j in xrange(n + 1)] for _ in xrange(2)]
        dp[1] = [j for j in xrange(n + 1)]
        for i in xrange(2, K + 1):
            dp[i % 2][0] = 0
            for j in xrange(1, n + 1):
                for k in xrange(1, j + 1):
                    dp[i % 2][j] = min(dp[i % 2][j], 1 + max(dp[(i - 1) % 2][k - 1], dp[i % 2][j - k]))
        return dp[K % 2][n]