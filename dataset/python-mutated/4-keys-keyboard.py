class Solution(object):

    def maxA(self, N):
        if False:
            print('Hello World!')
        '\n        :type N: int\n        :rtype: int\n        '
        if N < 7:
            return N
        if N == 10:
            return 20
        n = N // 5 + 1
        n3 = 5 * n - N - 1
        n4 = n - n3
        return 3 ** n3 * 4 ** n4

class Solution2(object):

    def maxA(self, N):
        if False:
            return 10
        '\n        :type N: int\n        :rtype: int\n        '
        if N < 7:
            return N
        dp = range(N + 1)
        for i in xrange(7, N + 1):
            dp[i % 6] = max(dp[(i - 4) % 6] * 3, dp[(i - 5) % 6] * 4)
        return dp[N % 6]