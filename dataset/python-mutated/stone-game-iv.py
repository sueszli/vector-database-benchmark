class Solution(object):

    def winnerSquareGame(self, n):
        if False:
            return 10
        '\n        :type n: int\n        :rtype: bool\n        '
        dp = [False] * (n + 1)
        for i in xrange(1, n + 1):
            j = 1
            while j * j <= i:
                if not dp[i - j * j]:
                    dp[i] = True
                    break
                j += 1
        return dp[-1]