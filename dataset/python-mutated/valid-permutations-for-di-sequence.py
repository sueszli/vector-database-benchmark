class Solution(object):

    def numPermsDISequence(self, S):
        if False:
            while True:
                i = 10
        '\n        :type S: str\n        :rtype: int\n        '
        dp = [1] * (len(S) + 1)
        for c in S:
            if c == 'I':
                dp = dp[:-1]
                for i in xrange(1, len(dp)):
                    dp[i] += dp[i - 1]
            else:
                dp = dp[1:]
                for i in reversed(xrange(len(dp) - 1)):
                    dp[i] += dp[i + 1]
        return dp[0] % (10 ** 9 + 7)