class Solution(object):

    def countGoodStrings(self, low, high, zero, one):
        if False:
            print('Hello World!')
        '\n        :type low: int\n        :type high: int\n        :type zero: int\n        :type one: int\n        :rtype: int\n        '
        MOD = 10 ** 9 + 7
        result = 0
        dp = [0] * (high + 1)
        dp[0] = 1
        for i in xrange(1, high + 1):
            if i >= zero:
                dp[i] = (dp[i] + dp[i - zero]) % MOD
            if i >= one:
                dp[i] = (dp[i] + dp[i - one]) % MOD
            if i >= low:
                result = (result + dp[i]) % MOD
        return result