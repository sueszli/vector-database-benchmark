class Solution(object):

    def peopleAwareOfSecret(self, n, delay, forget):
        if False:
            i = 10
            return i + 15
        '\n        :type n: int\n        :type delay: int\n        :type forget: int\n        :rtype: int\n        '
        MOD = 10 ** 9 + 7
        dp = [0] * forget
        dp[0] = 1
        for i in xrange(1, n):
            dp[i % forget] = ((dp[(i - 1) % forget] if i - 1 else 0) - dp[i % forget] + dp[(i - delay) % forget]) % MOD
        return sum(dp) % MOD