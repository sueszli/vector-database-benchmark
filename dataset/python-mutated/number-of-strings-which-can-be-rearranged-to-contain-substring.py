class Solution(object):

    def stringCount(self, n):
        if False:
            print('Hello World!')
        '\n        :type n: int\n        :rtype: int\n        '
        MOD = 10 ** 9 + 7
        return (pow(26, n, MOD) - (25 + 25 + 25 + n) * pow(25, n - 1, MOD) + (24 + 24 + 24 + n + n + 0) * pow(24, n - 1, MOD) - (23 + n + 0 + 0) * pow(23, n - 1, MOD)) % MOD

class Solution2(object):

    def stringCount(self, n):
        if False:
            i = 10
            return i + 15
        '\n        :type n: int\n        :rtype: int\n        '
        MOD = 10 ** 9 + 7
        (L, E, EE, T) = [1 << i for i in xrange(4)]
        dp = [0] * (1 << 4)
        dp[0] = 1
        for _ in xrange(n):
            new_dp = [0] * (1 << 4)
            for mask in xrange(len(dp)):
                new_dp[mask | L] = (new_dp[mask | L] + dp[mask]) % MOD
                if not mask & E:
                    new_dp[mask | E] = (new_dp[mask | E] + dp[mask]) % MOD
                else:
                    new_dp[mask | EE] = (new_dp[mask | EE] + dp[mask]) % MOD
                new_dp[mask | T] = (new_dp[mask | T] + dp[mask]) % MOD
                new_dp[mask] = (new_dp[mask] + 23 * dp[mask]) % MOD
            dp = new_dp
        return dp[-1]