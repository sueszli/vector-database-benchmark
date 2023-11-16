class Solution(object):

    def numberOfWays(self, num_people):
        if False:
            print('Hello World!')
        '\n        :type num_people: int\n        :rtype: int\n        '
        MOD = 10 ** 9 + 7

        def inv(x, m):
            if False:
                while True:
                    i = 10
            return pow(x, m - 2, m)

        def nCr(n, k, m):
            if False:
                print('Hello World!')
            if n - k < k:
                return nCr(n, n - k, m)
            result = 1
            for i in xrange(1, k + 1):
                result = result * (n - k + i) * inv(i, m) % m
            return result
        n = num_people // 2
        return nCr(2 * n, n, MOD) * inv(n + 1, MOD) % MOD

class Solution2(object):

    def numberOfWays(self, num_people):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type num_people: int\n        :rtype: int\n        '
        MOD = 10 ** 9 + 7
        dp = [0] * (num_people // 2 + 1)
        dp[0] = 1
        for k in xrange(1, num_people // 2 + 1):
            for i in xrange(k):
                dp[k] = (dp[k] + dp[i] * dp[k - 1 - i]) % MOD
        return dp[num_people // 2]