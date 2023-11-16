import collections

class Solution(object):

    def distinctSequences(self, n):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type n: int\n        :rtype: int\n        '

        def gcd(a, b):
            if False:
                i = 10
                return i + 15
            while b:
                (a, b) = (b, a % b)
            return a
        if n == 1:
            return 6
        MOD = 10 ** 9 + 7
        dp = [[0] * 6 for _ in xrange(6)]
        for i in xrange(6):
            for j in xrange(6):
                if i != j and gcd(i + 1, j + 1) == 1:
                    dp[i][j] = 1
        for _ in xrange(n - 2):
            new_dp = [[0] * 6 for _ in xrange(6)]
            for i in xrange(6):
                for j in xrange(6):
                    if not dp[i][j]:
                        continue
                    for k in xrange(6):
                        if not dp[j][k]:
                            continue
                        if k != i:
                            new_dp[i][j] = (new_dp[i][j] + dp[j][k]) % MOD
            dp = new_dp
        return sum((dp[i][j] for i in xrange(6) for j in xrange(6))) % MOD