class Solution(object):

    def count(self, num1, num2, min_sum, max_sum):
        if False:
            return 10
        '\n        :type num1: str\n        :type num2: str\n        :type min_sum: int\n        :type max_sum: int\n        :rtype: int\n        '
        MOD = 10 ** 9 + 7

        def f(x):
            if False:
                for i in range(10):
                    print('nop')
            dp = [[0] * (max_sum + 1) for _ in xrange(2)]
            dp[0][0] = dp[1][0] = 1
            for i in reversed(xrange(len(x))):
                new_dp = [[0] * (max_sum + 1) for _ in xrange(2)]
                for t in xrange(2):
                    for total in xrange(max_sum + 1):
                        for d in xrange(min(int(x[i]) if t else 9, total) + 1):
                            new_dp[t][total] = (new_dp[t][total] + dp[int(t and d == int(x[i]))][total - d]) % MOD
                dp = new_dp
            return reduce(lambda x, y: (x + y) % MOD, (dp[1][total] for total in xrange(min_sum, max_sum + 1)))
        return (f(num2) - f(str(int(num1) - 1))) % MOD