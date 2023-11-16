class Solution(object):

    def numWays(self, steps, arrLen):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type steps: int\n        :type arrLen: int\n        :rtype: int\n        '
        MOD = int(1000000000.0 + 7)
        l = min(1 + steps // 2, arrLen)
        dp = [0] * (l + 2)
        dp[1] = 1
        while steps > 0:
            steps -= 1
            new_dp = [0] * (l + 2)
            for i in xrange(1, l + 1):
                new_dp[i] = (dp[i] + dp[i - 1] + dp[i + 1]) % MOD
            dp = new_dp
        return dp[1]