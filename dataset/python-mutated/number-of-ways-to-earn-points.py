class Solution(object):

    def waysToReachTarget(self, target, types):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type target: int\n        :type types: List[List[int]]\n        :rtype: int\n        '
        MOD = 10 ** 9 + 7
        dp = [0] * (target + 1)
        dp[0] = 1
        for (c, m) in types:
            for i in reversed(xrange(1, target + 1)):
                for j in xrange(1, min(i // m, c) + 1):
                    dp[i] = (dp[i] + dp[i - j * m]) % MOD
        return dp[-1]

class Solution2(object):

    def waysToReachTarget(self, target, types):
        if False:
            print('Hello World!')
        '\n        :type target: int\n        :type types: List[List[int]]\n        :rtype: int\n        '
        MOD = 10 ** 9 + 7
        dp = [0] * (target + 1)
        dp[0] = 1
        for (c, m) in types:
            new_dp = [0] * (target + 1)
            for i in xrange(target + 1):
                for j in xrange(min((target - i) // m, c) + 1):
                    new_dp[i + j * m] = (new_dp[i + j * m] + dp[i]) % MOD
            dp = new_dp
        return dp[-1]