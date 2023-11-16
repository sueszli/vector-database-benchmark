import collections

class Solution(object):

    def numWays(self, words, target):
        if False:
            i = 10
            return i + 15
        '\n        :type words: List[str]\n        :type target: str\n        :rtype: int\n        '
        MOD = 10 ** 9 + 7
        dp = [0] * (len(target) + 1)
        dp[0] = 1
        for i in xrange(len(words[0])):
            count = collections.Counter((w[i] for w in words))
            for j in reversed(xrange(len(target))):
                dp[j + 1] += dp[j] * count[target[j]] % MOD
        return dp[-1] % MOD
import collections

class Solution2(object):

    def numWays(self, words, target):
        if False:
            return 10
        '\n        :type words: List[str]\n        :type target: str\n        :rtype: int\n        '
        MOD = 10 ** 9 + 7
        dp = [[0] * (len(target) + 1) for _ in xrange(2)]
        for i in xrange(len(dp)):
            dp[i][0] = 1
        for i in xrange(len(words[0])):
            count = collections.Counter((w[i] for w in words))
            for j in reversed(xrange(len(target))):
                dp[(i + 1) % 2][j + 1] = dp[i % 2][j + 1] + dp[i % 2][j] * count[target[j]] % MOD
        return dp[len(words[0]) % 2][-1] % MOD