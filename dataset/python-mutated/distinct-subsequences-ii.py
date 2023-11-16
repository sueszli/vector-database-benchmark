import collections

class Solution(object):

    def distinctSubseqII(self, S):
        if False:
            i = 10
            return i + 15
        '\n        :type S: str\n        :rtype: int\n        '
        MOD = 10 ** 9 + 7
        (result, dp) = (0, [0] * 26)
        for c in S:
            (result, dp[ord(c) - ord('a')]) = ((result + (result + 1 - dp[ord(c) - ord('a')])) % MOD, (result + 1) % MOD)
        return result