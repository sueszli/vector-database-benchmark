class Solution(object):

    def longestIdealString(self, s, k):
        if False:
            return 10
        '\n        :type s: str\n        :type k: int\n        :rtype: int\n        '
        dp = [0] * 26
        for c in s:
            x = ord(c) - ord('a')
            dp[x] = max((dp[i] for i in xrange(max(x - k, 0), min(x + k + 1, 26)))) + 1
        return max(dp)