class Solution(object):

    def longestPalindrome(self, word1, word2):
        if False:
            i = 10
            return i + 15
        '\n        :type word1: str\n        :type word2: str\n        :rtype: int\n        '
        s = word1 + word2
        dp = [[0] * len(s) for _ in xrange(len(s))]
        result = 0
        for j in xrange(len(s)):
            dp[j][j] = 1
            for i in reversed(xrange(j)):
                if s[i] == s[j]:
                    dp[i][j] = 2 if i + 1 == j else dp[i + 1][j - 1] + 2
                    if i < len(word1) <= j:
                        result = max(result, dp[i][j])
                else:
                    dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])
        return result

class Solution2(object):

    def longestPalindrome(self, word1, word2):
        if False:
            return 10
        '\n        :type word1: str\n        :type word2: str\n        :rtype: int\n        '
        s = word1 + word2
        dp = [[0] * len(s) for _ in xrange(len(s))]
        for j in xrange(len(s)):
            dp[j][j] = 1
            for i in reversed(xrange(j)):
                if s[i] == s[j]:
                    dp[i][j] = 2 if i + 1 == j else dp[i + 1][j - 1] + 2
                else:
                    dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])
        return max([dp[i][j] for i in xrange(len(word1)) for j in xrange(len(word1), len(s)) if s[i] == s[j]] or [0])