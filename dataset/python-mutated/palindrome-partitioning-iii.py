class Solution(object):

    def palindromePartition(self, s, k):
        if False:
            i = 10
            return i + 15
        '\n        :type s: str\n        :type k: int\n        :rtype: int\n        '
        dp1 = [[0] * len(s) for _ in xrange(len(s))]
        for l in xrange(1, len(s) + 1):
            for i in xrange(len(s) - l + 1):
                j = i + l - 1
                if i == j - 1:
                    dp1[i][j] = 0 if s[i] == s[j] else 1
                elif i != j:
                    dp1[i][j] = dp1[i + 1][j - 1] if s[i] == s[j] else dp1[i + 1][j - 1] + 1
        dp2 = [[float('inf')] * len(s) for _ in xrange(2)]
        dp2[1] = dp1[0][:]
        for d in xrange(2, k + 1):
            dp2[d % 2] = [float('inf')] * len(s)
            for i in xrange(d - 1, len(s)):
                for j in xrange(d - 2, i):
                    dp2[d % 2][i] = min(dp2[d % 2][i], dp2[(d - 1) % 2][j] + dp1[j + 1][i])
        return dp2[k % 2][len(s) - 1]