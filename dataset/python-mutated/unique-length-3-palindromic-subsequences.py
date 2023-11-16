class Solution(object):

    def countPalindromicSubsequence(self, s):
        if False:
            return 10
        '\n        :type s: str\n        :rtype: int\n        '
        (first, last) = ([len(s)] * 26, [-1] * 26)
        for (i, c) in enumerate(s):
            first[ord(c) - ord('a')] = min(first[ord(c) - ord('a')], i)
            last[ord(c) - ord('a')] = max(last[ord(c) - ord('a')], i)
        return sum((len(set((s[i] for i in xrange(first[c] + 1, last[c])))) for c in xrange(26)))