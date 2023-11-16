class Solution(object):

    def longestSemiRepetitiveSubstring(self, s):
        if False:
            print('Hello World!')
        '\n        :type s: str\n        :rtype: int\n        '
        result = left = prev = 0
        for right in xrange(len(s)):
            if right - 1 >= 0 and s[right - 1] == s[right]:
                (left, prev) = (prev, right)
            result = max(result, right - left + 1)
        return result