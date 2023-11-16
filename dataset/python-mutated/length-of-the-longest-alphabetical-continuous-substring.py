class Solution(object):

    def longestContinuousSubstring(self, s):
        if False:
            i = 10
            return i + 15
        '\n        :type s: str\n        :rtype: int\n        '
        result = l = 0
        for i in xrange(len(s)):
            l += 1
            if i + 1 == len(s) or ord(s[i]) + 1 != ord(s[i + 1]):
                result = max(result, l)
                l = 0
        return result