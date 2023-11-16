class Solution(object):

    def lengthOfLongestSubstring(self, s):
        if False:
            i = 10
            return i + 15
        '\n        :type s: str\n        :rtype: int\n        '
        (result, left) = (0, 0)
        lookup = {}
        for right in xrange(len(s)):
            if s[right] in lookup:
                left = max(left, lookup[s[right]] + 1)
            lookup[s[right]] = right
            result = max(result, right - left + 1)
        return result