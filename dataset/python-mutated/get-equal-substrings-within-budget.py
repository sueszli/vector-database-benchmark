class Solution(object):

    def equalSubstring(self, s, t, maxCost):
        if False:
            i = 10
            return i + 15
        '\n        :type s: str\n        :type t: str\n        :type maxCost: int\n        :rtype: int\n        '
        left = 0
        for right in xrange(len(s)):
            maxCost -= abs(ord(s[right]) - ord(t[right]))
            if maxCost < 0:
                maxCost += abs(ord(s[left]) - ord(t[left]))
                left += 1
        return right + 1 - left