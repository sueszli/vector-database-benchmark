class Solution(object):

    def numberOfSpecialSubstrings(self, s):
        if False:
            i = 10
            return i + 15
        '\n        :type s: str\n        :rtype: int\n        '
        result = left = 0
        lookup = [-1] * 26
        for right in xrange(len(s)):
            if lookup[ord(s[right]) - ord('a')] >= left:
                left = lookup[ord(s[right]) - ord('a')] + 1
            lookup[ord(s[right]) - ord('a')] = right
            result += right - left + 1
        return result

class Solution2(object):

    def numberOfSpecialSubstrings(self, s):
        if False:
            i = 10
            return i + 15
        '\n        :type s: str\n        :rtype: int\n        '
        result = left = 0
        lookup = [False] * 26
        for right in xrange(len(s)):
            while lookup[ord(s[right]) - ord('a')]:
                lookup[ord(s[left]) - ord('a')] = False
                left += 1
            lookup[ord(s[right]) - ord('a')] = True
            result += right - left + 1
        return result