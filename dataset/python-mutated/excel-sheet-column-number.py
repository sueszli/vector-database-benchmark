class Solution(object):

    def titleToNumber(self, s):
        if False:
            return 10
        '\n        :type s: str\n        :rtype: int\n        '
        result = 0
        for i in xrange(len(s)):
            result *= 26
            result += ord(s[i]) - ord('A') + 1
        return result