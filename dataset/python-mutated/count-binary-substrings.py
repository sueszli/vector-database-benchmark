class Solution(object):

    def countBinarySubstrings(self, s):
        if False:
            return 10
        '\n        :type s: str\n        :rtype: int\n        '
        (result, prev, curr) = (0, 0, 1)
        for i in xrange(1, len(s)):
            if s[i - 1] != s[i]:
                result += min(prev, curr)
                (prev, curr) = (curr, 1)
            else:
                curr += 1
        result += min(prev, curr)
        return result