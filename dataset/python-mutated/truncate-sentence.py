class Solution(object):

    def truncateSentence(self, s, k):
        if False:
            return 10
        '\n        :type s: str\n        :type k: int\n        :rtype: str\n        '
        for i in xrange(len(s)):
            if s[i] == ' ':
                k -= 1
                if not k:
                    return s[:i]
        return s