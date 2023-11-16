class Solution(object):

    def minChanges(self, s):
        if False:
            return 10
        '\n        :type s: str\n        :rtype: int\n        '
        return sum((s[i] != s[i + 1] for i in xrange(0, len(s), 2)))