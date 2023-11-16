class Solution(object):

    def minimumCost(self, s):
        if False:
            print('Hello World!')
        '\n        :type s: str\n        :rtype: int\n        '
        return sum((min(i + 1, len(s) - (i + 1)) for i in xrange(len(s) - 1) if s[i] != s[i + 1]))