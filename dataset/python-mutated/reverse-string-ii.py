class Solution(object):

    def reverseStr(self, s, k):
        if False:
            i = 10
            return i + 15
        '\n        :type s: str\n        :type k: int\n        :rtype: str\n        '
        s = list(s)
        for i in xrange(0, len(s), 2 * k):
            s[i:i + k] = reversed(s[i:i + k])
        return ''.join(s)