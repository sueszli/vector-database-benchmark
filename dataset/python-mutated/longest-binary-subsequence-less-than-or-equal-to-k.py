class Solution(object):

    def longestSubsequence(self, s, k):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type s: str\n        :type k: int\n        :rtype: int\n        '
        (result, base) = (0, 1)
        for i in reversed(xrange(len(s))):
            if s[i] == '0':
                result += 1
            elif base <= k:
                k -= base
                result += 1
            if base <= k:
                base <<= 1
        return result