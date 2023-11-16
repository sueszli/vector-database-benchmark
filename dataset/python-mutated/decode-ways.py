class Solution(object):

    def numDecodings(self, s):
        if False:
            return 10
        '\n        :type s: str\n        :rtype: int\n        '
        if len(s) == 0 or s[0] == '0':
            return 0
        (prev, prev_prev) = (1, 0)
        for i in xrange(len(s)):
            cur = 0
            if s[i] != '0':
                cur = prev
            if i > 0 and (s[i - 1] == '1' or (s[i - 1] == '2' and s[i] <= '6')):
                cur += prev_prev
            (prev, prev_prev) = (cur, prev)
        return prev