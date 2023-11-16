class Solution(object):

    def minOperations(self, s1, s2, x):
        if False:
            return 10
        '\n        :type s1: str\n        :type s2: str\n        :type x: int\n        :rtype: int\n        '
        parity = curr = prev = 0
        j = -1
        for i in xrange(len(s1)):
            if s1[i] == s2[i]:
                continue
            (curr, prev) = (min(curr + x, prev + (i - j) * 2 if j != -1 else float('inf')), curr)
            j = i
            parity ^= 1
        return curr // 2 if parity == 0 else -1