class Solution(object):

    def decodeAtIndex(self, S, K):
        if False:
            while True:
                i = 10
        '\n        :type S: str\n        :type K: int\n        :rtype: str\n        '
        i = 0
        for c in S:
            if c.isdigit():
                i *= int(c)
            else:
                i += 1
        for c in reversed(S):
            K %= i
            if K == 0 and c.isalpha():
                return c
            if c.isdigit():
                i /= int(c)
            else:
                i -= 1