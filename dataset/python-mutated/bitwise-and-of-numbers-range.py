class Solution(object):

    def rangeBitwiseAnd(self, m, n):
        if False:
            return 10
        while m < n:
            n &= n - 1
        return n

class Solution2(object):

    def rangeBitwiseAnd(self, m, n):
        if False:
            i = 10
            return i + 15
        (i, diff) = (0, n - m)
        while diff:
            diff >>= 1
            i += 1
        return n & m >> i << i