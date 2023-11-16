class Solution(object):

    def reverseBits(self, n):
        if False:
            for i in range(10):
                print('nop')
        n = n >> 16 | n << 16
        n = (n & 4278255360) >> 8 | (n & 16711935) << 8
        n = (n & 4042322160) >> 4 | (n & 252645135) << 4
        n = (n & 3435973836) >> 2 | (n & 858993459) << 2
        n = (n & 2863311530) >> 1 | (n & 1431655765) << 1
        return n

class Solution2(object):

    def reverseBits(self, n):
        if False:
            i = 10
            return i + 15
        result = 0
        for i in xrange(32):
            result <<= 1
            result |= n & 1
            n >>= 1
        return result