class Solution(object):

    def getSum(self, a, b):
        if False:
            return 10
        '\n        :type a: int\n        :type b: int\n        :rtype: int\n        '
        bit_length = 32
        (neg_bit, mask) = (1 << bit_length >> 1, ~(~0 << bit_length))
        a = a | ~mask if a & neg_bit else a & mask
        b = b | ~mask if b & neg_bit else b & mask
        while b:
            carry = a & b
            a ^= b
            a = a | ~mask if a & neg_bit else a & mask
            b = carry << 1
            b = b | ~mask if b & neg_bit else b & mask
        return a

    def getSum2(self, a, b):
        if False:
            return 10
        '\n        :type a: int\n        :type b: int\n        :rtype: int\n        '
        MAX = 2147483647
        MIN = 2147483648
        mask = 4294967295
        while b:
            (a, b) = ((a ^ b) & mask, (a & b) << 1 & mask)
        return a if a <= MAX else ~(a ^ mask)

    def minus(self, a, b):
        if False:
            return 10
        b = self.getSum(~b, 1)
        return self.getSum(a, b)

    def multiply(self, a, b):
        if False:
            return 10
        isNeg = (a > 0) ^ (b > 0)
        x = a if a > 0 else self.getSum(~a, 1)
        y = b if b > 0 else self.getSum(~b, 1)
        ans = 0
        while y & 1:
            ans = self.getSum(ans, x)
            y >>= 1
            x <<= 1
        return self.getSum(~ans, 1) if isNeg else ans

    def divide(self, a, b):
        if False:
            print('Hello World!')
        isNeg = (a > 0) ^ (b > 0)
        x = a if a > 0 else self.getSum(~a, 1)
        y = b if b > 0 else self.getSum(~b, 1)
        ans = 0
        for i in range(31, -1, -1):
            if x >> i >= y:
                x = self.minus(x, y << i)
                ans = self.getSum(ans, 1 << i)
        return self.getSum(~ans, 1) if isNeg else ans