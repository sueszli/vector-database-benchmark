class Solution(object):

    def myPow(self, x, n):
        if False:
            while True:
                i = 10
        '\n        :type x: float\n        :type n: int\n        :rtype: float\n        '
        result = 1
        abs_n = abs(n)
        while abs_n:
            if abs_n & 1:
                result *= x
            abs_n >>= 1
            x *= x
        return 1 / result if n < 0 else result

class Solution2(object):

    def myPow(self, x, n):
        if False:
            return 10
        '\n        :type x: float\n        :type n: int\n        :rtype: float\n        '
        if n < 0 and n != -n:
            return 1.0 / self.myPow(x, -n)
        if n == 0:
            return 1
        v = self.myPow(x, n / 2)
        if n % 2 == 0:
            return v * v
        else:
            return v * v * x