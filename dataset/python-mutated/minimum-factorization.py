class Solution(object):

    def smallestFactorization(self, a):
        if False:
            while True:
                i = 10
        '\n        :type a: int\n        :rtype: int\n        '
        if a < 2:
            return a
        (result, mul) = (0, 1)
        for i in reversed(xrange(2, 10)):
            while a % i == 0:
                a /= i
                result = mul * i + result
                mul *= 10
        return result if a == 1 and result < 2 ** 31 else 0