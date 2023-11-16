import math

class Solution(object):

    def smallestGoodBase(self, n):
        if False:
            print('Hello World!')
        '\n        :type n: str\n        :rtype: str\n        '
        num = int(n)
        max_len = int(math.log(num, 2))
        for l in xrange(max_len, 1, -1):
            b = int(num ** l ** (-1))
            if (b ** (l + 1) - 1) // (b - 1) == num:
                return str(b)
        return str(num - 1)