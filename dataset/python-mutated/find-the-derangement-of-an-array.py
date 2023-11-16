class Solution(object):

    def findDerangement(self, n):
        if False:
            while True:
                i = 10
        '\n        :type n: int\n        :rtype: int\n        '
        M = 1000000007
        (mul, total) = (1, 0)
        for i in reversed(xrange(n + 1)):
            total = (total + M + (1 if i % 2 == 0 else -1) * mul) % M
            mul = mul * i % M
        return total