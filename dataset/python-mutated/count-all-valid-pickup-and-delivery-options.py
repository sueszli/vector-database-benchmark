class Solution(object):

    def countOrders(self, n):
        if False:
            while True:
                i = 10
        '\n        :type n: int\n        :rtype: int\n        '
        MOD = 10 ** 9 + 7
        result = 1
        for i in reversed(xrange(2, 2 * n + 1, 2)):
            result = result * i * (i - 1) // 2 % MOD
        return result