class Solution(object):
    _num = [0]

    def numSquares(self, n):
        if False:
            while True:
                i = 10
        '\n        :type n: int\n        :rtype: int\n        '
        num = self._num
        while len(num) <= n:
            num += (min((num[-i * i] for i in xrange(1, int(len(num) ** 0.5 + 1)))) + 1,)
        return num[n]