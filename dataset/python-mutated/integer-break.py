class Solution(object):

    def integerBreak(self, n):
        if False:
            print('Hello World!')
        '\n        :type n: int\n        :rtype: int\n        '
        if n < 4:
            return n - 1
        res = 0
        if n % 3 == 0:
            res = 3 ** (n // 3)
        elif n % 3 == 2:
            res = 3 ** (n // 3) * 2
        else:
            res = 3 ** (n // 3 - 1) * 4
        return res

class Solution2(object):

    def integerBreak(self, n):
        if False:
            i = 10
            return i + 15
        '\n        :type n: int\n        :rtype: int\n        '
        if n < 4:
            return n - 1
        res = [0, 1, 2, 3]
        for i in xrange(4, n + 1):
            res[i % 4] = max(res[(i - 2) % 4] * 2, res[(i - 3) % 4] * 3)
        return res[n % 4]