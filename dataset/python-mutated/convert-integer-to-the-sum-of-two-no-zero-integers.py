class Solution(object):

    def getNoZeroIntegers(self, n):
        if False:
            return 10
        '\n        :type n: int\n        :rtype: List[int]\n        '
        (a, curr, base) = (0, n, 1)
        while curr:
            if curr % 10 == 0 or (curr % 10 == 1 and curr != 1):
                a += base
                curr -= 10
            a += base
            base *= 10
            curr //= 10
        return [a, n - a]

class Solution2(object):

    def getNoZeroIntegers(self, n):
        if False:
            i = 10
            return i + 15
        '\n        :type n: int\n        :rtype: List[int]\n        '
        return next(([a, n - a] for a in xrange(1, n) if '0' not in '{}{}'.format(a, n - a)))