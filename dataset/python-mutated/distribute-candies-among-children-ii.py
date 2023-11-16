class Solution(object):

    def distributeCandies(self, n, limit):
        if False:
            i = 10
            return i + 15
        '\n        :type n: int\n        :type limit: int\n        :rtype: int\n        '

        def nCr(n, r):
            if False:
                return 10
            if not 0 <= r <= n:
                return 0
            if n - r < r:
                r = n - r
            c = 1
            for k in xrange(1, r + 1):
                c *= n - k + 1
                c //= k
            return c

        def nHr(n, r):
            if False:
                return 10
            return nCr(n + (r - 1), r - 1)
        R = 3
        return sum(((-1 if r % 2 else 1) * nCr(R, r) * nHr(n - r * (limit + 1), R) for r in xrange(R + 1)))

class Solution2(object):

    def distributeCandies(self, n, limit):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type n: int\n        :type limit: int\n        :rtype: int\n        '
        return sum((min(limit, n - i) - max(n - i - limit, 0) + 1 for i in xrange(max(n - 2 * limit, 0), min(limit, n) + 1)))