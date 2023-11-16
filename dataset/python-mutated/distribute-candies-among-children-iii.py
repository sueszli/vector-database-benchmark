class Solution(object):

    def distributeCandies(self, n, limit):
        if False:
            return 10
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
                i = 10
                return i + 15
            return nCr(n + (r - 1), r - 1)
        R = 3
        return sum(((-1 if r % 2 else 1) * nCr(R, r) * nHr(n - r * (limit + 1), R) for r in xrange(R + 1)))