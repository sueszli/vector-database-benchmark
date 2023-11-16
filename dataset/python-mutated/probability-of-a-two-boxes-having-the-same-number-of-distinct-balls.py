import collections

class Solution(object):

    def getProbability(self, balls):
        if False:
            print('Hello World!')
        '\n        :type balls: List[int]\n        :rtype: float\n        '

        def nCrs(n):
            if False:
                i = 10
                return i + 15
            c = 1
            for k in xrange(n + 1):
                yield c
                c *= n - (k + 1) + 1
                c //= k + 1

        def nCr(n, r):
            if False:
                return 10
            if n - r < r:
                return nCr(n, n - r)
            c = 1
            for k in xrange(1, r + 1):
                c *= n - k + 1
                c //= k
            return c
        dp = collections.defaultdict(int)
        dp[0, 0] = 1
        for n in balls:
            new_dp = collections.defaultdict(int)
            for ((ndiff, cdiff), count) in dp.iteritems():
                for (k, new_count) in enumerate(nCrs(n)):
                    new_ndiff = ndiff + (k - (n - k))
                    new_cdiff = cdiff - 1 if k == 0 else cdiff + 1 if k == n else cdiff
                    new_dp[new_ndiff, new_cdiff] += count * new_count
            dp = new_dp
        total = sum(balls)
        return float(dp[0, 0]) / nCr(total, total // 2)