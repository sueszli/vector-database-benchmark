class Solution(object):

    def uniquePaths(self, m, n):
        if False:
            return 10
        '\n        :type m: int\n        :type n: int\n        :rtype: int\n        '

        def nCr(n, r):
            if False:
                for i in range(10):
                    print('nop')
            if n - r < r:
                r = n - r
            c = 1
            for k in xrange(1, r + 1):
                c *= n - k + 1
                c //= k
            return c
        return nCr(m - 1 + (n - 1), n - 1)

class Solution2(object):

    def uniquePaths(self, m, n):
        if False:
            while True:
                i = 10
        '\n        :type m: int\n        :type n: int\n        :rtype: int\n        '
        if m < n:
            (m, n) = (n, m)
        dp = [1] * n
        for i in xrange(1, m):
            for j in xrange(1, n):
                dp[j] += dp[j - 1]
        return dp[n - 1]