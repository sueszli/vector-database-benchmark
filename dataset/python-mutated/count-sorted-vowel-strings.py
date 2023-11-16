class Solution(object):

    def countVowelStrings(self, n):
        if False:
            return 10
        '\n        :type n: int\n        :rtype: int\n        '

        def nCr(n, r):
            if False:
                for i in range(10):
                    print('nop')
            if n - r < r:
                return nCr(n, n - r)
            c = 1
            for k in xrange(1, r + 1):
                c *= n - k + 1
                c //= k
            return c
        return nCr(n + 4, 4)