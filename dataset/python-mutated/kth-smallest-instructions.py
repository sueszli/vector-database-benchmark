class Solution(object):

    def kthSmallestPath(self, destination, k):
        if False:
            print('Hello World!')
        '\n        :type destination: List[int]\n        :type k: int\n        :rtype: str\n        '

        def nCr(n, r):
            if False:
                return 10
            if n < r:
                return 0
            if n - r < r:
                return nCr(n, n - r)
            c = 1
            for k in xrange(1, r + 1):
                c *= n - k + 1
                c //= k
            return c
        (r, c) = destination
        result = []
        while r + c:
            count = nCr(r + (c - 1), r)
            if k <= count:
                c -= 1
                result.append('H')
            else:
                k -= count
                r -= 1
                result.append('V')
        return ''.join(result)