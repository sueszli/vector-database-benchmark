class Solution(object):

    def minimumLines(self, stockPrices):
        if False:
            print('Hello World!')
        '\n        :type stockPrices: List[List[int]]\n        :rtype: int\n        '

        def gcd(a, b):
            if False:
                for i in range(10):
                    print('nop')
            while b:
                (a, b) = (b, a % b)
            return a
        stockPrices.sort()
        result = 0
        prev = None
        for i in xrange(1, len(stockPrices)):
            (dy, dx) = (stockPrices[i][1] - stockPrices[i - 1][1], stockPrices[i][0] - stockPrices[i - 1][0])
            g = gcd(dy, dx)
            if not prev or prev != (dy // g, dx // g):
                prev = (dy // g, dx // g)
                result += 1
        return result