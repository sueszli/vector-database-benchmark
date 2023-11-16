class Solution(object):

    def countPrimeSetBits(self, L, R):
        if False:
            print('Hello World!')
        '\n        :type L: int\n        :type R: int\n        :rtype: int\n        '

        def bitCount(n):
            if False:
                return 10
            result = 0
            while n:
                n &= n - 1
                result += 1
            return result
        primes = {2, 3, 5, 7, 11, 13, 17, 19}
        return sum((bitCount(i) in primes for i in xrange(L, R + 1)))