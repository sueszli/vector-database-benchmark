import math

class Solution(object):

    def getPermutation(self, n, k):
        if False:
            while True:
                i = 10
        '\n        :type n: int\n        :type k: int\n        :rtype: str\n        '
        (seq, k, fact) = ('', k - 1, math.factorial(n - 1))
        perm = [i for i in xrange(1, n + 1)]
        for i in reversed(xrange(n)):
            curr = perm[k / fact]
            seq += str(curr)
            perm.remove(curr)
            if i > 0:
                k %= fact
                fact /= i
        return seq