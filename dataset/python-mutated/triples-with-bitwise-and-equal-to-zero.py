import collections

class Solution(object):

    def countTriplets(self, A):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type A: List[int]\n        :rtype: int\n        '

        def FWT(A, v):
            if False:
                return 10
            B = A[:]
            d = 1
            while d < len(B):
                for i in xrange(0, len(B), d << 1):
                    for j in xrange(d):
                        B[i + j] += B[i + j + d] * v
                d <<= 1
            return B
        k = 3
        (n, max_A) = (1, max(A))
        while n <= max_A:
            n *= 2
        count = collections.Counter(A)
        B = [count[i] for i in xrange(n)]
        C = FWT(map(lambda x: x ** k, FWT(B, 1)), -1)
        return C[0]
import collections

class Solution2(object):

    def countTriplets(self, A):
        if False:
            print('Hello World!')
        '\n        :type A: List[int]\n        :rtype: int\n        '
        count = collections.defaultdict(int)
        for i in xrange(len(A)):
            for j in xrange(len(A)):
                count[A[i] & A[j]] += 1
        result = 0
        for k in xrange(len(A)):
            for v in count:
                if A[k] & v == 0:
                    result += count[v]
        return result