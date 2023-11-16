class Solution(object):

    def sumSubseqWidths(self, A):
        if False:
            while True:
                i = 10
        '\n        :type A: List[int]\n        :rtype: int\n        '
        M = 10 ** 9 + 7
        (result, c) = (0, 1)
        A.sort()
        for i in xrange(len(A)):
            result = (result + (A[i] - A[len(A) - 1 - i]) * c % M) % M
            c = (c << 1) % M
        return result