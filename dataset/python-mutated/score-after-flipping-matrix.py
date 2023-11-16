class Solution(object):

    def matrixScore(self, A):
        if False:
            while True:
                i = 10
        '\n        :type A: List[List[int]]\n        :rtype: int\n        '
        (R, C) = (len(A), len(A[0]))
        result = 0
        for c in xrange(C):
            col = 0
            for r in xrange(R):
                col += A[r][c] ^ A[r][0]
            result += max(col, R - col) * 2 ** (C - 1 - c)
        return result