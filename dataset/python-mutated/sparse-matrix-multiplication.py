class Solution(object):

    def multiply(self, A, B):
        if False:
            print('Hello World!')
        '\n        :type A: List[List[int]]\n        :type B: List[List[int]]\n        :rtype: List[List[int]]\n        '
        (m, n, l) = (len(A), len(A[0]), len(B[0]))
        res = [[0 for _ in xrange(l)] for _ in xrange(m)]
        for i in xrange(m):
            for k in xrange(n):
                if A[i][k]:
                    for j in xrange(l):
                        res[i][j] += A[i][k] * B[k][j]
        return res