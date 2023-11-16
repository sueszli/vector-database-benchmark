class Solution(object):

    def minFallingPathSum(self, A):
        if False:
            return 10
        '\n        :type A: List[List[int]]\n        :rtype: int\n        '
        for i in xrange(1, len(A)):
            for j in xrange(len(A[i])):
                A[i][j] += min(A[i - 1][max(j - 1, 0):j + 2])
        return min(A[-1])