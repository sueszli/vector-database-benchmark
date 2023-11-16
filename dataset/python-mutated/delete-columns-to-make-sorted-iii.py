class Solution(object):

    def minDeletionSize(self, A):
        if False:
            while True:
                i = 10
        '\n        :type A: List[str]\n        :rtype: int\n        '
        dp = [1] * len(A[0])
        for j in xrange(1, len(A[0])):
            for i in xrange(j):
                if all((A[k][i] <= A[k][j] for k in xrange(len(A)))):
                    dp[j] = max(dp[j], dp[i] + 1)
        return len(A[0]) - max(dp)