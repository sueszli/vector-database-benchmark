class Solution(object):

    def minScoreTriangulation(self, A):
        if False:
            return 10
        '\n        :type A: List[int]\n        :rtype: int\n        '
        dp = [[0 for _ in xrange(len(A))] for _ in xrange(len(A))]
        for p in xrange(3, len(A) + 1):
            for i in xrange(len(A) - p + 1):
                j = i + p - 1
                dp[i][j] = float('inf')
                for k in xrange(i + 1, j):
                    dp[i][j] = min(dp[i][j], dp[i][k] + dp[k][j] + A[i] * A[j] * A[k])
        return dp[0][-1]