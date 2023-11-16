import collections

class Solution(object):

    def longestArithSeqLength(self, A):
        if False:
            i = 10
            return i + 15
        '\n        :type A: List[int]\n        :rtype: int\n        '
        dp = collections.defaultdict(int)
        for i in xrange(len(A) - 1):
            for j in xrange(i + 1, len(A)):
                v = A[j] - A[i]
                dp[v, j] = max(dp[v, j], dp[v, i] + 1)
        return max(dp.values()) + 1