class Solution(object):

    def smallestRangeII(self, A, K):
        if False:
            return 10
        '\n        :type A: List[int]\n        :type K: int\n        :rtype: int\n        '
        A.sort()
        result = A[-1] - A[0]
        for i in xrange(len(A) - 1):
            result = min(result, max(A[-1] - K, A[i] + K) - min(A[0] + K, A[i + 1] - K))
        return result