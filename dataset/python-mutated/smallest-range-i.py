class Solution(object):

    def smallestRangeI(self, A, K):
        if False:
            return 10
        '\n        :type A: List[int]\n        :type K: int\n        :rtype: int\n        '
        return max(0, max(A) - min(A) - 2 * K)