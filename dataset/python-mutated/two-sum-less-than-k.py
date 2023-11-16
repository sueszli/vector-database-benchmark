class Solution(object):

    def twoSumLessThanK(self, A, K):
        if False:
            return 10
        '\n        :type A: List[int]\n        :type K: int\n        :rtype: int\n        '
        A.sort()
        result = -1
        (left, right) = (0, len(A) - 1)
        while left < right:
            if A[left] + A[right] >= K:
                right -= 1
            else:
                result = max(result, A[left] + A[right])
                left += 1
        return result