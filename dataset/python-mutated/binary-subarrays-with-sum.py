class Solution(object):

    def numSubarraysWithSum(self, A, S):
        if False:
            i = 10
            return i + 15
        '\n        :type A: List[int]\n        :type S: int\n        :rtype: int\n        '
        result = 0
        (left, right, sum_left, sum_right) = (0, 0, 0, 0)
        for (i, a) in enumerate(A):
            sum_left += a
            while left < i and sum_left > S:
                sum_left -= A[left]
                left += 1
            sum_right += a
            while right < i and (sum_right > S or (sum_right == S and (not A[right]))):
                sum_right -= A[right]
                right += 1
            if sum_left == S:
                result += right - left + 1
        return result