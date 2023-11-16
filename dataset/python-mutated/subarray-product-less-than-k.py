class Solution(object):

    def numSubarrayProductLessThanK(self, nums, k):
        if False:
            print('Hello World!')
        '\n        :type nums: List[int]\n        :type k: int\n        :rtype: int\n        '
        if k <= 1:
            return 0
        (result, start, prod) = (0, 0, 1)
        for (i, num) in enumerate(nums):
            prod *= num
            while prod >= k:
                prod /= nums[start]
                start += 1
            result += i - start + 1
        return result