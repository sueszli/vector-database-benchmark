class Solution(object):

    def maximumOr(self, nums, k):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type nums: List[int]\n        :type k: int\n        :rtype: int\n        '
        right = [0] * (len(nums) + 1)
        for i in reversed(xrange(len(nums))):
            right[i] = right[i + 1] | nums[i]
        result = left = 0
        for i in xrange(len(nums)):
            result = max(result, left | nums[i] << k | right[i + 1])
            left |= nums[i]
        return result