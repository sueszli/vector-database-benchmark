class Solution(object):

    def goodIndices(self, nums, k):
        if False:
            return 10
        '\n        :type nums: List[int]\n        :type k: int\n        :rtype: List[int]\n        '
        left = [1] * len(nums)
        for i in xrange(1, len(nums) - 1):
            if nums[i] <= nums[i - 1]:
                left[i] = left[i - 1] + 1
        right = [1] * len(nums)
        for i in reversed(xrange(1, len(nums) - 1)):
            if nums[i] <= nums[i + 1]:
                right[i] = right[i + 1] + 1
        return [i for i in xrange(k, len(nums) - k) if min(left[i - 1], right[i + 1]) >= k]