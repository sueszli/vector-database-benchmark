class Solution(object):

    def minArrayLength(self, nums, k):
        if False:
            i = 10
            return i + 15
        '\n        :type nums: List[int]\n        :type k: int\n        :rtype: int\n        '
        if 0 in nums:
            return 1
        result = len(nums)
        curr = nums[0]
        for i in xrange(1, len(nums)):
            if curr * nums[i] > k:
                curr = nums[i]
            else:
                curr *= nums[i]
                result -= 1
        return result