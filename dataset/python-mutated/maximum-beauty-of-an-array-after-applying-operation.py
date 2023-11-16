class Solution(object):

    def maximumBeauty(self, nums, k):
        if False:
            while True:
                i = 10
        '\n        :type nums: List[int]\n        :type k: int\n        :rtype: int\n        '
        nums.sort()
        left = 0
        for right in xrange(len(nums)):
            if nums[right] - nums[left] > k * 2:
                left += 1
        return right - left + 1