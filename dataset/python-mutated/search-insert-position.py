class Solution(object):

    def searchInsert(self, nums, target):
        if False:
            return 10
        '\n        :type nums: List[int]\n        :type target: int\n        :rtype: int\n        '
        (left, right) = (0, len(nums) - 1)
        while left <= right:
            mid = left + (right - left) / 2
            if nums[mid] >= target:
                right = mid - 1
            else:
                left = mid + 1
        return left