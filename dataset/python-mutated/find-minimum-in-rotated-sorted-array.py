class Solution(object):

    def findMin(self, nums):
        if False:
            return 10
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        (left, right) = (0, len(nums))
        target = nums[-1]
        while left < right:
            mid = left + (right - left) / 2
            if nums[mid] <= target:
                right = mid
            else:
                left = mid + 1
        return nums[left]

class Solution2(object):

    def findMin(self, nums):
        if False:
            i = 10
            return i + 15
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        (left, right) = (0, len(nums) - 1)
        while left < right and nums[left] >= nums[right]:
            mid = left + (right - left) / 2
            if nums[mid] < nums[left]:
                right = mid
            else:
                left = mid + 1
        return nums[left]