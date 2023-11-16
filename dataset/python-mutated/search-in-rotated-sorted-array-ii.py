class Solution(object):

    def search(self, nums, target):
        if False:
            while True:
                i = 10
        '\n        :type nums: List[int]\n        :type target: int\n        :rtype: int\n        '
        (left, right) = (0, len(nums) - 1)
        while left <= right:
            mid = left + (right - left) / 2
            if nums[mid] == target:
                return True
            elif nums[mid] == nums[left]:
                left += 1
            elif nums[mid] > nums[left] and nums[left] <= target < nums[mid] or (nums[mid] < nums[left] and (not nums[mid] < target <= nums[right])):
                right = mid - 1
            else:
                left = mid + 1
        return False