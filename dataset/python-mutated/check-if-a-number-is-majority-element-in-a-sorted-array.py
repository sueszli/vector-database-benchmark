import bisect

class Solution(object):

    def isMajorityElement(self, nums, target):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type nums: List[int]\n        :type target: int\n        :rtype: bool\n        '
        if len(nums) % 2:
            if nums[len(nums) // 2] != target:
                return False
        elif not nums[len(nums) // 2 - 1] == nums[len(nums) // 2] == target:
            return False
        left = bisect.bisect_left(nums, target)
        right = bisect.bisect_right(nums, target)
        return (right - left) * 2 > len(nums)