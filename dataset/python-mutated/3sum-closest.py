class Solution(object):

    def threeSumClosest(self, nums, target):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type nums: List[int]\n        :type target: int\n        :rtype: int\n        '
        (result, min_diff) = (0, float('inf'))
        nums.sort()
        for i in reversed(xrange(2, len(nums))):
            if i + 1 < len(nums) and nums[i] == nums[i + 1]:
                continue
            (left, right) = (0, i - 1)
            while left < right:
                total = nums[left] + nums[right] + nums[i]
                if total < target:
                    left += 1
                elif total > target:
                    right -= 1
                else:
                    return target
                if abs(total - target) < min_diff:
                    min_diff = abs(total - target)
                    result = total
        return result