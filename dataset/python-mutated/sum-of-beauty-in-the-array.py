class Solution(object):

    def sumOfBeauties(self, nums):
        if False:
            i = 10
            return i + 15
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        right = [nums[-1]] * len(nums)
        for i in reversed(xrange(2, len(nums) - 1)):
            right[i] = min(right[i + 1], nums[i])
        (result, left) = (0, nums[0])
        for i in xrange(1, len(nums) - 1):
            if left < nums[i] < right[i + 1]:
                result += 2
            elif nums[i - 1] < nums[i] < nums[i + 1]:
                result += 1
            left = max(left, nums[i])
        return result