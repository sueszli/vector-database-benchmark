class Solution(object):

    def maxAlternatingSum(self, nums):
        if False:
            print('Hello World!')
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        result = nums[0]
        for i in xrange(len(nums) - 1):
            result += max(nums[i + 1] - nums[i], 0)
        return result