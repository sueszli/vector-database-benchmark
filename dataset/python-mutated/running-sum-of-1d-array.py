class Solution(object):

    def runningSum(self, nums):
        if False:
            i = 10
            return i + 15
        '\n        :type nums: List[int]\n        :rtype: List[int]\n        '
        for i in xrange(len(nums) - 1):
            nums[i + 1] += nums[i]
        return nums