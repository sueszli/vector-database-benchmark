class Solution(object):

    def minPairSum(self, nums):
        if False:
            i = 10
            return i + 15
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        nums.sort()
        return max((nums[i] + nums[-1 - i] for i in xrange(len(nums) // 2)))