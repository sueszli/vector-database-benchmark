class Solution(object):

    def minimizeSum(self, nums):
        if False:
            return 10
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        nums.sort()
        return min((nums[-3 + i] - nums[i] for i in xrange(3)))