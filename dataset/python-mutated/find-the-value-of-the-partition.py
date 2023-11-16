class Solution(object):

    def findValueOfPartition(self, nums):
        if False:
            while True:
                i = 10
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        nums.sort()
        return min((nums[i + 1] - nums[i] for i in xrange(len(nums) - 1)))