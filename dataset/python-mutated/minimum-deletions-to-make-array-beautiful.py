class Solution(object):

    def minDeletion(self, nums):
        if False:
            return 10
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        result = 0
        for i in xrange(len(nums) - 1):
            result += int(i % 2 == result % 2 and nums[i] == nums[i + 1])
        return result + (len(nums) - result) % 2