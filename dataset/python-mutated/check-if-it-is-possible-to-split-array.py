class Solution(object):

    def canSplitArray(self, nums, m):
        if False:
            while True:
                i = 10
        '\n        :type nums: List[int]\n        :type m: int\n        :rtype: bool\n        '
        return len(nums) <= 2 or any((nums[i] + nums[i + 1] >= m for i in xrange(len(nums) - 1)))