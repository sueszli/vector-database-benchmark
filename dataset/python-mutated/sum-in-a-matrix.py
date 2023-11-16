class Solution(object):

    def matrixSum(self, nums):
        if False:
            while True:
                i = 10
        '\n        :type nums: List[List[int]]\n        :rtype: int\n        '
        for row in nums:
            row.sort()
        return sum((max((nums[r][c] for r in xrange(len(nums)))) for c in xrange(len(nums[0]))))