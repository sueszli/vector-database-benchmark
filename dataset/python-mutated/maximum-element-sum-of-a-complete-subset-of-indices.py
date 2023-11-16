class Solution(object):

    def maximumSum(self, nums):
        if False:
            return 10
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        return max((sum((nums[i * x ** 2 - 1] for x in xrange(1, int((len(nums) // i) ** 0.5) + 1))) for i in xrange(1, len(nums) + 1)))