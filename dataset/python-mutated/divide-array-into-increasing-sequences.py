class Solution(object):

    def canDivideIntoSubsequences(self, nums, K):
        if False:
            i = 10
            return i + 15
        '\n        :type nums: List[int]\n        :type K: int\n        :rtype: bool\n        '
        (curr, max_count) = (1, 1)
        for i in xrange(1, len(nums)):
            curr = 1 if nums[i - 1] < nums[i] else curr + 1
            max_count = max(max_count, curr)
        return K * max_count <= len(nums)