import bisect

class Solution(object):

    def maximumCount(self, nums):
        if False:
            return 10
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        return max(bisect.bisect_left(nums, 0) - 0, len(nums) - bisect.bisect_left(nums, 1))