class Solution(object):

    def maximizeSum(self, nums, k):
        if False:
            i = 10
            return i + 15
        '\n        :type nums: List[int]\n        :type k: int\n        :rtype: int\n        '
        return max(nums) * k + k * (k - 1) // 2