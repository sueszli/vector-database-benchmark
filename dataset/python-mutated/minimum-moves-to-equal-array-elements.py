class Solution(object):

    def minMoves(self, nums):
        if False:
            i = 10
            return i + 15
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        return sum(nums) - len(nums) * min(nums)