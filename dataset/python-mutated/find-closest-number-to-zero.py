class Solution(object):

    def findClosestNumber(self, nums):
        if False:
            return 10
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        return max(nums, key=lambda x: (-abs(x), x))