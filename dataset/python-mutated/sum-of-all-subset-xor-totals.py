class Solution(object):

    def subsetXORSum(self, nums):
        if False:
            print('Hello World!')
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        result = 0
        for x in nums:
            result |= x
        return result * 2 ** (len(nums) - 1)