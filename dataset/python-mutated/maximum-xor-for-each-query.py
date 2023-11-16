class Solution(object):

    def getMaximumXor(self, nums, maximumBit):
        if False:
            print('Hello World!')
        '\n        :type nums: List[int]\n        :type maximumBit: int\n        :rtype: List[int]\n        '
        result = [0] * len(nums)
        mask = 2 ** maximumBit - 1
        for i in xrange(len(nums)):
            mask ^= nums[i]
            result[-1 - i] = mask
        return result