class Solution(object):

    def decompressRLElist(self, nums):
        if False:
            return 10
        '\n        :type nums: List[int]\n        :rtype: List[int]\n        '
        return [nums[i + 1] for i in xrange(0, len(nums), 2) for _ in xrange(nums[i])]