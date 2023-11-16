class Solution(object):

    def distinctDifferenceArray(self, nums):
        if False:
            return 10
        '\n        :type nums: List[int]\n        :rtype: List[int]\n        '
        result = [0] * len(nums)
        lookup = set()
        for i in xrange(len(nums)):
            lookup.add(nums[i])
            result[i] = len(lookup)
        lookup.clear()
        for i in reversed(xrange(len(nums))):
            result[i] -= len(lookup)
            lookup.add(nums[i])
        return result