class Solution(object):

    def minImpossibleOR(self, nums):
        if False:
            print('Hello World!')
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        lookup = set(nums)
        return next((1 << i for i in xrange(31) if 1 << i not in lookup))