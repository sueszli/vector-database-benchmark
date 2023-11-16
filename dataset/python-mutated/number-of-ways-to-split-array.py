class Solution(object):

    def waysToSplitArray(self, nums):
        if False:
            print('Hello World!')
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        total = sum(nums)
        result = curr = 0
        for i in xrange(len(nums) - 1):
            curr += nums[i]
            result += int(curr >= total - curr)
        return result