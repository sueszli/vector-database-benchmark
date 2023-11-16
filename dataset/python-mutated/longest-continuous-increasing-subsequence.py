class Solution(object):

    def findLengthOfLCIS(self, nums):
        if False:
            i = 10
            return i + 15
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        (result, count) = (0, 0)
        for i in xrange(len(nums)):
            if i == 0 or nums[i - 1] < nums[i]:
                count += 1
                result = max(result, count)
            else:
                count = 1
        return result