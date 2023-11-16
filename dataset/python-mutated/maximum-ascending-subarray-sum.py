class Solution(object):

    def maxAscendingSum(self, nums):
        if False:
            while True:
                i = 10
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        result = curr = 0
        for i in xrange(len(nums)):
            if not (i and nums[i - 1] < nums[i]):
                curr = 0
            curr += nums[i]
            result = max(result, curr)
        return result