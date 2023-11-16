class Solution(object):

    def maxArrayValue(self, nums):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        result = curr = 0
        for i in reversed(xrange(len(nums))):
            if nums[i] > curr:
                curr = 0
            curr += nums[i]
            result = max(result, curr)
        return result