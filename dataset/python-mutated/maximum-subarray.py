class Solution(object):

    def maxSubArray(self, nums):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        (result, curr) = (float('-inf'), float('-inf'))
        for x in nums:
            curr = max(curr + x, x)
            result = max(result, curr)
        return result