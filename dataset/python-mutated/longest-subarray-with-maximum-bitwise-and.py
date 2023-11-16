class Solution(object):

    def longestSubarray(self, nums):
        if False:
            while True:
                i = 10
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        mx = max(nums)
        (result, l) = (1, 0)
        for x in nums:
            if x == mx:
                l += 1
                result = max(result, l)
            else:
                l = 0
        return result