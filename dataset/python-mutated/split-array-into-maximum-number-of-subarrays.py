class Solution(object):

    def maxSubarrays(self, nums):
        if False:
            while True:
                i = 10
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        result = curr = 0
        for x in nums:
            curr = curr & x if curr else x
            if not curr:
                result += 1
        return max(result, 1)