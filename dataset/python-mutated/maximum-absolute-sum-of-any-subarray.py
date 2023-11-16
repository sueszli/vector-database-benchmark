class Solution(object):

    def maxAbsoluteSum(self, nums):
        if False:
            i = 10
            return i + 15
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        curr = mx = mn = 0
        for num in nums:
            curr += num
            mx = max(mx, curr)
            mn = min(mn, curr)
        return mx - mn