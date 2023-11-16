class Solution(object):

    def minSubsequence(self, nums):
        if False:
            i = 10
            return i + 15
        '\n        :type nums: List[int]\n        :rtype: List[int]\n        '
        (result, total, curr) = ([], sum(nums), 0)
        nums.sort(reverse=True)
        for (i, x) in enumerate(nums):
            curr += x
            if curr > total - curr:
                break
        return nums[:i + 1]