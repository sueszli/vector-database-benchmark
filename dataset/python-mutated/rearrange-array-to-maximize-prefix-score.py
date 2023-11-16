class Solution(object):

    def maxScore(self, nums):
        if False:
            return 10
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        nums.sort(reverse=True)
        curr = 0
        for (i, x) in enumerate(nums):
            curr += x
            if curr <= 0:
                return i
        return len(nums)