class Solution(object):

    def maxScoreIndices(self, nums):
        if False:
            return 10
        '\n        :type nums: List[int]\n        :rtype: List[int]\n        '
        result = []
        mx = zeros = 0
        total = sum(nums)
        for i in xrange(len(nums) + 1):
            zeros += (nums[i - 1] if i else 0) == 0
            if zeros + (total - (i - zeros)) > mx:
                mx = zeros + (total - (i - zeros))
                result = []
            if zeros + (total - (i - zeros)) == mx:
                result.append(i)
        return result