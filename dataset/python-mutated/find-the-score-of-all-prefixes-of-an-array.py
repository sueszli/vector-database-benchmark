class Solution(object):

    def findPrefixScore(self, nums):
        if False:
            while True:
                i = 10
        '\n        :type nums: List[int]\n        :rtype: List[int]\n        '
        curr = 0
        for i in xrange(len(nums)):
            curr = max(curr, nums[i])
            nums[i] += (nums[i - 1] if i - 1 >= 0 else 0) + curr
        return nums