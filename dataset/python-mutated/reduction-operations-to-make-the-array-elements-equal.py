class Solution(object):

    def reductionOperations(self, nums):
        if False:
            while True:
                i = 10
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        nums.sort()
        result = curr = 0
        for i in xrange(1, len(nums)):
            if nums[i - 1] < nums[i]:
                curr += 1
            result += curr
        return result