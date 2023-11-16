class Solution(object):

    def maximumSumScore(self, nums):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        prefix = suffix = 0
        result = float('-inf')
        right = len(nums) - 1
        for left in xrange(len(nums)):
            prefix += nums[left]
            suffix += nums[right]
            right -= 1
            result = max(result, prefix, suffix)
        return result

class Solution2(object):

    def maximumSumScore(self, nums):
        if False:
            i = 10
            return i + 15
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        total = sum(nums)
        prefix = 0
        result = float('-inf')
        for x in nums:
            prefix += x
            result = max(result, prefix, total - prefix + x)
        return result