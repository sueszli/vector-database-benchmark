class Solution(object):

    def maxSumMinProduct(self, nums):
        if False:
            while True:
                i = 10
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        MOD = 10 ** 9 + 7
        prefix = [0] * (len(nums) + 1)
        for i in xrange(len(nums)):
            prefix[i + 1] = prefix[i] + nums[i]
        (stk, result) = ([-1], 0)
        for i in xrange(len(nums) + 1):
            while stk[-1] != -1 and (i == len(nums) or nums[stk[-1]] >= nums[i]):
                result = max(result, nums[stk.pop()] * (prefix[i - 1 + 1] - prefix[stk[-1] + 1]))
            stk.append(i)
        return result % MOD