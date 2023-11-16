class Solution(object):

    def numberOfGoodSubarraySplits(self, nums):
        if False:
            return 10
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        MOD = 10 ** 9 + 7
        (result, prev) = (1, -1)
        for i in xrange(len(nums)):
            if nums[i] != 1:
                continue
            if prev != -1:
                result = result * (i - prev) % MOD
            prev = i
        return result if prev != -1 else 0