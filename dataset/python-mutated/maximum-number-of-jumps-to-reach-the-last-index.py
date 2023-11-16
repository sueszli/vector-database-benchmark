class Solution(object):

    def maximumJumps(self, nums, target):
        if False:
            return 10
        '\n        :type nums: List[int]\n        :type target: int\n        :rtype: int\n        '
        dp = [-1] * len(nums)
        dp[0] = 0
        for i in xrange(1, len(nums)):
            for j in xrange(i):
                if abs(nums[i] - nums[j]) <= target:
                    if dp[j] != -1:
                        dp[i] = max(dp[i], dp[j] + 1)
        return dp[-1]