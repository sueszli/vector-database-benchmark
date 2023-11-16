class Solution(object):

    def maximumScore(self, nums, multipliers):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type nums: List[int]\n        :type multipliers: List[int]\n        :rtype: int\n        '
        dp = [0] * (len(multipliers) + 1)
        for (l, m) in enumerate(reversed(multipliers), start=len(nums) - len(multipliers)):
            dp = [max(m * nums[i] + dp[i + 1], m * nums[i + l] + dp[i]) for i in xrange(len(dp) - 1)]
        return dp[0]