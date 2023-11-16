class Solution(object):

    def PredictTheWinner(self, nums):
        if False:
            while True:
                i = 10
        '\n        :type nums: List[int]\n        :rtype: bool\n        '
        if len(nums) % 2 == 0 or len(nums) == 1:
            return True
        dp = [0] * len(nums)
        for i in reversed(xrange(len(nums))):
            dp[i] = nums[i]
            for j in xrange(i + 1, len(nums)):
                dp[j] = max(nums[i] - dp[j], nums[j] - dp[j - 1])
        return dp[-1] >= 0