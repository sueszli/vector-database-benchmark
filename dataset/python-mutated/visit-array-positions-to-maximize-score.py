class Solution(object):

    def maxScore(self, nums, x):
        if False:
            print('Hello World!')
        '\n        :type nums: List[int]\n        :type x: int\n        :rtype: int\n        '
        dp = [float('-inf')] * 2
        dp[nums[0] % 2] = nums[0]
        for i in xrange(1, len(nums)):
            dp[nums[i] % 2] = max(dp[nums[i] % 2], dp[(nums[i] + 1) % 2] - x) + nums[i]
        return max(dp)