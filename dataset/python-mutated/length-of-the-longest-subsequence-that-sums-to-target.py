class Solution(object):

    def lengthOfLongestSubsequence(self, nums, target):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type nums: List[int]\n        :type target: int\n        :rtype: int\n        '
        dp = [-1] * (target + 1)
        dp[0] = 0
        for x in nums:
            for i in reversed(xrange(x, len(dp))):
                if dp[i - x] != -1:
                    dp[i] = max(dp[i], dp[i - x] + 1)
        return dp[-1]