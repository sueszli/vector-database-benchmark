class Solution(object):

    def findNumberOfLIS(self, nums):
        if False:
            return 10
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        (result, max_len) = (0, 0)
        dp = [[1, 1] for _ in xrange(len(nums))]
        for i in xrange(len(nums)):
            for j in xrange(i):
                if nums[i] > nums[j]:
                    if dp[i][0] == dp[j][0] + 1:
                        dp[i][1] += dp[j][1]
                    elif dp[i][0] < dp[j][0] + 1:
                        dp[i] = [dp[j][0] + 1, dp[j][1]]
            if max_len == dp[i][0]:
                result += dp[i][1]
            elif max_len < dp[i][0]:
                max_len = dp[i][0]
                result = dp[i][1]
        return result