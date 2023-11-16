class Solution(object):

    def maxSumDivThree(self, nums):
        if False:
            i = 10
            return i + 15
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        dp = [0, 0, 0]
        for num in nums:
            for i in [num + x for x in dp]:
                dp[i % 3] = max(dp[i % 3], i)
        return dp[0]