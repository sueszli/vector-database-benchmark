class Solution(object):

    def countSpecialSubsequences(self, nums):
        if False:
            while True:
                i = 10
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        MOD = 10 ** 9 + 7
        dp = [0] * 3
        for x in nums:
            dp[x] = ((dp[x] + dp[x]) % MOD + (dp[x - 1] if x - 1 >= 0 else 1)) % MOD
        return dp[-1]