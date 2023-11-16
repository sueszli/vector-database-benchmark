class Solution(object):

    def countPartitions(self, nums, k):
        if False:
            return 10
        '\n        :type nums: List[int]\n        :type k: int\n        :rtype: int\n        '
        MOD = 10 ** 9 + 7
        if sum(nums) < 2 * k:
            return 0
        dp = [0] * k
        dp[0] = 1
        for x in nums:
            for i in reversed(xrange(k - x)):
                dp[i + x] = (dp[i + x] + dp[i]) % MOD
        return (pow(2, len(nums), MOD) - 2 * reduce(lambda total, x: (total + x) % MOD, dp, 0)) % MOD