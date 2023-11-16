class Solution(object):

    def combinationSum4(self, nums, target):
        if False:
            i = 10
            return i + 15
        '\n        :type nums: List[int]\n        :type target: int\n        :rtype: int\n        '
        dp = [0] * (target + 1)
        dp[0] = 1
        nums.sort()
        for i in xrange(1, target + 1):
            for j in xrange(len(nums)):
                if nums[j] <= i:
                    dp[i] += dp[i - nums[j]]
                else:
                    break
        return dp[target]