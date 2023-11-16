import collections

class Solution(object):

    def minCost(self, nums, k):
        if False:
            print('Hello World!')
        '\n        :type nums: List[int]\n        :type k: int\n        :rtype: int\n        '
        dp = [float('inf')] * (len(nums) + 1)
        dp[0] = 0
        for i in xrange(len(dp) - 1):
            cnt = [0] * len(nums)
            d = 0
            for j in xrange(i + 1, len(dp)):
                cnt[nums[j - 1]] += 1
                if cnt[nums[j - 1]] == 1:
                    d += 1
                elif cnt[nums[j - 1]] == 2:
                    d -= 1
                dp[j] = min(dp[j], dp[i] + k + (j - i - d))
        return dp[-1]