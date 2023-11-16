import collections

class Solution(object):

    def countMaxOrSubsets(self, nums):
        if False:
            print('Hello World!')
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        dp = collections.Counter([0])
        for x in nums:
            for (k, v) in dp.items():
                dp[k | x] += v
        return dp[reduce(lambda x, y: x | y, nums)]